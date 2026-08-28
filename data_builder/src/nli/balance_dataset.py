import argparse
import logging
import math
import re
import uuid
from typing import Optional
from warnings import simplefilter

import pandas as pd
import tqdm

from article_utils.util import dump_jsonl, read_jsonl

from .claim_bias import compute_lmi_fast, get_ngrams
from .util import PARTITION_COL, concat_text

PARTITIONS = ['train', 'dev', 'test']
DOC_PARTITION_COL = 'pmid.partition'
MISC_GROUP_NAME = 'MISC'
MAX_TOP_TERMS = 200
NUM_LABELS = 3
PREF_DECAY = 0.5
DATE_CUTOFF = '2026-03-03'
# https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def percent(value):
    value = int(value)
    if value < 0 or value > 100:
        raise argparse.ArgumentTypeError('value must be an integer between 0 and 100')
    return value


def has_review(reviews):
    for review in reviews:
        if review.get('round', '') != '' and review.get('username', 'bot') != 'bot':
            return True
    return False


def preference_order(row):
    if 'pmc' in row['evidence.source'].lower():
        ev_pref = 0
    elif 'pubmed' in row['evidence.source'].lower():
        ev_pref = 1
    else:
        ev_pref = 2

    if 'hypothesis' == row['evidence.selection_method'].lower():
        sel_pref = 0
    else:
        sel_pref = 1

    pref = (
        1 if row['_exclude'] else 0,
        0 if row['_manual'] else 1,
        # civic primarily catelogues somatic content, therefore we should focus on that as well
        1 if 'germline' in row['claim.flat'].lower() else 0,
        # prefer evidence sources directly from articles vs using CIViC Evidence Items descriptions
        ev_pref,
        # prefer manually selected evidence
        sel_pref,
    )
    pref = [1 / (p + 1) for p in pref]
    weights = [1 * math.pow(PREF_DECAY, i) for i in range(len(pref))]
    return sum([p * w for p, w in zip(pref, weights)]) / sum(weights)


def random_rebalancing_partition(
    input_df,
    total_max_per_label_count: int,
    initial_row_selections: Optional[set[int]],
    iterations=10000,
) -> set[int]:
    """
    Make random selections for each class to balance the labels up to the required amount
    """
    for req_col in ['bigram', 'row_index', '_prefer', '_exclude', 'gold_label_name']:
        assert req_col in input_df.columns, f'missing required column: {req_col}'

    if initial_row_selections is None:
        initial_row_selections = set()
    df = input_df[input_df._exclude == False].copy()
    assert df.row_index.nunique() == df.shape[0], 'row_index is not a unique index'

    rows = []

    initial_counts = dict(
        df[df.row_index.isin(initial_row_selections)].gold_label_name.value_counts()
    )
    print(initial_counts)
    for label, label_count in initial_counts.items():
        if label_count > total_max_per_label_count:
            raise AssertionError(
                f'Label ({label}) has already assigned ({label_count}) more than the total number expected ({total_max_per_label_count}) may fail to balance well'
            )

    logging.info(
        f'selecting {sum([total_max_per_label_count - v for v in initial_counts.values()])} rows'
    )
    for boot_i in tqdm.tqdm(list(range(iterations))):
        curr_selected_indices = set()
        curr_selected_indices.update(initial_row_selections)
        for label, label_df in df.groupby('gold_label_name'):
            target = total_max_per_label_count - initial_counts.get(label, 0)
            if target <= 0:
                continue
            max_available = label_df[(~label_df.row_index.isin(curr_selected_indices))].shape[0]
            assert (
                target <= max_available
            ), f'insufficient rows available to select for {label} label. Required {target} but only {max_available} were present'
            curr_selected_indices.update(
                label_df[(~label_df.row_index.isin(curr_selected_indices))]
                .sample(n=target, weights='_prefer')
                .row_index.tolist()
            )
        counts = (
            df[df.row_index.isin(curr_selected_indices)]
            .gold_label_name.value_counts()
            .reset_index(name='freq')
            .freq.nunique()
        )
        if counts != 1:
            logging.warning(
                f'labels are not properly balanced: {dict(df[df.row_index.isin(curr_selected_indices)].gold_label_name.value_counts())}'
            )
            exit(1)

        lmi_df = compute_lmi_fast(df[df.row_index.isin(curr_selected_indices)], grams_col='bigram')

        rows.append(
            {
                'boot_i': boot_i,
                'bias': lmi_df.groupby(['bigram']).agg({'bias': 'max'}).reset_index().bias.sum(),
                'bias@0.01': lmi_df[lmi_df.bigram_freq >= 0.01]
                .groupby(['bigram'])
                .agg({'bias': 'max'})
                .reset_index()
                .bias.sum(),
                'counts': dict(
                    df[df.row_index.isin(curr_selected_indices)].gold_label_name.value_counts()
                ),
                'dataset_size': len(curr_selected_indices),
                'selected_rows': curr_selected_indices,
                'lmi': lmi_df.lmi.sum(),
            }
        )

    selected = min(rows, key=lambda x: x['bias'])['selected_rows']

    return selected


def balance_claims(
    df,
    total_max_per_label_count: int,
    initial_row_selections: Optional[set[int]],
) -> set[int]:
    """
    Start by balancing any full claims across label types where possible to match those already selected
    """
    for req_col in ['row_index', 'gold_label_name', 'claim.flat', '_exclude', '_prefer']:
        assert req_col in df.columns, f'missing required column: {req_col}'
    if initial_row_selections is None:
        initial_row_selections = set()
    df = df[df._exclude == False].copy()
    df['selected'] = df.row_index.isin(initial_row_selections)

    result = set()
    label_counts = dict(df[df.selected]['gold_label_name'].value_counts())

    selected_claims = set(df[df.selected]['claim.flat'].unique())

    for _, claim_df in df[df['claim.flat'].isin(selected_claims)].groupby('claim.flat'):
        if claim_df.gold_label_name.nunique() < df.gold_label_name.nunique():
            continue
        selected_counts = dict(claim_df[claim_df.selected]['gold_label_name'].value_counts())
        target_label_count = max(selected_counts.values())

        for label, label_df in claim_df.groupby('gold_label_name'):
            if label_counts.get(label, 0) >= total_max_per_label_count:
                # skip labels that are already at max capacity
                continue

            label_n = min(
                [label_df.shape[0], target_label_count - label_df[label_df.selected].shape[0]]
            )
            if label_n > 0:
                result.update(label_df.sample(n=label_n, weights='_prefer').row_index.tolist())
                label_counts[label] = label_counts.get(label, 0) + label_n

    return result


def balance_data(
    df,
    gram_n: int,
    maximization_factor: Optional[float],
    sampling_iterations: int,
    max_claim_repeat: int,
    max_evidence_repeat: int,
):
    """
    Subselect the data to create a final partitioning of the data that is approximately balanced across relevant categories and labels

    subsample the CIViC descriptions as evidence dataset so it does not represent more than X percent of the data

    Args:
        df: the input datagrame
        allowable_deviation: the total for some category which it is allowed to differ between labels among a given partition or across partitions
        min_group_prop: the minimum prportion of total data a group must account for in order for use to attempt to balance it, groups outside this will be merged
    """
    assert gram_n >= 1
    assert maximization_factor is None or maximization_factor >= 1
    assert sampling_iterations >= 1
    assert max_claim_repeat >= 1
    assert max_evidence_repeat >= 1

    for req_col in ['row_index']:
        assert req_col in df.columns, f'missing required column: {req_col}'

    df = df.copy()
    df['civic.lastUpdatedAt'] = pd.to_datetime(df['civic.lastUpdatedAt'])
    df['_exclude'] = False
    # exclude entries built from civic descriptions that were not already accepted and review in civic
    df.loc[
        (df['evidence.source'] == 'civic')
        & ((df['civic.status'] != 'ACCEPTED') | (df['civic.flagged'] == True)),
        '_exclude',
    ] = True
    # exclude entries flagged by hypothesis as needing a fix
    df.loc[df['civic.flagged'] == 'civic', '_exclude'] = True
    # exlclude 1-star evidence items
    df.loc[df['civic.evidenceRating'] == 1, '_exclude'] = True
    # these generally should be 1-star
    df.loc[
        (df['civic.evidenceLevel'] == 'C') & (df['civic.evidenceType'] == 'ONCOGENIC'), '_exclude'
    ] = True
    # exclude N/A evidenceDirections
    df.loc[df['civic.evidenceDirection'] == 'N/A', '_exclude'] = True
    # exclude anything that hasn't been updated since before CIViC was tracking
    df.loc[
        (df['civic.lastUpdatedAt'].dt.year < 2017)
        & (~df['evidence.selection_method'].str.contains('swap')),
        '_exclude',
    ] = True

    # exclude anything submitted past our cut-off date for the time split. keep them in the unassigned partition
    df.loc[df['civic.submittedAt'] > DATE_CUTOFF, '_exclude'] = True
    df.loc[df['civic.submittedAt'] > DATE_CUTOFF, '_keep'] = True

    df['generated'] = (
        (df.generated.fillna(False).astype(bool))
        | (df['claim.generated'].fillna(False))
        | (df['evidence.generated'].fillna(False))
    )
    df['evidence.selection_method'] = df['evidence.selection_method'].fillna('n/a')
    df.loc[df['evidence.source'] == 'civic', 'evidence.selection_method'] = 'n/a'
    df["_keep"] = False

    # keep everything manual from hypothesis
    df['_manual'] = (
        (~df.generated)
        & (df['evidence.source'] != 'civic')
        & (df['evidence.selection_method'] == 'hypothesis')
    )
    df.loc[df._manual, '_keep'] = True
    # keep the neutral-bot generated entries (most similar to what actual neutral entries will look like)
    df.loc[
        (df['evidence.selection_method'] == 'neutral-bot') & (df['claim.source'] == 'neutral-bot'),
        '_keep',
    ] = True
    # never select descriptions as evidence to start
    df.loc[(df['evidence.source'] == 'civic'), '_keep'] = False

    # keep anything with a proper group review
    df['_reviewed'] = df.reviews.apply(has_review)
    df.loc[df._reviewed, '_keep'] = True

    df.loc[df._keep == True, 'prebalancing_set'] = True
    df['bigram'] = df.claim.apply(lambda c: get_ngrams(c, gram_n))
    df['_prefer'] = df.apply(preference_order, axis=1)
    all_selected_rows = set(df[df._keep == True].row_index.tolist())
    df.loc[df._exclude == True, '_keep'] = False
    logging.info(f'maximization factor: {maximization_factor}')

    # exclude any repeated claims more than max_claim_repeat times
    df = df.sample(frac=1).sort_values('_prefer')
    df['_claim_repeat_index'] = df.groupby(['claim.flat', 'gold_label_name']).cumcount()
    exclude = df[(df._claim_repeat_index >= max_claim_repeat) & (df._exclude == False)].shape[0]
    logging.warning(f'will exclude {exclude} pairs to to max_claim_repeat > {max_claim_repeat}')
    df.loc[df._claim_repeat_index >= max_claim_repeat, '_exclude'] = True

    # now exclude any repeated pieces of evidence for the same reason
    df = df.sample(frac=1).sort_values('_prefer')
    df['_evidence_repeat_index'] = df.groupby(['evidence.flat', 'gold_label_name']).cumcount()
    exclude = df[(df._evidence_repeat_index >= max_evidence_repeat) & (df._exclude == False)].shape[
        0
    ]
    logging.warning(
        f'will exclude {exclude} pairs to to max_evidence_repeat > {max_evidence_repeat}'
    )
    df.loc[df._evidence_repeat_index >= max_evidence_repeat, '_exclude'] = True

    for partition, part_df in df[df._exclude == False].groupby(DOC_PARTITION_COL):
        smallest_label = part_df.gold_label_name.value_counts().reset_index(name='freq').freq.min()
        if part_df[part_df.row_index.isin(all_selected_rows)].shape[0] == 0:
            raise AssertionError(
                f'cannot use a maximization factor when there are no initially selected rows for any given partition'
            )
        else:
            largest_selected_label = (
                part_df[part_df.row_index.isin(all_selected_rows)]
                .gold_label_name.value_counts()
                .reset_index(name='freq')
                .freq.max()
            )
        logging.info(f'{partition} largest initial label count: {largest_selected_label}')
        logging.info(f'{partition} smallest available label count: {smallest_label}')

        if maximization_factor is None:
            total_max_per_label_count = smallest_label
        else:
            total_max_per_label_count = min(
                [smallest_label, int(largest_selected_label * maximization_factor)]
            )

        current_counts = lambda: dict(
            part_df[part_df.row_index.isin(all_selected_rows)].gold_label_name.value_counts()
        )
        logging.info(f'{partition} target per label count: {total_max_per_label_count}')
        logging.info(f'{partition} INITIAL: {current_counts()}')
        logging.info(f'{partition} AVAILABLE: {dict(part_df.gold_label_name.value_counts())}')

        selected_rows = balance_claims(
            part_df,
            total_max_per_label_count=total_max_per_label_count,
            initial_row_selections=all_selected_rows,
        )
        all_selected_rows.update(selected_rows)
        logging.info(f'{partition} CLAIMS: {current_counts()}')

        selected_rows = random_rebalancing_partition(
            part_df,
            total_max_per_label_count=total_max_per_label_count,
            initial_row_selections=all_selected_rows,
            iterations=sampling_iterations,
        )
        all_selected_rows.update(selected_rows)
        logging.info(f'{partition} POST RANDOM FILL: {current_counts()}')
        assert all([v == total_max_per_label_count for v in current_counts().values()])

    df.loc[df.row_index.isin(all_selected_rows), '_keep'] = True
    logging.info(f'{df._keep.sum()} total entries were selected')
    logging.info(f'{dict(df[df._keep == True].gold_label_name.value_counts())}')
    for part, part_df in df.groupby(DOC_PARTITION_COL):
        logging.info(
            f'{part}: {dict(part_df[part_df._keep == True].gold_label_name.value_counts())}'
        )
        logging.info(
            f'{part}: {dict(part_df[part_df._keep == True].gold_label_name.value_counts(normalize=True))}'
        )
    return df.drop(columns=['bigram', '_reviewed', '_prefer'])


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to the JSONL of examples to be partitioned')
    parser.add_argument('--partitioning', required=True)
    parser.add_argument('output', help='path to the JSONL output')
    parser.add_argument('--maximization_factor', default=None, type=float)
    parser.add_argument('--sampling_iterations', default=1000, type=int)
    parser.add_argument('--ngram', default=2, type=int)
    # lowering these two will encourage diversity
    parser.add_argument(
        '--max_claim_repeat',
        help='maximum number of times a single claim can be re-used with different evidence for the same label',
        default=5,
        type=int,
    )
    parser.add_argument(
        '--max_evidence_repeat',
        help='maximum number of times a single piece of evidence can be re-used with different claims for the same label. There are often spans that cover several mutations with the same selection so this number should be much more permissive than the claim repeat limit',
        default=20,
        type=int,
    )
    args = parser.parse_args()

    assert args.sampling_iterations >= 1

    df = read_jsonl(args.input)

    if 'partition' in df.columns:
        df = df.drop(columns=['partition'])
    if DOC_PARTITION_COL in df.columns:
        df = df.drop(columns=[DOC_PARTITION_COL])

    logging.info(f'reading: {args.partitioning}')
    part_df = pd.read_csv(args.partitioning, dtype={'pmid': 'str'}).rename(
        columns={'partition': DOC_PARTITION_COL, 'pmid': 'document.pmid'}
    )

    df = df.merge(part_df[['document.pmid', DOC_PARTITION_COL]])

    input_examples = df.shape[0]
    logging.info(f'read {input_examples} examples')

    doc_df = df[['document.pmid', DOC_PARTITION_COL]].drop_duplicates()

    assert (
        doc_df[DOC_PARTITION_COL].isna().sum() == 0
    ), f'Every document should have a pre-assigned partition. There are {doc_df[DOC_PARTITION_COL].isna().sum()} of {doc_df.shape[0]} documents missing assignments'

    df['row_index'] = df.index.tolist()

    df = balance_data(
        df,
        gram_n=args.ngram,
        maximization_factor=args.maximization_factor,
        sampling_iterations=args.sampling_iterations,
        max_claim_repeat=args.max_claim_repeat,
        max_evidence_repeat=args.max_evidence_repeat,
    )
    df[PARTITION_COL] = df[DOC_PARTITION_COL]
    df.loc[(df._keep == False) | (df._exclude == True), PARTITION_COL] = 'unassigned'

    logging.info(f'partitioning results: {dict(df.partition.value_counts())}')

    df['_manual'] = (
        (df.generated != True)
        & (df['evidence.source'] != 'civic')
        & (df['evidence.selection_method'] == 'hypothesis')
    )
    logging.info(
        f'{dict(df[df._manual]["partition"].value_counts())} total assigned manual examples per partition'
    )
    df['display_id'] = df['id'].apply(lambda _: str(uuid.uuid4()))
    df = df[(df._manual) | (df._keep)]
    logging.info(f'writing {df.shape[0]} examples')
    df = df.drop(columns=[c for c in df.columns if c.startswith('_')])

    for field in ['maximization_factor', 'ngram']:
        df[f'balance_args.{field}'] = getattr(args, field)
    logging.info(f'{df[df.partition != "unassigned"].shape[0]} total assigned examples')

    dump_jsonl(df, args.output)


if __name__ == '__main__':
    main()
