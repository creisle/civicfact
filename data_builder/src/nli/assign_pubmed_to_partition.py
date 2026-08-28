"""
Split pubmed IDs across partitions. Try to balance by metadata (journal, full-text available)
and which entries currently have annotations (so that the manually curated data is relatively balanced)
otherwise just assign randomly
"""

import argparse
import logging
import random

import pandas as pd
import requests_cache

from article_utils.db import (
    batch_get_article_metadata,
    connect,
    get_partition_assignments,
    set_partition_assignments,
)
from article_utils.ncbi import ESUMMARY_URL, get_pubmed_metadata
from civic.util import load_dump

from .util import PARTITION_COL

# globally cache requests
requests_cache.install_cache(
    '.requests_cache', urls_expire_after={ESUMMARY_URL: requests_cache.NEVER_EXPIRE}
)

PARTITIONS = ['train', 'dev', 'test']
DOC_PARTITION_COL = 'pmid.partition'


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


def ft_license_status(article_metadata):
    if article_metadata['author_fulltext']:
        return True
    elif article_metadata['license']:
        return True
    return False


def calculate_balance_score(
    df, balance_factor: str, expected_fractions: dict[str, float], min_group_size: int = 5
) -> float:
    """
    Calculate a score of the percent of the content that is not balanced. Ignore small groups below a
    min_group_size
    """
    df = (
        df[[balance_factor, PARTITION_COL]]
        .value_counts()
        .reset_index(name='freq')
        .merge(df[[balance_factor]].value_counts().reset_index(name='total'))
        .merge(df[[PARTITION_COL]].value_counts().reset_index(name='partition_size'))
    )
    df = df[df.freq >= min_group_size]

    df['expected_fraction'] = df[PARTITION_COL].apply(lambda p: expected_fractions[p])
    df['expected_freq'] = (df.expected_fraction * df.total).round().astype(int)
    df['actual_fraction'] = df.freq / df.total
    df['error'] = df.freq - df.expected_freq

    # what proportion of total article groups are unbalanced across partitions
    return df.error.abs().sum() / df.freq.sum()


def pick_assignments(
    df: pd.DataFrame,
    expected_fractions: dict[str, float],
    balance_factors: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = df.copy()
    if PARTITION_COL not in df.columns:
        df[PARTITION_COL] = ''
    df[PARTITION_COL] = df[PARTITION_COL].fillna('')

    current_split = {
        k: v for k, v in dict(df[PARTITION_COL].value_counts()).items() if k in PARTITIONS
    }
    logging.info(f'current assignments: {current_split}')
    for p in PARTITIONS:
        current_split.setdefault(p, 0)

    assignments = []

    for _, subgroup_df in df.groupby(balance_factors, dropna=False):
        if subgroup_df[subgroup_df[PARTITION_COL] == ''].shape[0] == 0:
            # nothing to assign, keep assignments as is
            assignments.append(subgroup_df)
            continue
        adjusted_fractions = dict(**expected_fractions)
        for part in PARTITIONS:
            adjusted_fractions[part] = max(
                [
                    adjusted_fractions[part]
                    - (
                        subgroup_df[subgroup_df[PARTITION_COL] == part].shape[0]
                        / subgroup_df.shape[0]
                    ),
                    0,
                ]
            )

        subgroup_df.loc[subgroup_df[PARTITION_COL] == '', PARTITION_COL] = random.choices(
            list(adjusted_fractions),
            weights=[adjusted_fractions[e] for e in expected_fractions],
            k=subgroup_df[subgroup_df[PARTITION_COL] == ''].shape[0],
        )
        assignments.append(subgroup_df)

    if assignments:
        df = pd.concat(assignments)

    scores = {}
    for col in balance_factors:
        score = calculate_balance_score(df, col, expected_fractions)
        scores[col] = round(100 * score)
    return df, scores


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_percent', default=20, type=percent)
    parser.add_argument('--test_percent', default=20, type=percent)
    parser.add_argument('--article_db_file', help='path to the sqlite db file', required=True)
    parser.add_argument('--civic_dump', help='path to the civic dump jsonl', required=True)
    parser.add_argument(
        '--annotations', help='path to the JSONL of examples with annotations', required=True
    )
    parser.add_argument(
        '--force',
        default=False,
        help='delete previous partitioning records in the db',
        action='store_true',
    )
    parser.add_argument('output', help='path to the JSONL output')
    args = parser.parse_args()

    if args.dev_percent + args.test_percent > 100:
        parser.error('dev and test percentage cannot be more than 100%')

    expected_fractions = {'dev': args.dev_percent / 100, 'test': args.test_percent / 100}
    expected_fractions['train'] = 1 - expected_fractions['dev'] - expected_fractions['test']

    annot_df = pd.read_json(args.annotations, lines=True, dtype=False)
    annot_df['annotated'] = annot_df['generated'] == False
    annot_df = (
        annot_df.rename(columns={'document.pmid': 'pmid'})
        .groupby(['pmid'])
        .agg({'annotated': 'sum'})
        .reset_index()
    )

    def custom_round(x, base=5):
        return int(base * round(float(x) / base))

    annot_df['annotated'] = annot_df['annotated'].apply(custom_round)

    civic_df = load_dump(args.civic_dump).drop_duplicates(['id'])
    civic_df['civic_status'] = civic_df.status == 'ACCEPTED'
    civic_df['lastUpdatedAt'] = pd.to_datetime(civic_df['lastUpdatedAt'])
    civic_df['pre_mp'] = civic_df.lastUpdatedAt.dt.year < 2023
    civic_df = civic_df[civic_df.lastUpdatedAt.dt.year >= 2017] # other stuff from initial bulk upload
    civic_df = civic_df.groupby(['pmid'], dropna=True).agg({'civic_status': 'any', 'pre_mp': 'all'}).reset_index()
    civic_df = civic_df[civic_df.pmid != '']


    pmids = [
        i
        for i in sorted(
            set(civic_df.pmid.unique().tolist() + annot_df.pmid.unique().tolist()) - {''}
        )
        if not i.startswith('A') and '/' not in i
    ]

    db_conn = connect(args.article_db_file)
    try:
        if args.force:
            db_conn.execute('DELETE FROM partition_assignment')
        part_df = pd.DataFrame.from_records(get_partition_assignments(db_conn))
        assert part_df.pmid.nunique() == part_df.shape[0]
        logging.info(
            f'db partition assignements: {dict(part_df.partition.value_counts(normalize=True, dropna=False))}'
        )
        articles_df = pd.DataFrame.from_records(
            batch_get_article_metadata(db_conn, article_ids=pmids, id_type='pmid')
        )
        assert articles_df.pmid.nunique() == articles_df.shape[0], 'PMID is not unique'
        articles_df['ft_status'] = articles_df.apply(ft_license_status, axis=1)

        df = articles_df.merge(annot_df[['pmid', 'annotated']], how='outer')
        df['annotated'] = df.annotated.fillna(0)

        assert df.pmid.nunique() == df.shape[0]

        df = df.merge(civic_df[['pmid', 'civic_status', 'pre_mp']], how='outer', on=['pmid'])
        df['civic_status'] = df.civic_status.fillna(False)

        # try to make sure these are split evenly-ish when partitioning the data to ensure each partition is representative of the variation in the dataset
        # the first two factors are b/c those are the items we are likely to have actual data for (not all articles in civic have been annotated yet)
        balance_factors = ['annotated', 'ft_status', 'civic_status', 'pre_mp', 'pubtype', 'journal']

        assert df.pmid.nunique() == df.shape[0], 'PMID is not unique'

        if part_df.shape[0] > 0:
            df = df.merge(part_df, how='left')

        logging.info(
            f'initial assignements: {dict(df.partition.value_counts(normalize=True, dropna=False))}'
        )

        assign_df, scores = pick_assignments(
            df,
            expected_fractions,
            balance_factors,
        )
        logging.info(dict(assign_df.partition.value_counts(normalize=True)))
        logging.info(f'scores: {scores}')

        new_assignments_df = assign_df[['pmid', 'partition']].copy()
        if part_df.shape[0] > 0:
            new_assignments_df = new_assignments_df.merge(
                part_df.rename(columns={PARTITION_COL: 'prev_partition'}), how='left'
            ).fillna('')
        else:
            new_assignments_df['prev_partition'] = ''
        new_assignments_df = new_assignments_df[new_assignments_df.prev_partition == ''][
            ['pmid', PARTITION_COL]
        ]

        if new_assignments_df.shape[0]:
            logging.info(f'adding {new_assignments_df.shape[0]} new assignments')
            set_partition_assignments(
                db_conn,
                new_assignments_df.to_dict('records'),
            )

    finally:
        db_conn.close()

    logging.info(f'writing: {args.output}')
    assign_df[
        ['pmid', 'partition'] + balance_factors + ['author_fulltext', 'license', 'pmcid']
    ].to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
