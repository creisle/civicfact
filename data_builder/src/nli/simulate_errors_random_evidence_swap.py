"""
For all hypothesis retrieved evidence, create a version where the evidence is all overlapping chunks instead of only the hypothesis selcted parts
this is to get the model used to longer contexts

Also generate some completly irrelevant NEI examples by re-pairing claims with evidence from other papers that do not share any claim elements (expcet type/significance)
"""

import argparse
import json
import logging
import random
from uuid import uuid4

import pandas as pd

from article_utils.util import dump_jsonl, read_jsonl
from nli.build_dataset import merge_overlapping_selections
from nli.util import Text, concat_text


def filter_pair_dups(input_df, df):
    # drop any generated entries that duplicate an existing entry
    df['_id'] = df['claim.flat'] + df['evidence.flat'] + df['document.pmid']
    input_df['_id'] = input_df['claim.flat'] + input_df['evidence.flat'] + input_df['document.pmid']
    input_df['_exists'] = True
    df = df.merge(input_df[['_id', '_exists']].drop_duplicates(), how='left')
    df = df[df._exists.isna()]
    logging.info(f'created {df.shape[0]} new pairs')

    return df.drop(columns=[c for c in df.columns if c.startswith('_')])


def convert_chunks_to_evidence(chunks):
    evidence = []
    for chunk in chunks:
        evidence.append(
            Text(
                chunk['chunk.text'],
                type='unstructured',
                meta={
                    'start': chunk['chunk.start'],
                    'end': chunk['chunk.end'],
                    'id': chunk['chunk.id'],
                },
            )
        )
    return merge_overlapping_selections(evidence)


def make_new_evidence(chunks_file, scores_file, min_length: float, max_length: float):
    chunk_df = read_jsonl(
        chunks_file,
        required_columns=['chunk.size', 'chunk.start', 'chunk.end', 'document.pmid'],
    )
    chunk_size = chunk_df['chunk.size'].mode().tolist()[0]
    print('chunk_size', chunk_size)

    print(chunk_df.columns)
    scores_df = read_jsonl(
        scores_file, required_columns=['similarity.score', 'document.pmid', 'claim.flat']
    )
    print(scores_df.columns)

    scores_df = scores_df.sample(frac=1)
    scores_df = scores_df.sort_values(['similarity.score'], ascending=False)
    scores_df['_claim_index'] = scores_df.groupby(['document.pmid', 'claim.flat']).cumcount()
    scores_df = scores_df.sort_values(['document.pmid', 'claim.flat', 'similarity.score'])
    print(scores_df)
    max_chunks = round(max_length / chunk_size)
    scores_df = scores_df[scores_df._claim_index < max_chunks]

    df = scores_df.merge(chunk_df)
    df = (
        df.groupby(['claim.flat'] + [c for c in df.columns if c.startswith('document.')])[
            [c for c in df.columns if c.startswith('chunk.')]
        ]
        .apply(lambda s: s.to_dict('records'))
        .reset_index(name='evidence')
    )
    chunk_choices = list(
        range(max(1, int(round(min_length / chunk_size))), min([2, max_chunks + 1]))
    )
    print('chunk choices', chunk_choices)
    df['evidence'] = df.evidence.apply(
        lambda evidence: evidence[: random.choice(chunk_choices) + 1]
    )
    df['evidence'] = df['evidence'].apply(convert_chunks_to_evidence)
    df['evidence.flat'] = df.evidence.apply(concat_text)
    df['evidence.source'] = df['document.content_source']
    df['evidence.selection_method'] = 'pubtator-elements-score'
    return df.drop(columns=['claim.flat']).drop_duplicates(['evidence.flat'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--disimilar_pmids', required=True)
    parser.add_argument('--partitioning', required=True)
    parser.add_argument('--document_chunks', required=True)
    parser.add_argument('--scores', required=True)
    parser.add_argument('output')
    parser.add_argument('--dup_level', type=int, default=5)
    args = parser.parse_args()
    assert args.dup_level >= 1

    logging.info(f'reading: {args.partitioning}')
    part_df = pd.read_csv(args.partitioning, dtype={'pmid': 'str'}).rename(
        columns={'pmid': 'document.pmid'}
    )
    partition_mapping = {}
    for _, row in part_df.iterrows():
        partition_mapping[row['document.pmid']] = row['partition']

    logging.info(f'reading: {args.disimilar_pmids}')
    with open(args.disimilar_pmids, 'r') as fh:
        disimilar_pmids = json.load(fh)

        logging.info(f'read {len(disimilar_pmids)} disimilar pmids')

        # make sure we swap claims within partitions
        for pmid in list(disimilar_pmids):
            disimilar_pmids[pmid] = [
                p for p in disimilar_pmids[pmid] if partition_mapping[p] == partition_mapping[pmid]
            ]
            random.shuffle(disimilar_pmids[pmid])
            if len(disimilar_pmids[pmid]) == 0:
                del disimilar_pmids[pmid]

        logging.info(f'{len(disimilar_pmids)} disimilar pmids after matching partition')

    disimiliar_by_partition = {}
    for pmid in disimilar_pmids:
        partition = partition_mapping[pmid]
        disimiliar_by_partition[partition] = disimiliar_by_partition.get(partition, 0) + 1
    logging.info(f'disimiliar PMIDs by partition: {disimiliar_by_partition}')

    input_df = read_jsonl(args.input)
    logging.info(f'read {input_df.shape[0]} input pairs')
    input_df = input_df[
        input_df.gold_label_name != 'NEI'
    ]  # don't need to make more NEI examples of claims that already are NEI
    ev_df = input_df[['evidence.flat']].drop_duplicates()
    min_evidence_length = ev_df['evidence.flat'].str.len().quantile(0.05).astype(int)
    max_evidence_length = ev_df['evidence.flat'].str.len().quantile(0.95).astype(int)
    logging.info(
        f'the evidence lengths 0.05-0.95 range is: {min_evidence_length}-{max_evidence_length}'
    )

    if 'partition' in input_df.columns:
        input_df = input_df.drop(columns=['partition'])
    original_columns = input_df.columns

    df = input_df.copy()
    document_columns = [c for c in df.columns if c.startswith('document')] + [
        'pmid.partition',
        'partition',
    ]
    evidence_columns = set([c for c in df.columns if c.startswith('evidence')] + document_columns)
    ev_df = df[[c for c in df.columns if c in evidence_columns]].drop_duplicates(
        ['document.pmid', 'evidence.flat']
    )
    # ev_df = ev_df[ev_df['evidence.source'] != 'civic']

    new_ev_df = make_new_evidence(
        args.document_chunks, args.scores, min_evidence_length, max_evidence_length
    )

    ev_df = pd.concat([new_ev_df, ev_df]).reset_index()
    ev_df = ev_df[ev_df['document.pmid'].isin(disimilar_pmids)]

    evidence_pmids = set(ev_df['document.pmid'].unique())
    for pmid in sorted(disimilar_pmids):
        disimilar_pmids[pmid] = set(disimilar_pmids[pmid]) & evidence_pmids
        if not disimilar_pmids[pmid]:
            del disimilar_pmids[pmid]
    logging.info(
        f'there are {len(disimilar_pmids)} dismilar pmids after filtering evidence matches'
    )
    logging.info(f'There are {ev_df.shape[0]} unique pieces of evidence')

    # now make the claims
    df = df.drop(columns=[c for c in df.columns if c in evidence_columns and c != 'document.pmid'])
    df = df[df['document.pmid'].isin(disimilar_pmids)]
    df = df.rename(columns={'document.pmid': 'old_pmid'})
    df['document.pmid'] = df['old_pmid'].apply(lambda pmid: disimilar_pmids[pmid])
    df = df.explode('document.pmid')
    logging.info(f'There are {df.shape[0]} possible swaps')
    df['gold_label_name'] = 'NEI'
    df['reviews'] = df.apply(lambda _: [], axis=1)
    assert df['claim.flat'].isna().sum() == 0
    prev_count = df['claim.flat'].nunique()
    if len(set(df.columns) & set(ev_df.columns)) > 1:
        raise AssertionError(
            f'should only share the document.pmid column: {set(df.columns) & set(ev_df.columns)}'
        )
    df = df.merge(ev_df, on=['document.pmid'])
    assert df['claim.flat'].isna().sum() == 0
    assert df['claim.flat'].nunique() == prev_count
    logging.info(
        f'There are {df.shape[0]} possible pairs across {df["claim.flat"].nunique()} unique claims and {df["evidence.flat"].nunique()} unique pieces of evidence and {df["document.pmid"].nunique()} unique documents'
    )
    df = df.sample(frac=1)
    df['_ev_dup_index'] = df.groupby(['evidence.flat']).cumcount()
    df = df[df._ev_dup_index < args.dup_level]
    df['_ev_index'] = df['evidence.source'].apply(lambda ev: 1 if ev == 'civic' else 0)
    df = df.sort_values('_ev_index')
    df['_claim_index'] = df.groupby(['claim.flat']).cumcount()

    print(df._claim_index.value_counts(dropna=False))

    df = df[df._claim_index < args.dup_level]
    logging.info(f'There are {df.shape[0]} possible swaps post de-duping')
    df['evidence.selection_method'] += ' + evidence-swap'
    df['id'] = df.apply(lambda _: uuid4(), axis=1)
    df['generated'] = True
    df['pmid.partition'] = df['partition'] = df['document.pmid'].apply(
        lambda pmid: partition_mapping[pmid]
    )
    df['revised_claim.flat'] = None
    for part, part_df in df.groupby('pmid.partition', dropna=False):
        logging.info(f'counts [{part}]: {dict(part_df.gold_label_name.value_counts(dropna=False))}')

    # now merge with original data
    df = pd.concat([input_df, df])
    logging.info(f'{df.shape[0]} combined examples')
    logging.info(f'{dict(df.gold_label_name.value_counts())}')
    assert df[df.claim.isna()].shape[0] == 0
    df = df.drop(columns=[c for c in df.columns if c.startswith('_')])
    df['pmid.partition'] = df['partition'] = df['document.pmid'].apply(
        lambda pmid: partition_mapping[pmid]
    )

    for part, part_df in df.groupby('pmid.partition', dropna=False):
        logging.info(f'counts [{part}]: {dict(part_df.gold_label_name.value_counts(dropna=False))}')

    dump_jsonl(df[original_columns], args.output)
