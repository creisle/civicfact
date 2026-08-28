import argparse
import itertools
import json
import logging

import bioc
import pandas as pd
import tqdm

from article_utils.db import (
    backfill_bioc,
    connect,
    fetch_xml,
    get_or_create_textid,
    get_partition_assignments,
)
from article_utils.util import dump_jsonl

from .util import Text

tqdm.tqdm.pandas()


def convert_bioc_to_plain_text(bioc_text):
    col = bioc.loads(bioc_text)
    assert len(col.documents) == 1

    passages = []
    for passage in col.documents[0].passages:
        passages.append(passage.text)
    return ''.join(passages)


def first_non_falsy(series):
    series = series.dropna()
    if series.size == 0:
        return None
    return series.iat[0]


def main(args, db_conn):
    logging.info(f'reading: {args.input}')
    data_df = pd.read_json(args.input, lines=True, dtype=False)

    # filter out entries from asco abstracts etc
    data_df = data_df[data_df['document.pmid'].str.match(r'^\d+')]

    backfill_bioc(db_conn, set(data_df['document.pmid'].unique()), strict=False)
    logging.info('fetching XML files')
    pub_df = pd.DataFrame.from_records(
        list(fetch_xml(db_conn, data_df, article_id_col='document.pmid', strict=False).values())
    )
    pub_df = pub_df[(pub_df['content_source'] != 'pubmed') & (pub_df['content_type'] == 'bioc')]

    logging.info('converting bioc files to plain text')
    pub_df['text'] = pub_df['content'].progress_apply(convert_bioc_to_plain_text)
    pub_df['text_length'] = pub_df['text'].str.len()

    # filter out the shortest publications as they are likely truncated and not fulltext
    filter_length = pub_df['text_length'].quantile(args.min_quantile_length_filter)
    logging.info(f'filtering out articles with less than {filter_length} characters')
    pub_df = pub_df[pub_df['text_length'] >= filter_length]
    pub_df = pub_df.rename(columns={c: f'document.{c}' for c in pub_df})

    # filter down to data entries with matched fulltext
    logging.info(f'filtering {data_df.shape[0]} input rows based on fulltext')
    data_df = data_df.merge(pub_df[['document.pmid', 'document.text']])
    logging.info(f'{data_df.shape[0]} remaining rows')

    # drop NEI examples where the same claim has both NEI and non-NEI label for the same paper
    logging.info(
        f'filtering out NEI examples from {data_df["document.pmid"].nunique()} documents where an explicitly labelled alternative exists'
    )
    non_nei_df = data_df.copy()
    non_nei_df = non_nei_df[non_nei_df.gold_label_name != 'NEI']

    non_nei_df = non_nei_df[['claim.flat', 'document.pmid']].drop_duplicates()
    all_claims_count = non_nei_df.shape[0]
    logging.info(f'There are {all_claims_count} non-redundant claims')
    non_nei_df['has_explicit_label'] = True
    data_df = data_df.merge(non_nei_df, how='left')

    data_df = data_df[(data_df.gold_label_name != 'NEI') | (data_df.has_explicit_label != True)]
    logging.info(
        f'{data_df.shape[0]} remaining rows after filtering NEI examples with explicit alternates'
    )
    assert data_df.shape[0] >= all_claims_count

    # merge claims/civic evidence items
    agg_functions = {}
    key_columns = ['civic.id', 'claim.flat', 'document.pmid', 'gold_label_name']
    for col in data_df.columns:
        if col.startswith('evidence.') or col.startswith('hypothesis.'):
            # skip evidence./hypothesis. columns since that data is also contained in the evidence object and in many cases will change once evidence is combined
            continue
        if col not in key_columns:
            if col.startswith('document') or col.startswith('claim') or col.startswith('civic'):
                agg_functions[col] = first_non_falsy
            elif col in {'is_flagged', 'in_table', 'flagged'}:
                agg_functions[col] = 'any'
            elif col in {'reviews', 'evidence'}:
                agg_functions[col] = lambda list_like_col: list(
                    itertools.chain.from_iterable(list_like_col)
                )
            else:
                agg_functions[col] = list

    data_df = data_df.groupby(key_columns, dropna=False).agg(agg_functions).reset_index()
    logging.info(f'{data_df.shape[0]} remaining rows after merging claims')

    data_df = data_df.drop(columns=['has_explicit_label', 'example', 'pmid.partition', 'partition'])

    logging.info(
        f'label distribution: {dict(data_df.gold_label_name.value_counts(normalize=True))}'
    )

    # add partitions
    partitions_df = pd.DataFrame.from_records(get_partition_assignments(db_conn)).rename(
        columns={'pmid': 'document.pmid'}
    )
    data_df = data_df.merge(partitions_df, how='left')
    assert data_df[data_df.partition.isna()].shape[0] == 0

    logging.info(f'partition distribution: {dict(data_df.partition.value_counts(normalize=True))}')

    data_df = data_df.rename(columns={'evidence': 'selections', 'document.text': 'evidence.flat'})
    data_df['id'] = (
        'pmid:'
        + data_df['document.pmid'].astype(str)
        + '_claim:'
        + data_df['claim.hash'].astype(str)
        + '_civic:'
        + data_df['civic.id'].fillna('').astype(str)
    )
    data_df['evidence'] = data_df['evidence.flat'].apply(lambda doc: [Text(doc)])

    def hash_text(texts):
        return get_or_create_textid(db_conn, json.dumps([text.content for text in texts]))

    data_df['evidence.hash'] = data_df['evidence'].progress_apply(hash_text)
    dump_jsonl(data_df, args.output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--article_db_file', required=True)
    parser.add_argument('--min_quantile_length_filter', default=0.05, type=float)
    parser.add_argument(
        'input',
        help='path to the JSONL file containing the NLI examples (with excerpts not fulltext)',
    )
    parser.add_argument('output')
    args = parser.parse_args()
    conn = connect(args.article_db_file)

    try:
        main(args, conn)
    finally:
        conn.close()
