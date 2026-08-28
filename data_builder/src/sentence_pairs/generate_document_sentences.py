import argparse
import logging
import re

import bioc
import pandas as pd
import pysbd
import tqdm

from article_utils.db import connect, fetch_xml
from article_utils.match_text import regex_match_substring
from article_utils.util import (
    dump_jsonl,
    listify,
    normalize_and_expand_column,
    read_jsonl,
)
from civic.util import load_dump
from nli.util import Text
from sentence_pairs.generate_document_chunks import convert_biocxml_to_text

splitter = pysbd.Segmenter('en', char_span=True)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('chunks')
    parser.add_argument('output', help='path to output document chunks to')
    parser.add_argument('--article_db_file', default='./data/articles_db.sqlite')
    args = parser.parse_args()

    chunks_df = read_jsonl(
        args.chunks,
        required_columns=[
            'document.pmid',
            'chunk.start',
            'chunk.end',
            'passages',
            'pubtator',
            'chunk.text',
        ],
    ).sample(n=10)

    db_conn = connect(args.article_db_file)
    try:
        articles = fetch_xml(db_conn, chunks_df, 'document.pmid', strict=False)
    finally:
        db_conn.close()

    for pmid, doc in articles.items():
        sentences_df = pd.DataFrame.from_records(
            [
                {'text': s.sent, 'start': s.start, 'end': s.end}
                for s in splitter.segment(convert_biocxml_to_text(doc['content']))
            ]
        )
        print(sentences_df)
        print(sentences_df.text.str.len().describe())
        break

    logging.info(f'overlapping passage metadata with document chunks')
    chunks_df = overlap_chunks_by_range(
        chunks_df,
        passages_df,
        prefix='passage',
        new_col_name='passages',
        min_overlap=args.chunk_overlap,
    )

    # extend pubtator annotations
    logging.info(f'reading: {args.pubtator_annotations}')
    pub_df = pd.read_json(args.pubtator_annotations, lines=True, dtype=False).rename(
        columns={'pmid': 'document.pmid'}
    )
    logging.info(f'loaded {pub_df.shape[0]} pubtator annotations')
    pub_df = pub_df[
        pub_df['text.match_error'].fillna('') == ''
    ]  # ignore elements with matching errors
    logging.info(f'there are {pub_df.shape[0]} pubtator annotations with valid matching')

    pub_df = expand_pubtator_annotations(
        pub_df.merge(articles_df[['plaintext', 'document.pmid']]), min_length=4
    )
    ### now calculate overlap with pubtator annotations

    logging.info(f'calculating pubtator annotations that overlap with document chunks')

    chunks_df = overlap_chunks_by_range(
        chunks_df,
        pub_df,
        prefix='match',
        new_col_name='pubtator',
        min_overlap=args.min_pubtator_overlap,
        filter_by_percent=True,
        additional_keep_columns=[
            c for c in pub_df.columns if c.startswith('pubtator.') or c.startswith('match.')
        ],
    )

    # logging.info('calculating manual evidence selection overlap with document chunks')
    # chunks_df = overlap_chunks_by_range(
    #     chunks_df,
    #     selections_df,
    #     prefix='evidence.meta',
    #     new_col_name='selections',
    #     min_overlap=args.min_selection_overlap,
    #     filter_by_percent=True,
    #     additional_keep_columns=['claim.flat', 'civic.id'],
    # )

    chunks_df['chunk.overlap'] = args.chunk_overlap
    chunks_df['chunk.size'] = args.chunk_size
    print(chunks_df)
    print(chunks_df.dtypes)

    total_chunks = (
        chunks_df.groupby(['document.pmid'])
        .agg({'chunk.id': 'nunique'})
        .reset_index()
        .rename(columns={'chunk.id': 'document.total_chunks'})
    )
    chunks_df = chunks_df.merge(total_chunks)
    chunks_df = chunks_df.rename(columns={'content_source': 'document.content_source'})

    dump_jsonl(chunks_df, args.output)


if __name__ == '__main__':
    main()
