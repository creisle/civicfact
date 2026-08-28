import argparse
import logging

import pandas as pd

from article_utils.db import connect, fetch_xml
from article_utils.util import dump_jsonl

FIX_DUP_DELETIONS = {'22713161': '25998099'}
BAD_PMIDS = ['361866502']


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('civic')
parser.add_argument('--article_db_file', required=True)
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.civic}')
civic_df = pd.read_json(args.civic, lines=True, dtype=False)
civic_df['_pmid'] = civic_df.source.apply(
    lambda source: source['citationId'] if source['sourceType'] == 'PUBMED' else ''
)
civic_df['_pmid'] = civic_df._pmid.apply(
    lambda pmid: FIX_DUP_DELETIONS.get(str(pmid), pmid)
).astype(str)

civic_df = civic_df[~civic_df._pmid.isin(BAD_PMIDS)]

db_conn = connect(args.article_db_file)
try:
    articles = fetch_xml(db_conn, civic_df, article_id_col='_pmid', strict=False)
finally:
    db_conn.close()

print(f'loaded {len(articles)} articles')
articles_df = (
    pd.DataFrame.from_records(list(articles.values()))
    .drop(columns=['id', 'article_id', 'content_type', 'content', 'author_fulltext'])
    .drop_duplicates()
)
articles_df = (
    articles_df.groupby(['pmid'])
    .apply(lambda group_df: group_df.to_dict('records'))
    .rename('document')
    .reset_index()
    .rename(columns={'pmid': '_pmid'})
)
articles_df['document'] = articles_df.document.str[0]
civic_df = civic_df.merge(articles_df, how='left').drop(columns=['_pmid'])
dump_jsonl(civic_df, args.output)
