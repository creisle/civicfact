import argparse
import logging

import pandas as pd

from article_utils.util import dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('--scores', required=True, nargs='+')
parser.add_argument('--output', required=True)
parser.add_argument('--partitions', required=True)

args = parser.parse_args()

logging.info(f'reading: {args.partitions}')
partitions_df = pd.read_csv(args.partitions, dtype={'pmid': 'str'}).rename(
    columns={'pmid': 'document.pmid'}
)

dfs = []
for input_file in args.scores:
    logging.info(f'reading: {input_file}')
    df = pd.read_json(input_file, lines=True, dtype=False)
    dfs.append(df)

df = pd.concat(dfs)
total_rows = df.shape[0]
df = df.merge(partitions_df[['document.pmid', 'partition']])
assert total_rows == df.shape[0]

# for each input file, read in and filter to keep k-similarity matches
dump_jsonl(df, args.output)
