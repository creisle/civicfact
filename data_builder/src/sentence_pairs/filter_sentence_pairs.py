import argparse
import logging

import pandas as pd

from article_utils.util import dump_jsonl

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument(
    '-n', help='number of sample pairs to keep for each claim', required=True, type=int
)

args = parser.parse_args()


logging.info(f'reading: {args.input}')
df = pd.read_json(args.input, lines=True, dtype=False)
logging.info(f'loaded {df.shape[0]} rows')
df = df.sample(frac=1)  # randomize order
df = df.sort_values(['claim.flat', 'similarity.annotation_match'], ascending=False)

# attempt to make sure the scores for each claim are varied
df['_sim'] = df['similarity.score'].round(1)
df['_sim_index'] = df.groupby(['claim.flat'])[
    '_sim'
].cumcount()  # incidence index of each score per claim

# order by incidence index to ensure a varity of and then keep x pairs
df = df.sort_values('_sim_index')
df['_claim_index'] = df.groupby('claim.flat').cumcount()
df = df[(df._claim_index < args.n) | (df['similarity.document_match'])]
logging.info(f'filtered down to {df.shape[0]} rows')
df = df.drop(columns=[c for c in df.columns if c.startswith('_')])

dump_jsonl(df, args.output)
