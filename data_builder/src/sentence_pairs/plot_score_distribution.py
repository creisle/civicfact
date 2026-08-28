import argparse
import logging

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from figutils import savefig

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.input}')
df = pd.read_json(args.input, lines=True, dtype=False)
logging.info(f'loaded {df.shape[0]} rows')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
sns.histplot(data=df, x='similarity.score', ax=ax1)

fig_df = (
    df.groupby('claim.flat')
    .agg({'similarity.score': 'mean'})
    .reset_index()
    .rename(columns={'similarity.score': 'mean_similarity'})
)
sns.histplot(data=fig_df, x='mean_similarity', ax=ax2)
savefig(fig, args.output)
