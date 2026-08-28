"""
retrieves chunks by position rather than score. This is a simple baseline to measure against
"""

import argparse
import logging

import pandas as pd

from utils import dump_jsonl, read_jsonl

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--pairs", required=True, help="claim/evidence pairs")
parser.add_argument("--output", required=True)
parser.add_argument("--partition", required=True)

pmid_colname = "document.pmid"

args = parser.parse_args()

# chunks_df = read_jsonl(args.document_chunks)
pairs_df = read_jsonl(args.pairs)

print(pairs_df.columns)

if args.partition != "all":
    pairs_df = pairs_df[pairs_df.partition == args.partition]
total_input_pairs = pairs_df.shape[0]
logging.info(f"there are {total_input_pairs} claim/chunk pairs")

assert total_input_pairs > 0

all_scores = []

for document_id, subset_df in pairs_df.groupby("document.pmid"):

    document_chunks = (
        subset_df[["chunk.text", "chunk.start"]]
        .drop_duplicates()
        .sort_values(["chunk.start"])
        .drop_duplicates(["chunk.text"], keep="first")
    )
    claims = subset_df[["claim.flat"]].drop_duplicates()
    scores_df = document_chunks.merge(claims, how="cross")
    scores_df["predicted_similarity.score"] = (scores_df["chunk.start"].max() - scores_df["chunk.start"]) / scores_df[
        "chunk.start"
    ].max()
    scores_df = scores_df.drop(columns=["chunk.start"])
    scores_df["document.pmid"] = document_id
    all_scores.append(scores_df)

result_df = pd.concat(all_scores)
print(result_df)


input_pairs = pairs_df.shape[0]

result_df = pairs_df.merge(result_df)
assert result_df.shape[0] == input_pairs, f"post-merging change: {result_df.shape[0]} vs {input_pairs}"


dump_jsonl(result_df, args.output)
