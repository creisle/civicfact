"""
first order by most overlap with human picks, then by position in document
This is a simple baseline to measure against
"""

import argparse
import logging

import pandas as pd

from utils import dump_jsonl, fix_bad_df_merge, read_jsonl

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True)
parser.add_argument("--pairs", required=True, help="claim/evidence pairs")
parser.add_argument("--output", required=True)
parser.add_argument(
    "--oracle_only", default=False, help="exclude all pairs except those annotated by the human", action="store_true"
)
parser.add_argument("--partition", required=True)

pmid_colname = "document.pmid"

args = parser.parse_args()

ref_df = read_jsonl(args.reference)
ref_df = fix_bad_df_merge(ref_df)
ref_df["document.pmid"] = ref_df["document.pmid"].astype(str)

ID_COLUMNS = ["claim.flat", "document.pmid"]
ref_df = ref_df.groupby(ID_COLUMNS + ["chunk.id", "chunk.section"]).agg({"overlap": "max"}).reset_index()
ref_df["positive"] = True

# chunks_df = read_jsonl(args.document_chunks)
pairs_df = fix_bad_df_merge(read_jsonl(args.pairs))

print(pairs_df.columns)

if args.partition != "all":
    pairs_df = pairs_df[pairs_df.partition == args.partition]
print(pairs_df.columns)
total_input_pairs = pairs_df.shape[0]
logging.info(f"there are {total_input_pairs} claim/chunk pairs")

pairs_df = pairs_df.merge(ref_df, how="left", validate="m:1")

assert total_input_pairs > 0

all_scores = []

for (claim, document_id), subset_df in pairs_df.groupby(["claim.flat", "document.pmid"]):

    scores_df = (
        subset_df[["chunk.text", "chunk.id", "chunk.start", "overlap"]]
        .drop_duplicates()
        .sort_values(["chunk.start"])
        .drop_duplicates(["chunk.start"], keep="first")
    )
    if args.oracle_only:
        scores_df = scores_df[scores_df.overlap > 0]

    max_overlap = scores_df["overlap"].max()
    scores_df["relative_overlap"] = scores_df["overlap"].fillna(0) / max_overlap
    scores_df["relative_pos"] = (scores_df["chunk.start"].max() - scores_df["chunk.start"]) / scores_df[
        "chunk.start"
    ].max()
    scores_df["predicted_similarity.score"] = scores_df["relative_overlap"].fillna(0) + scores_df["relative_pos"] * (
        1e-6
    )
    scores_df = scores_df.drop(columns=["chunk.start"])
    scores_df["document.pmid"] = document_id
    scores_df["claim.flat"] = claim
    scores_df = scores_df.drop(columns=["overlap"])
    all_scores.append(scores_df)

result_df = pd.concat(all_scores)
print(result_df)


input_pairs = pairs_df.shape[0]

print(set(result_df.columns) & set(pairs_df.columns))
result_df = pairs_df.merge(result_df)

if not args.oracle_only:
    assert result_df.shape[0] == input_pairs, f"post-merging change: {result_df.shape[0]} vs {input_pairs}"


dump_jsonl(result_df, args.output)
