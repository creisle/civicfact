"""
Take in the claims paired with the evidence selected by a retrieval model that has already been convereted to
claim/evidence format and pair if with the dataset it is derived from to create the automatic assignments
of examples from the training data.

Since there may be multiple entries in the original dataset but only 1 in the retrieved dataset,
pick the first "examples"
"""

import argparse
import logging
import os

from utils import dump_jsonl, read_jsonl

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--assignments", required=True)
args = parser.parse_args()


df = read_jsonl(
    args.input,
    required_columns=["claim.flat", "evidence.flat", "document.pmid", "gold_label_name"],
    warning_columns=["partition"],
)

if "assignment.round" in df.columns:
    df.drop(columns=["assignment.round"])

if df.gold_label_name.nunique() != 3:
    logging.warning(f"dataset does not have the expected set of 3 labels: {df.gold_label_name.unique().tolist()}")


source_df = read_jsonl(args.assignments, required_columns=["claim.flat", "document.pmid", "prompt_examples"])

source_df = source_df.drop_duplicates(["claim.flat", "document.pmid", "assignment.round"])

original_count = df.shape[0]
rounds = source_df["assignment.round"].nunique()

df = df.merge(source_df[["claim.flat", "document.pmid", "assignment.round", "prompt_examples"]], validate="one_to_many")
assert df.shape[0] == original_count * rounds

dump_jsonl(df, args.output)
