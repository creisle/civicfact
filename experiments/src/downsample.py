import argparse
import logging
import os

import pandas as pd

from utils import dump_jsonl, read_jsonl

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--fraction", type=float, default=0.2)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--filter")
parser.add_argument("output")
args = parser.parse_args()


assert args.fraction >= 0 and args.fraction <= 1


df = read_jsonl(args.input, required_columns=["gold_label_name", "partition"])

if args.filter:
    filter_df = read_jsonl(args.filter)
    shared_cols = list(set(df.columns) & set(filter_df.columns))
    assert len(shared_cols) > 0, f"cannot filter on a file without common columns"
    logging.info(f"filtering on join via: {shared_cols}")
    df = df.merge(filter_df[shared_cols].drop_duplicates())

# prioritize unique data from subsampling
if "assignment.round" in df.columns:
    if df["assignment.round"].nunique() > 1:
        assert df["id"].nunique() == df.shape[0] / df["assignment.round"].nunique()
        first_assn = df["assignment.round"].min()
        rows = df.shape[0]
        df = df[df["assignment.round"] == first_assn]
        logging.info(f"reduced rows from {rows} to {df.shape[0]} by dropping redundancy rows")


if args.fraction < 1:
    dfs = []

    for _, subsetdf in df.groupby(["partition", "gold_label_name"]):
        dfs.append(subsetdf.sample(frac=args.fraction, random_state=args.seed))

    df = pd.concat(dfs).reset_index(drop=True)
    logging.info(f"final selection contains {df.shape[0]} rows")

dump_jsonl(df, args.output)
