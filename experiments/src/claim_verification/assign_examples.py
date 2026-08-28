import argparse
import json
import logging
import random

import numpy as np
import pandas as pd

from prompting import EXAMPLES_COLNAME
from utils import dump_jsonl, read_jsonl

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def choose_examples(row, train_df, n: int = 1) -> "list[dict]":
    labels = train_df["gold_label_name"].unique()
    n_label = n // len(labels)
    examples = []
    selected_indices = set()
    for label in labels:
        indices = np.random.choice(
            train_df[
                (train_df.gold_label_name == label)
                & (train_df.partition == "train")
                & (train_df["document.pmid"] != row["document.pmid"])
            ].index,
            size=n_label,
            replace=False,
        )
        selected_indices.update(indices)
        examples.append(train_df.iloc[indices].to_dict("records"))

    result = []
    # maximize diversity in labels when we select the first X
    for i in range(n_label):
        for label_set in examples:
            result.append(label_set[i])

    if len(result) < n:
        indices = np.random.choice(
            train_df[
                (train_df["example.id"] != row["example.id"])
                & (~train_df.index.isin(selected_indices))
                & (train_df["document.pmid"] != row["document.pmid"])
            ].index,
            size=len(examples) - n,
            replace=False,
        )
        result.extend(train_df.iloc[indices].to_dict("records"))
    return result


def assign_examples(df: pd.DataFrame, examples_df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    df = df.copy()
    if n < 1:
        df[EXAMPLES_COLNAME] = df["id"].apply(lambda _: [])
        return df

    df[EXAMPLES_COLNAME] = df.apply(lambda row: choose_examples(row, examples_df, n=n), axis=1)
    return df


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to the JSONL inputs")
    parser.add_argument("output", help="path to a new JSONL which contains the prompts")
    parser.add_argument("--sample", default=None, type=int)
    parser.add_argument(
        "--redundancy",
        help="number of different times each row should be included with different preceding example sets",
        default=1,
        type=int,
    )
    parser.add_argument("--max_examples", type=int, required=True)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--examples_databank")
    g.add_argument("--partition", choices=["dev", "train", "test", "unassigned"])
    args = parser.parse_args()

    if not args.examples_databank:
        assert (
            args.partition
        ), f'either partition or examples databank must be given. If examples databank is given we assume the target partition on the input to be "all"'

        df = read_jsonl(
            args.input,
            required_columns=["partition", "claim.flat", "document.pmid", "id", "gold_label_name"],
            warning_columns=["evidence.flat"],
        )
        logging.info(f"read {df.shape[0]} examples")

        databank_df = df[df.partition == "train"].copy().reset_index(drop=True)
        df = df[df.partition.isin({args.partition})].copy().reset_index(drop=True)

    else:
        df = read_jsonl(
            args.input,
            required_columns=["claim.flat", "document.pmid", "id", "gold_label_name"],
            warning_columns=["evidence.flat"],
        )
        databank_df = read_jsonl(args.examples_databank)

    # explode reasons since this is for LLMs specifically
    if "reviews" in databank_df.columns:
        databank_df["reasons"] = databank_df.reviews.apply(
            lambda reviews: (
                [""] if not hasattr(reviews, "__iter__") else [r["reason"] for r in reviews if r["reason"]]
            )
        ).apply(lambda reasons: [""] if not reasons else reasons)
        databank_df = databank_df.explode("reasons")
        databank_df = databank_df.reset_index(drop=True).rename(columns={"reasons": "reason"})
    elif "reason" not in databank_df.columns:
        databank_df["reason"] = ""

    databank_df["reason.id"] = databank_df.groupby(["id"]).cumcount()
    databank_df["example.id"] = databank_df["id"]
    databank_df["id"] = databank_df["id"] + "_r" + databank_df["reason.id"].astype(str)

    examples_df = databank_df.reset_index(drop=True).copy()
    examples_df = examples_df.sample(frac=1, random_state=RANDOM_SEED)
    examples_df["pmid_index"] = examples_df.groupby("document.pmid").cumcount()

    if args.sample is not None and args.sample < df.shape[0]:
        logging.info(f"downsampling the inputs from {df.shape[0]} examples to {args.sample}")
        examples_df = examples_df.sample(n=args.sample)

    dfs = []

    expected_count = df.shape[0] * args.redundancy
    for i in range(args.redundancy):
        logging.info(f"assigning {args.max_examples} examples round {i + 1}")
        assign_df = assign_examples(df, examples_df, n=args.max_examples)
        assign_df["assignment.round"] = i + 1
        dfs.append(assign_df)
    df = pd.concat(dfs)

    df["redundancy"] = args.redundancy

    assert (
        df.shape[0] == expected_count
    ), f"Expected {expected_count} examples but the resulting file has {df.shape[0]} examples"

    print(df.columns)
    dump_jsonl(df, args.output)


if __name__ == "__main__":
    main()
