import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from utils import EXPECTED_LABELS, dump_jsonl, read_jsonl


def boostrap_mean(series):
    means = []
    for i in range(1000):
        boot_df = series.sample(n=len(series), replace=True, random_state=i)
        means.append(boot_df.mean())
    means = sorted(means)
    return means


def pick_label(labels):
    hist = {}
    for l in labels:
        hist[l] = hist.get(l, 0) + 1

    labels = sorted(hist.items(), key=lambda x: x[1], reverse=True)
    return labels[0][0]


def normalize_rounds(df):
    df = df.copy()

    if "generated_text" in df.columns:
        df["missing"] = df.generated_text.isna()
    else:
        df["missing"] = False

    id_columns = ["id", "document.pmid", "claim.flat", "evidence.flat", "gold_label_name"]

    if "example.id" in df.columns:
        id_columns[0] = "example.id"

    id_columns = [c for c in id_columns if c in df.columns]
    if "claim.flat" not in id_columns:
        id_columns.append("prompt")

    file_columns = ["filename"] + [c for c in df.columns if c.startswith("filepath")]
    if "reason.id" in df.columns and "assignment.round" in df.columns:
        # if reason.id is present, merge across reasons since they only affect the training
        df = (
            df.groupby(file_columns + id_columns + ["assignment.round"], dropna=False)
            .agg({"predicted_label_name": pick_label, "missing": "all"})
            .reset_index()
        )
    elif "assignment.round" not in df.columns:
        if "reason.id" in df.columns:
            raise NotImplementedError(
                f"not sure how to handle a reason column but no assignment column: {sorted(df.columns.tolist())}"
            )
        df["assignment.round"] = df.groupby(id_columns, dropna=False).cumcount() + 1
    # add asiignment round variable if not present
    if df[df["assignment.round"].isna()].shape[0] > 0:
        print(df[df["assignment.round"].isna()])
        raise AssertionError("some example rows did not have a valid assignment")
    df["freq"] = 1
    df["correct"] = df.gold_label_name == df.predicted_label_name
    df["plausible"] = df.predicted_label_name.isin(EXPECTED_LABELS)
    return df


def compute_round_accuracy(df):
    result = {
        "correct": df.correct.sum(),
        "plausible": df.plausible.sum(),
        "missing": df.missing.sum(),
        "freq": df.freq.sum(),
        "claims": df["claim.flat"].nunique(),
        "documents": df["document.pmid"].nunique(),
    }

    labels = df.gold_label_name.dropna().unique()
    scores = precision_recall_fscore_support(
        df.gold_label_name, df.predicted_label_name, average="macro", labels=labels, zero_division=0
    )

    for score, value in zip(["precision", "recall", "fbeta_score", "freq"], scores):
        result[f"macro_avg.{score}"] = value

    scores = precision_recall_fscore_support(
        df.gold_label_name, df.predicted_label_name, labels=labels, average=None, zero_division=0
    )

    # recall == accuracy when computing recall/accuracy per label
    for score, value in zip(["precision", "accuracy", "fbeta_score", "freq"], scores):
        for lbl_i, lbl in enumerate(labels):
            result[f"class_{lbl}.{score}"] = value[lbl_i]

    result["accuracy"] = result["correct"] / result["freq"]
    result["plausibility"] = result["plausible"] / result["freq"]

    return result


def _compute_accuracy(df):
    file_columns = ["filename"] + [c for c in df.columns if c.startswith("filepath")]
    rows = []
    grouping_cols = file_columns + ["assignment.round"]
    for grouping_vals, subset_df in df.groupby(grouping_cols, dropna=False):
        round = compute_round_accuracy(subset_df)
        round["boot_i"] = None
        for col, val in zip(grouping_cols, grouping_vals):
            round[col] = val
        rows.append(round)
        for i in range(1000):
            boot_df = subset_df.sample(n=subset_df.shape[0], replace=True, random_state=i)
            boot = compute_round_accuracy(boot_df)
            for col, val in zip(grouping_cols, grouping_vals):
                boot[col] = val
            boot["boot_i"] = i
            rows.append(boot)

    return pd.DataFrame.from_records(rows)


def compute_accuracy(df, gold_labels):
    rounds_df = normalize_rounds(df)

    result_df = _compute_accuracy(rounds_df)

    for label in gold_labels:
        for col in [f"class_{label}.accuracy", f"class_{label}.freq"]:
            if col not in result_df.columns:
                logging.warning(f"missing col: {col}")
                result_df[col] = None

    result_df["macro_avg.accuracy"] = result_df[[f"class_{c}.accuracy" for c in gold_labels]].mean(axis=1)
    result_df["majority_class.freq"] = result_df[[f"class_{c}.freq" for c in gold_labels]].max(axis=1)
    result_df["majority_class.accuracy"] = result_df["majority_class.freq"] / result_df["freq"]
    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--filter", help="subset of claim/document pairs to compute on")
    args = parser.parse_args()
    df = read_jsonl(
        args.input,
        add_filename=True,
        add_filepath_metadata=True,
        required_columns=["claim.flat", "document.pmid", "predicted_label_name", "gold_label_name"],
    )
    gold_labels = df.gold_label_name.unique().tolist()
    if len(gold_labels) != 3:
        logging.warning(f"did not find 3 different gold labels: {gold_labels}")

    df["document.pmid"] = df["document.pmid"].astype(str)
    if args.filter:
        filter_df = read_jsonl(args.filter, required_columns=["claim.flat", "document.pmid"])[
            ["claim.flat", "document.pmid"]
        ].drop_duplicates()
        filter_df["document.pmid"] = filter_df["document.pmid"].astype(str)
        df = df.merge(filter_df)
    df["ft"] = bool("fine-tuning" in args.input)
    df = compute_accuracy(df, gold_labels=gold_labels)
    dump_jsonl(df, args.output)
