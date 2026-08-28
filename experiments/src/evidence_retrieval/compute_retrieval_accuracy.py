import argparse
import itertools
import logging

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import average_precision_score

from utils import dump_jsonl, fix_bad_df_merge, read_jsonl


def compute_average_precision_at_k(status, scores, ascending: bool = False, jitter=True):
    # https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map
    epsilon = 1e-8
    y_true = np.array(status)
    y_score = np.array(scores)

    if jitter:
        noise = np.random.uniform(0, epsilon, size=y_score.shape)
        y_score = y_score + noise

    ordering = np.argsort(y_score)
    if not ascending:
        ordering = ordering[::-1]

    # get the rank index of each true value
    rank_true = np.nonzero(y_true[ordering])[0] + 1
    # make an array that counts the number of true elements
    count_true = np.arange(1, rank_true.shape[0] + 1)

    precision = count_true / rank_true

    ap = np.zeros(precision.shape)
    for i in range(precision.shape[0]):
        ap[i] = precision[: i + 1].sum() / count_true[i]
    return [
        {"k": k.item(), "ap": ap_value.item(), "positive": c.item()}
        for k, ap_value, c in zip(rank_true, ap, count_true)
    ]


def bootstrap_average_precision(status, scores, bootstraps, ascending, jitter=True):
    """
    Docstring for compute_ap_at_k

    :param row: Description
    """
    np.random.seed(42)  # For reproducibility
    result = []
    if not jitter:
        # no point in doing multiple if not introducing random element
        bootstraps = 1

    for boot_i in range(bootstraps):

        # fill out the rest of the k values to make downstream life easier, will make last k found
        ap_values = []
        initial_ap_values = sorted(
            compute_average_precision_at_k(status, scores, ascending=ascending, jitter=jitter), key=lambda ap: ap["k"]
        )
        ap_by_k = {ap["k"]: ap for ap in initial_ap_values}

        last_ap = 0
        last_pos = 0

        for k in range(1, len(scores) + 1):
            if k in ap_by_k:
                last_ap = ap_by_k[k]["ap"]
                last_pos = ap_by_k[k]["positive"]

            ap_values.append({"ap": last_ap, "positive": last_pos, "k": k})

        boot_df = pd.DataFrame.from_records(ap_values)
        boot_df["boot_i"] = boot_i
        boot_df["recall"] = boot_df["positive"] / sum(status)
        result.append(boot_df)
    df = pd.concat(result)
    return df


def calculate_retrieval_accuracy(df, reference_file, bootstrap=1, agg_level="claim"):
    # chunks with expected evidence matchups
    ref_df = read_jsonl(reference_file)
    ref_df = fix_bad_df_merge(ref_df)
    ref_df["document.pmid"] = ref_df["document.pmid"].astype(str)

    ID_COLUMNS = ["claim.flat", "document.pmid"]
    if agg_level == "instance":
        ID_COLUMNS += ["id"]

    ref_df = ref_df.groupby(ID_COLUMNS + ["chunk.id", "chunk.section"]).agg({"overlap": "max"}).reset_index()
    ref_df["positive"] = True

    # we need to convert this to agg_level elements
    logging.info(f'the reference chunk dataset has {ref_df["claim.flat"].nunique()} claims')
    # drop any from the refefence set where we do not at least have inner merge on document and claim (not part of our evaluation)
    ref_df = ref_df.merge(df[["claim.flat", "document.pmid"]].drop_duplicates())
    logging.info(
        f'the merged (on claim and document) reference chunk dataset has {ref_df["claim.flat"].nunique()} claims'
    )

    # convert scores to rank for each claim / document pair
    similarity_function = "n/a"

    if "predicted_similarity.function" in df.columns:
        assert (
            df["predicted_similarity.function"].nunique() == 1
        ), f'expected single similarity function: {df["predicted_similarity.function"].unique()}'
        similarity_function = df["predicted_similarity.function"].unique().tolist()[0]
        assert similarity_function in ["euclidean", "cosine"]

    rank_ascending = bool(similarity_function == "euclidean")

    META_DOCUMENT_COLS = [c for c in df.columns if c.startswith("document.") and c not in ID_COLUMNS]

    for col in ["positive", "chunk.section"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.merge(
        ref_df[ID_COLUMNS + ["chunk.id", "positive", "chunk.section"]],
        how="left",
        on=ID_COLUMNS + ["chunk.id"],
    )
    df["positive"] = df["positive"].fillna(False)
    assert not any(
        [c.endswith("_x") for c in df.columns]
    ), f'bad merge columns: {[c for c in df.columns if c.endswith("_x")]}'  # check we didn't bad merge any columns

    rows = []

    groups = list(df.groupby(ID_COLUMNS))

    for id_values, subset_df in groups:
        doc_chunks = subset_df["chunk.id"].nunique()
        assert (
            subset_df["document.total_chunks"].nunique() == 1
        ), f'should only be one valid total_chunks per document: {subset_df["document.total_chunks"].unique()}'
        assert doc_chunks == subset_df["document.total_chunks"].tolist()[0]
        subset_df = subset_df.drop_duplicates(["chunk.id"])
        rank_df = subset_df.copy()
        rank_df["position_score"] = 1 - (rank_df["chunk.start"] / rank_df["chunk.start"].max())

        all_chunks = (
            rank_df[["chunk.id", "chunk.start", "predicted_similarity.score", "positive"]]
            .sort_values(["chunk.start"])
            .to_dict("records")
        )
        positive_chunks = (
            rank_df[rank_df.positive == True][
                ["chunk.id", "chunk.start", "chunk.section", "predicted_similarity.score"]
            ]
            .sort_values(["chunk.start"])
            .to_dict("records")
        )
        if len(positive_chunks) > 20:
            print(id_values)
            print(rank_df[["claim.flat", "document.pmid", "chunk.id", "positive"]])
            print(ref_df[(ref_df["claim.flat"] == id_values[0]) & (ref_df["document.pmid"] == id_values[1])])
            raise AssertionError("something weird is happening here")
        row = {c: v for c, v in zip(ID_COLUMNS, id_values)}
        row.update({c: rank_df[c].tolist()[0] for c in META_DOCUMENT_COLS})
        row["accuracy_level"] = agg_level
        for col in ["revised_claim.flat", "civic.id"]:
            if col in rank_df.columns:
                row[col] = rank_df[col].unique()

        # row["ap_bootstrap"] = args.bootstrap
        row["positive_chunks"] = positive_chunks
        row["all_chunks"] = all_chunks

        rank_df["predicted_similarity.score"] = rank_df["predicted_similarity.score"] + (
            rank_df["position_score"] / 1e8
        )  # to break ties, pick first element from page

        if positive_chunks:
            ap_df = bootstrap_average_precision(
                rank_df["positive"].astype(int).tolist(),
                rank_df["predicted_similarity.score"].tolist(),
                jitter=False,
                bootstraps=bootstrap,
                ascending=rank_ascending,
            )
            ap_df = ap_df.groupby(["k"]).agg({"ap": "mean", "recall": "mean", "positive": "mean"}).reset_index()
            row["ap@k"] = ap_df.to_dict("records")

        rows.append(row)

    df = pd.DataFrame.from_records(rows)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--reference", required=True)
    parser.add_argument("output")
    parser.add_argument("--agg_level", choices=["instance", "claim"], default="claim")
    parser.add_argument("--bootstrap", default=10, type=int)
    args = parser.parse_args()

    assert args.bootstrap >= 1

    df = read_jsonl(args.input)
    logging.info(f'the input dataset has {df["claim.flat"].nunique()} claims')
    df = fix_bad_df_merge(df)
    df["document.pmid"] = df["document.pmid"].astype(str)
    if "chunk.section" in df.columns:
        # get this from the reference chunks file instead
        df = df.drop(columns=["chunk.section"])

    df = calculate_retrieval_accuracy(df, args.reference, bootstrap=args.bootstrap, agg_level=args.agg_level)

    dump_jsonl(df, args.output)
