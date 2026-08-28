"""
Given all document chunks and the input CIViC-Fact dataset create a set of pairs for training a cross-encoder retrieval models like

(claim, positive_chunk, negative_chunk)
"""

import argparse
import logging

import pandas as pd
import tqdm
import transformers
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

from utils import apply_query_template, dump_jsonl, read_jsonl


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("factdata")
    # parser.add_argument("documents")
    # parser.add_argument("output")
    # args = parser.parse_args()

    FACTS_FILE = "data/civicfact-2026.03.03/data.jsonl.gz"
    DOC_FILE = "data/civicfact-2026.03.03/document_chunks.jsonl.gz"
    REF_CHUNKS = "data/civicfact-2026.03.03/chunk_retrieval_reference.jsonl.gz"
    QUERY_TEMPLATE = 0
    MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
    MAX_HARD_NEGATIVES = 5

    logging.basicConfig(level=logging.INFO)
    query_df = read_jsonl(FACTS_FILE)
    docs_df = read_jsonl(DOC_FILE)
    ref_df = read_jsonl(REF_CHUNKS).merge(docs_df[["chunk.id", "chunk.text", "partition"]].drop_duplicates())

    ref_df = (
        ref_df.groupby(["claim.flat", "partition"])
        .agg({"chunk.text": "unique"})
        .reset_index()
        .rename(columns={"chunk.text": "positive"})
    )

    query_df = query_df.merge(ref_df)

    docs_df = docs_df.merge(query_df[["claim.flat", "document.pmid"]].drop_duplicates())
    docs_df = (
        docs_df.groupby(["claim.flat", "partition"])
        .agg({"chunk.text": "unique"})
        .reset_index()
        .rename(columns={"chunk.text": "documents"})
    )

    query_df = query_df.merge(docs_df)

    query_df["anchor"] = query_df.apply(
        lambda row: apply_query_template(row, query_column="claim.flat", template=QUERY_TEMPLATE), axis=1
    )

    # create the hard negatives

    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={"torch_dtype": "float32"},
    )

    negatives = []
    for pmid, pmid_df in tqdm.tqdm(query_df.groupby("document.pmid"), total=int(query_df["document.pmid"].nunique())):
        pmid_df = pmid_df.explode("positive")
        ref_ds = Dataset.from_pandas(pmid_df[["anchor", "positive"]].drop_duplicates())
        corpus = list(pmid_df["documents"].tolist()[0])
        negatives = set(corpus) - set(pmid_df["positive"].unique())
        if not negatives:
            logging.warning(f"skipping: insuffucient chunks")
            continue

        # determine which chunks overlap evidence for a given claim. These should be considered positive pairs. Note we need to exclude any generated evidence pairings

        neg_ds = mine_hard_negatives(
            dataset=ref_ds,
            model=model,
            anchor_column_name="anchor",
            positive_column_name="positive",
            corpus=corpus,
            range_min=10,
            range_max=100,
            max_score=0.8,
            relative_margin=0.05,
            num_negatives=min([len(corpus), MAX_HARD_NEGATIVES]),
            sampling_strategy="top",
            batch_size=128,
            use_faiss=True,
            verbose=False,
            include_positives=False,
        )

        output_df = neg_ds.to_pandas().drop_duplicates()
        output_df["document.pmid"] = pmid
        negatives.append(output_df)
    neg_df = pd.concat(negatives)
    print(neg_df)
    # exclude negatives that are also in positives
    pos_df = query_df.explode("positive")[["anchor", "positive", "document.pmid"]].rename(
        columns={"positive": "negative"}
    )
    pos_df["in_pos"] = True
    neg_df = neg_df.merge(pos_df, how="left")
    drop_n = neg_df.in_pos.sum()
    if drop_n:
        logging.warning(f"dropping {drop_n} negatives as they conflict with positives")

    neg_df = neg_df[neg_df.in_pos != True]
    neg_df = neg_df.groupby(["document.pmid", "anchor"]).agg({"negative": "unique"}).reset_index()

    query_df = query_df.merge(neg_df, how="left")

    print(query_df[["anchor", "positive", "negative", "documents", "partition"]])
    dump_jsonl(query_df[["anchor", "positive", "negative", "documents", "partition"]], "anchor.jsonl.gz")

    pass


if __name__ == "__main__":
    main()
