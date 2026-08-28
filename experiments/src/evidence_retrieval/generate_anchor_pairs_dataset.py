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

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--documents", required=True)
parser.add_argument("--reference", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--query_template", type=int, required=True)
args = parser.parse_args()

# FACTS_FILE = "data/civicfact-2026.03.03/data.jsonl.gz"
# DOC_FILE = "data/civicfact-2026.03.03/document_chunks.jsonl.gz"
# REF_CHUNKS = "data/civicfact-2026.03.03/chunk_retrieval_reference.jsonl.gz"
# QUERY_TEMPLATE = 4
# MODEL_NAME = "Qwen/Qwen3-Reranker-4B"

logging.basicConfig(level=logging.INFO)
query_df = read_jsonl(args.data)
query_df["anchor"] = query_df.apply(
    lambda row: apply_query_template(row, query_column="claim.flat", template=args.query_template), axis=1
)
print(query_df)

docs_df = read_jsonl(args.documents)
print(docs_df)

ref_df = read_jsonl(args.reference)
# add the newly formatted anchor column
ref_df = ref_df.merge(query_df[["anchor", "claim.flat", "document.pmid"]].drop_duplicates(), how="left")
ref_df = ref_df.merge(docs_df[["chunk.text", "chunk.id"]], how="left", validate="many_to_one")
assert ref_df.anchor.isna().sum() == 0
ref_df = ref_df.rename(columns={"chunk.text": "positive"})
print(ref_df.columns.tolist())
ref_df = ref_df.groupby(["anchor", "document.pmid"]).agg({"positive": "unique"}).reset_index()

docs_df = docs_df.rename(columns={"chunk.text": "documents"})
docs_df = docs_df.groupby(["document.pmid", "partition"], dropna=False).agg({"documents": "unique"}).reset_index()

ref_df = ref_df.merge(docs_df, how="left", validate="many_to_one")
print(ref_df)
print(ref_df.partition.value_counts(dropna=False))
# model = SentenceTransformer(MODEL_NAME)

# result = []
# for (pmid, partition), pmid_df in tqdm.tqdm(
#     ref_df.groupby(["document.pmid", "partition"]), total=int(ref_df["document.pmid"].nunique())
# ):
#     pmid_df = (
#         pmid_df[["chunk.text", "claim.flat"]]
#         .drop_duplicates()
#         .rename(columns={"claim.flat": "query", "chunk.text": "answer"})
#     )
#     ref_ds = Dataset.from_pandas(pmid_df[["query", "answer"]])
#     # determine which chunks overlap evidence for a given claim. These should be considered positive pairs. Note we need to exclude any generated evidence pairings

#     neg_ds = mine_hard_negatives(
#         dataset=ref_ds,
#         model=model,
#         corpus=docs_df[docs_df["document.pmid"] == pmid]["chunk.text"].unique().tolist(),
#         range_min=8,
#         range_max=50,
#         max_score=0.8,
#         relative_margin=0.05,
#         num_negatives=5,
#         sampling_strategy="random",
#         batch_size=128,
#         use_faiss=True,
#         verbose=False,
#         include_positives=False,
#     )

#     output_df = neg_ds.to_pandas().drop_duplicates()
#     output_df["document.pmid"] = pmid
#     output_df["partition"] = partition
#     result.append(output_df)
# output_df = pd.concat(result)
# positives_df = output_df[["query", "answer"]].drop_duplicates()
# positives_df["positive"] = True
# positives_df = positives_df.rename(columns={"answer": "negative"})
# output_df = output_df.merge(positives_df, how="left")
# print(output_df)
# logging.warning(
#     f"dropping {output_df[output_df.positive == True].shape[0]} rows where a negative matches an existing positive"
# )
# output_df = output_df[output_df.positive != True]
# output_df = output_df.drop(columns=["positive"])
dump_jsonl(ref_df, args.output)

pass
