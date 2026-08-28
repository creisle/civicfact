# HACK: these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
# TODO: actually upgrade sqlite3
# https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import argparse
import logging

import pandas as pd

# from sentence_transformers.similarity_functions import SimilarityFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from utils import dump_jsonl, read_jsonl

SIM_FUNCTIONS = {"cosine": cosine_similarity, "euclidean": euclidean_distances}


def dedup_list_keep_order(input_list):
    dup_check = set()
    result = []
    for item in input_list:
        if item not in dup_check:
            result.append(item)
            dup_check.add(item)
    return result


def split_list(input_list, chunk_size):
    # https://github.com/chroma-core/chroma/issues/1049#issuecomment-1699859480
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--document_chunks",
    required=True,
    help="jsonl file containing the definitions for every document chunk",
)
parser.add_argument("--pairs", required=True, help="claim/evidence pairs")
parser.add_argument("--output", required=True)
parser.add_argument("--partition", required=True)
parser.add_argument("--similarity_function", choices=SIM_FUNCTIONS, required=True)

pmid_colname = 'document.pmid'

args = parser.parse_args()

chunks_df = read_jsonl(args.document_chunks)
pairs_df = read_jsonl(args.pairs)

if pmid_colname not in pairs_df.columns:
    pmid_colname = 'pmid'
chunks_df = chunks_df.merge(
    pairs_df[[pmid_colname, "partition"]].rename(columns={pmid_colname: "pmid"}).drop_duplicates(), how="left"
)
chunks_df["partition"] = chunks_df["partition"].fillna("")

chunks_df = chunks_df[chunks_df.partition == "train"]
pairs_df = pairs_df[pairs_df.partition == args.partition]
total_input_pairs = pairs_df.shape[0]
logging.info(f"there are {total_input_pairs} claim/chunk pairs")

assert total_input_pairs > 0

vectorizer = TfidfVectorizer()

sim_func = SIM_FUNCTIONS[args.similarity_function]

ref_chunks = chunks_df["chunk.text"].unique()
logging.info(f"fitting {len(ref_chunks)} chunks")

X = vectorizer.fit_transform(ref_chunks)  # Store tf-idf representations of all docs

scores = []
iter_size = 1000  # takes too much mem to do all at once


for i, pairs_set in enumerate(
    split_list(list(pairs_df[["claim.flat", "chunk.text"]].drop_duplicates().itertuples(index=False)), iter_size)
):
    logging.info(f"scoring pairs {i*iter_size}-{i*iter_size + iter_size} of {len(pairs_set)}")
    logging.info(f"embedding {len(pairs_set)} queries and chunks")

    query_vec = vectorizer.transform([p[0] for p in pairs_set])
    chunk_vec = vectorizer.transform([p[1] for p in pairs_set])
    logging.info(f"calculating {args.similarity_function} similarity/distance")
    results = sim_func(query_vec, chunk_vec)

    rows = []
    for p_i in range(len(pairs_set)):
        rows.append(
            {
                "chunk.text": pairs_set[p_i][1],
                "claim.flat": pairs_set[p_i][0],
                "predicted_similarity.score": results[p_i][p_i],
            }
        )
    logging.info(f"calculated {len(rows)} scores")
    scores_df = pd.DataFrame.from_records(rows)
    scores_df["predicted_similarity.function"] = args.similarity_function
    scores.append(scores_df)

result_df = pd.concat(scores)
result_df["model_name"] = "tfidf/tfidf"

input_pairs = pairs_df.shape[0]

result_df = pairs_df.merge(result_df)
assert result_df.shape[0] == input_pairs


dump_jsonl(result_df, args.output)
