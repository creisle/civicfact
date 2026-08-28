import argparse
import logging

import bm25s
import pandas as pd

from utils import apply_query_template, dump_jsonl, read_jsonl


def calculate_bm25(document_chunks, claims, method, query_col="claim.flat", passage_col="chunk.text") -> pd.DataFrame:
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25(corpus=document_chunks, method=method)
    retriever.index(bm25s.tokenize(document_chunks))

    # Query the corpus and get top-k results
    selected, scores = retriever.retrieve(bm25s.tokenize(claims), k=len(document_chunks))

    results = []

    for claim_i, claim in enumerate(claims):
        for selection, score in zip(selected[claim_i], scores[claim_i]):
            results.append({query_col: claim, "predicted_similarity.score": score, passage_col: selection})

    df = pd.DataFrame.from_records(results)
    df["model_name"] = "bm25-" + retriever.method
    return df


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--pairs", required=True, help="claim/evidence pairs")
parser.add_argument("--output", required=True)
parser.add_argument("--partition", required=True)
parser.add_argument(
    "--method", required=True, choices=["robertson", "atire", "bm25l", "bm25+", "lucene"]
)  # https://pypi.org/project/bm25s/
parser.add_argument("--query_template", required=True, type=int)

pmid_colname = "document.pmid"

args = parser.parse_args()

# chunks_df = read_jsonl(args.document_chunks)
pairs_df = read_jsonl(args.pairs)

print(pairs_df.columns)

pairs_df = pairs_df[pairs_df.partition == args.partition]
pairs_df["_query_text"] = pairs_df.apply(
    lambda row: apply_query_template(row, "claim.flat", args.query_template), axis=1
)
total_input_pairs = pairs_df.shape[0]
logging.info(f"there are {total_input_pairs} claim/chunk pairs")

assert total_input_pairs > 0

all_scores = []

for document_id, subset_df in pairs_df.groupby("document.pmid"):
    document_chunks = subset_df.drop_duplicates("chunk.id")["chunk.text"].unique().tolist()
    claims = subset_df["_query_text"].unique().tolist()
    scores_df = calculate_bm25(document_chunks, claims, args.method, query_col="_query_text", passage_col="chunk.text")
    scores_df["document.pmid"] = document_id
    all_scores.append(scores_df)

result_df = pd.concat(all_scores)
print(result_df)


input_pairs = pairs_df.shape[0]

result_df = pairs_df.merge(result_df)
assert result_df.shape[0] == input_pairs, f"post-merging change: {result_df.shape[0]} vs {input_pairs}"


dump_jsonl(result_df.drop(columns=["_query_text"]), args.output)
