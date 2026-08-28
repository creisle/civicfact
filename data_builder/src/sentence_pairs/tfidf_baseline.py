# HACK: these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
# TODO: actually upgrade sqlite3
# https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import argparse
import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from article_utils.utils import dump_jsonl, read_jsonl


def split_list(input_list, chunk_size):
    # https://github.com/chroma-core/chroma/issues/1049#issuecomment-1699859480
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document_chunks",
        required=True,
        help="jsonl file containing the definitions for every document chunk",
    )
    parser.add_argument("--pairs", required=True, help='claim/evidence pairs')
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    chunks_df = read_jsonl(args.document_chunks)
    pairs_df = read_jsonl(args.pairs)

    vectorizer = TfidfVectorizer()

    chunks = chunks_df["chunk.text"].unique()
    logging.info(f"fitting {len(chunks)} chunks")

    X = vectorizer.fit_transform(chunks)  # Store tf-idf representations of all docs

    scores = []
    iter_size = 1000  # takes too much mem to do all at once

    queries = list(pairs_df['claim.flat'].unique())
    for i, queries_chunk in enumerate(split_list(queries, iter_size)):
        logging.info(f"scoring queries {i*iter_size}-{i*iter_size + iter_size} of {len(queries)}")
        logging.info(f"embedding {len(queries_chunk)} queries")
        query_vec = vectorizer.transform(queries_chunk)
        logging.info("calculating cosine similarity")
        results = cosine_similarity(X, query_vec)
        rows = []
        for chunk_i in range(len(chunks)):
            for query_i in range(len(queries_chunk)):
                rows.append(
                    {
                        "chunk.text": chunks[chunk_i],
                        'claim.flat': queries[query_i],
                        "similarity.score": results[chunk_i][query_i],
                    }
                )

        scores_df = pd.DataFrame.from_records(rows)
        scores_df['similarity.function'] = 'tfidf'
        scores.append(pairs_df.merge(scores_df))

    result_df = pd.concat(scores)

    dump_jsonl(result_df, args.output)


if __name__ == "__main__":
    main()
