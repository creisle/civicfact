"""
Take the output from the combined retreive and verify task and convert it to make the output format
of the other retrieval models so we can compute downstream stats
"""

import argparse
import logging
import re

from utils import dump_jsonl, read_jsonl

PASSAGE_ORDER_REGEX = re.compile(r"^passages:\s*(none|(\d+(,\s*\d+)*))$", flags=re.IGNORECASE)


def get_passage_order(generated_text):
    for line in generated_text.split("\n"):
        m = PASSAGE_ORDER_REGEX.match(line.strip())
        if m:
            ordering = m.group(1)
            if ordering.lower() == "none":
                return None
            return [int(d.strip()) for d in ordering.split(",")]
    raise ValueError(f"did not find a passage ordering in the output")


def order_chunks(row):
    chunks = row["document.chunks"]  # these are already in order
    scores_by_chunk_id = {}
    if row["predicted_passage_order"] is not None:
        for chunk_score, chunk_index in enumerate(row["predicted_passage_order"][::-1]):
            chunk = chunks[chunk_index]
            scores_by_chunk_id[chunk["chunk.id"]] = chunk_score + 1

    scores = []
    for chunk in chunks:
        chunk = {**chunk, "predicted_similarity.score": scores_by_chunk_id.get(chunk["chunk.id"], 0)}
        scores.append(chunk)
    return scores


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()


df = read_jsonl(
    args.input,
    required_columns=["document.chunks", "id", "claim.flat", "gold_label_name", "generated_text", "template"],
    static_columns=["template"],
)
df["predicted_passage_order"] = df.generated_text.map(get_passage_order)
df["chunks"] = df.apply(lambda row: order_chunks(row), axis=1)
df = df.explode("chunks")
df = df.drop(columns=["document.chunks"])
for col in [
    "chunk.text",
    "chunk.id",
    "chunk.start",
    "chunk.end",
    "chunk.overlap",
    "chunk.size",
    "predicted_similarity.score",
]:
    df[col] = df.chunks.str[col]
df = df.drop(columns=["chunks"])

dump_jsonl(df, args.output)
