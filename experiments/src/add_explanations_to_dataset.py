import argparse
import logging
import os

from utils import dump_jsonl, read_jsonl

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("explanations")
parser.add_argument("output")
args = parser.parse_args()


df = read_jsonl(args.input, required_columns=["claim.flat", "evidence.flat", "document.pmid", "id"])
if "reviews" in df:
    df = df.drop(columns=["reviews"])
ex_df = read_jsonl(
    args.explanations,
    required_columns=["claim.flat", "evidence.flat", "document.pmid", "generated_text", "request.model"],
    add_filepath_metadata=True,
)
print(ex_df[["claim.flat", "evidence.flat", "document.pmid", "generated_text"]])
ex_df = ex_df.drop_duplicates(["claim.flat", "evidence.flat", "document.pmid", "generated_text"])

ex_df["reviews"] = ex_df.apply(
    lambda row: [
        {
            "reason": row["generated_text"],
            "generation_config": row["filepath.generation_config"],
            "model": row["request.model"],
        }
    ],
    axis=1,
)

mapping_df = df.merge(ex_df[["claim.flat", "evidence.flat", "document.pmid", "reviews"]])[
    ["claim.flat", "document.pmid", "id", "reviews"]
]

df = df.merge(mapping_df[["claim.flat", "id", "document.pmid", "reviews"]], how="left")


dump_jsonl(df, args.output)
