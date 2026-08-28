"""
Create a copy of the input dataset which uses the corresponding document chunks in addition to the human selected portions
this is to ensure the model is able to deal with extraneous text and mid-word chunks

we can also create a copy of each chunk where we input additional chunks where possible to ensure that the positive examples
length distribution vs NEI makes sense
"""

import argparse
import logging
import random

import pandas as pd

from evidence_retrieval.convert_retrieval_to_verification import (
    convert_chunks_to_evidence,
)
from utils import compute_overlap, concat_text, dump_jsonl, read_jsonl

RANDOM_STATE = 42


def add_distrator_to_chunks(chunks_df, num_distractors=2):
    positives = chunks_df[chunks_df.overlap > 0][["chunk.id", "chunk.start", "chunk.end", "chunk.text"]].to_dict(
        "records"
    )
    negatives = chunks_df[chunks_df.overlap <= 0][["chunk.id", "chunk.start", "chunk.end", "chunk.text"]].to_dict(
        "records"
    )
    if len(negatives) < len(positives) * 2:
        return None

    return convert_chunks_to_evidence(positives + random.sample(negatives, k=num_distractors), [])


all_sections = set()


def assign_section_to_chunk(passages):
    if not hasattr(passages, "__iter__"):
        return None
    global all_sections

    sections = {p["passage.section"] for p in passages}
    all_sections.update(sections)
    if sections & {"abstract", "title"}:
        return "abstract"
    else:
        return "article"


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--document_chunks", required=True)
parser.add_argument("--filterfile", required=True)
parser.add_argument("output")

random.seed(RANDOM_STATE)
args = parser.parse_args()

hard_df = read_jsonl(args.filterfile, required_columns=["claim.flat", "document.pmid"])

chunks_df = read_jsonl(
    args.document_chunks,
    required_columns=[
        "document.content_source",
        "document.length",
        "document.pmid",
        "chunk.start",
        "chunk.end",
        "chunk.text",
        "document.total_chunks",
    ],
)
chunks_df["chunk.section"] = "abstract"
chunks_df.loc[chunks_df["document.content_source"] != "pubmed", "chunk.section"] = chunks_df[
    chunks_df["document.content_source"] != "pubmed"
].passages.map(assign_section_to_chunk)
print(all_sections)
print(chunks_df.columns)
print(chunks_df.passages.tolist()[0])
print(chunks_df["chunk.section"].value_counts(dropna=False))

abstracts_df = (
    chunks_df[chunks_df["chunk.section"] == "abstract"]
    .groupby(["document.pmid"])
    .apply(
        lambda subset_df: convert_chunks_to_evidence(
            subset_df[["chunk.id", "chunk.start", "chunk.end", "chunk.text"]].to_dict("records"), []
        )
    )
    .rename("evidence")
    .reset_index()
)
abstracts_df["evidence.flat"] = abstracts_df["evidence"].map(concat_text)
print(abstracts_df)


input_df = read_jsonl(
    args.input,
    required_columns=["revised_claim.flat", "claim.flat", "gold_label_name", "document.pmid", "evidence"],
    unique_columns=["id"],
)
label_df = input_df.groupby(["claim.flat", "document.pmid"]).agg({"gold_label_name": "nunique"}).reset_index()
label_df = label_df[label_df.gold_label_name == 1][["claim.flat", "document.pmid"]].drop_duplicates()

ev_df = input_df.merge(label_df).explode("evidence")
ev_df["evidence.start"] = ev_df.evidence.str["meta"].str["start"]
ev_df["evidence.end"] = ev_df.evidence.str["meta"].str["end"]
ev_df["evidence.id"] = (
    ev_df["document.pmid"]
    + ":"
    + ev_df["evidence.start"].astype(int).astype(str)
    + "-"
    + ev_df["evidence.end"].astype(int).astype(str)
)

chunks_by_doc = {}
for pmid, chunks_subset_df in chunks_df.groupby("document.pmid"):
    chunks_by_doc[pmid] = chunks_subset_df[["chunk.start", "chunk.end"]].sort_values(["chunk.start"]).to_dict("records")

mapping_df = ev_df.merge(
    chunks_df[["document.pmid", "chunk.start", "chunk.end", "chunk.text", "chunk.id"]], on=["document.pmid"]
)
mapping_df["overlap"] = mapping_df.apply(
    lambda row: compute_overlap((row["evidence.start"], row["evidence.end"]), (row["chunk.start"], row["chunk.end"])),
    axis=1,
)


# create a version of the evidence with chunks instead of direct human selections
df = (
    mapping_df[mapping_df.overlap > 0]
    .sort_values(["chunk.start"])
    .groupby(["id", "document.pmid"])
    .apply(
        lambda subset_df: convert_chunks_to_evidence(
            subset_df[["chunk.id", "chunk.start", "chunk.end", "chunk.text"]].to_dict("records"), []
        )
    )
    .rename("evidence")
    .reset_index()
)
df = input_df.drop(columns=[c for c in input_df.columns if c.startswith("evidence")]).merge(df)
df["evidence.flat"] = df.evidence.map(concat_text)

# now make a version with distactors as well for longer documents
distract_df = (
    mapping_df.sort_values(["chunk.start"])
    .groupby(["id", "document.pmid"])
    .apply(add_distrator_to_chunks)
    .rename("evidence")
    .reset_index()
)
distract_df = distract_df[~distract_df.evidence.isna()]
distract_df = input_df.drop(columns=[c for c in input_df.columns if c.startswith("evidence")]).merge(distract_df)
distract_df["evidence.flat"] = distract_df.evidence.map(concat_text)

# make a version with title/abstract chunks only for claims where it is not verifiable by abstract alone. these should all be NEI examples

hard_df = input_df.merge(hard_df[["claim.flat", "document.pmid"]])
hard_df = hard_df.drop(columns=[c for c in input_df.columns if c.startswith("evidence")]).merge(abstracts_df)
hard_df = hard_df[~hard_df.evidence.isna()]
hard_df["revised_claim.flat"] = ""
hard_df["gold_label_name"] = "NEI"
hard_df = hard_df.drop(columns=["reviews"])

df = pd.concat([input_df, df, distract_df, hard_df])
df = df.drop_duplicates(["claim.flat", "evidence.flat", "document.pmid"], keep="first")
df = df.sample(frac=1, random_state=RANDOM_STATE)

print(df[["partition", "gold_label_name"]].value_counts())

dump_jsonl(df, args.output)
