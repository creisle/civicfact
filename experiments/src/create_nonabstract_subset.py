"""
Given the CIViC-Fact dataset, pull out the examples which require evidence outside of the abstract.

These will be considered our "hard" subset. In particular they should represent a more challenging subset for retrieval

see the evidence_retrieval/filter_civicfact_claims.py script
"""

import argparse

from utils import dump_jsonl, read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument(
    "retrieval_reference",
    help="path to the file containing the expected chunks to be retrieved for each claim and their metadata",
)
parser.add_argument("output")
args = parser.parse_args()


ref_df = read_jsonl(args.retrieval_reference)

ref_df["in_abstract"] = ref_df["chunk.section"] == "abstract"
ref_df = ref_df.groupby(["id", "claim.flat", "document.pmid"]).agg({"in_abstract": "all"}).reset_index()
print(ref_df["in_abstract"].value_counts())
ref_df = ref_df.groupby(["claim.flat", "document.pmid"]).agg({"in_abstract": "any"}).reset_index()
ref_df = ref_df[ref_df.in_abstract == False]

df = read_jsonl(args.input)
print(df.gold_label_name.value_counts())
df = df.merge(ref_df[["claim.flat", "document.pmid"]])
print(df.gold_label_name.value_counts())

dump_jsonl(df[["claim.flat", "document.pmid"]].drop_duplicates(), args.output)
