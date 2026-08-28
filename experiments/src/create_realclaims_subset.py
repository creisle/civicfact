"""
Given the CIViC-Fact dataset, pull out the examples which require evidence outside of the abstract.

These will be considered our "hard" subset. In particular they should represent a more challenging subset for retrieval

see the evidence_retrieval/filter_civicfact_claims.py script
"""

import argparse

from utils import dump_jsonl, read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()


df = read_jsonl(args.input)
df = df[df["claim.generated"] != True]

dump_jsonl(df[["claim.flat", "document.pmid"]].drop_duplicates(), args.output)
