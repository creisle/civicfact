"""
Make a list of long documents only

filter out pubmed abstracts and civic descriptions as documents to begin with, only look at full-text articles

Filter documents on being >= the quantile of all document chunk amounts
"""

import argparse

from utils import dump_jsonl, read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--min_quantile", type=float, required=True)
parser.add_argument("output")
args = parser.parse_args()

assert args.min_quantile >= 0 and args.min_quantile <= 1

df = read_jsonl(args.input, required_columns=["document.total_chunks", "document.content_source", "evidence.source"])
df = df[(df["document.content_source"] != "pubmed") & (df["evidence.source"] != "civic")]

min_size = df.drop_duplicates(["document.pmid"])["document.total_chunks"].quantile(args.min_quantile)
df = df[df["document.total_chunks"] >= min_size]

dump_jsonl(df[["document.pmid"]].drop_duplicates(), args.output)
