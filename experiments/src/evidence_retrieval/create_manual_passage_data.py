import argparse

from utils import dump_jsonl, read_jsonl
import logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('document_chunks')
parser.add_argument('fact_data')
parser.add_argument('output')
args = parser.parse_args()
chunks_df = read_jsonl(args.document_chunks)

fact_df = read_jsonl(args.fact_data)

if 'document.pmid' not in chunks_df.columns:
    # versions < 2025.01.14
    df = fact_df.merge(
        chunks_df[["chunk.id", "pmid", "chunk.start", "chunk.end", "chunk.text"]].rename(columns={"pmid": "document.pmid"}),
        on="document.pmid",
    )
else:
    chunks_cols = [c for c in chunks_df.columns if c.startswith('chunk.') or c.startswith('document.')]
    df = fact_df.merge(chunks_df[chunks_cols], on="document.pmid")

dump_jsonl(df, args.output)
