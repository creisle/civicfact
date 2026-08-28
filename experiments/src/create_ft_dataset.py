import argparse

from utils import dump_jsonl, read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("document_chunks")
parser.add_argument("output")
args = parser.parse_args()

df = read_jsonl(args.input)
chunks_df = read_jsonl(args.document_chunks)

chunks_df = (
    chunks_df.groupby(["document.pmid", "document.content_source"])
    .apply(
        lambda subset_df: subset_df[[c for c in subset_df.columns if c.startswith("chunk.")]]
        .sort_values(["chunk.start"])
        .to_dict("records")
    )
    .rename("document.chunks")
    .reset_index()
)
print(chunks_df)
df = df.merge(chunks_df)
dump_jsonl(df, args.output)
