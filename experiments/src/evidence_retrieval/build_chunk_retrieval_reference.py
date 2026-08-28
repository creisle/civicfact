"""
Combine the original civicfact dataset with the document chunks to have per chunk annotations of which chunks overlap expected annotations
"""

import argparse
import logging

from utils import compute_overlap, dump_jsonl, read_jsonl


def flatten_evidence_chunk(chunk):
    if chunk["type"] != "table":
        return chunk["content"]
    header = []
    row = []
    for col, val in chunk["content"]:
        header.append(col)
        row.append(val)

    return "\n".join(["\t".join(header), "\t".join(row)])


def simplify_chunk_section(evidence_sections):
    section_mapping = {
        "results and discussion": ["result", "discussion", "finding", "conclusion", "interpretation"],
        "methods": ["material", "method", "experimental design"],
        "background": ["background", "intro", "purpose", "objective"],
    }
    evidence_sections = evidence_sections.dropna()
    if len(evidence_sections) == 0:
        return None

    sections = set()
    for secs in evidence_sections.unique():
        for sec in secs.split(";"):
            main_sec = sec.split("//")[0]
            subsection = sec.split("//")[-1]
            simplified = None
            if main_sec in {"abstract", "title"}:
                sections.add("abstract")
                continue

            for sec_name, key_words in section_mapping.items():
                if any([word in subsection for word in key_words]):
                    simplified = sec_name
                    break
            if simplified:
                sections.add(simplified)
            else:
                sections.add(main_sec)
    sections = sections - {"None"}
    if "abstract" in sections or "title" in sections:
        return "abstract"

    if len(sections) > 1:
        # remove most non-specific part
        sections -= {"article"}

    return ";".join(sorted(sections))


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("fact_data")
parser.add_argument("chunks")
parser.add_argument("output")
args = parser.parse_args()

fact_df = read_jsonl(args.fact_data)
fact_df = fact_df[(fact_df["evidence.source"] != "civic")]  # no true expected retrieval for NEI results
fact_df = fact_df[
    (fact_df.gold_label_name != "NEI") | (fact_df["evidence.selection_method"] == "hypothesis")
]  # no true expected retrieval for NEI results
fact_df = fact_df[
    (fact_df["claim.source"].str.count("\+") < 3)
]  # only allow elements with at most 2 element revisions required
print(fact_df["claim.source"].value_counts(dropna=False))

print(
    fact_df.groupby(
        ["evidence.source", "evidence.generated", "evidence.selection_method", "claim.generated", "gold_label_name"]
    ).agg({"claim.flat": "nunique", "claim.source": "unique"})
)

fact_df["evidence.parts"] = fact_df.evidence.str.len()
fact_df = fact_df.explode("evidence")
fact_df["evidence.start"] = fact_df.evidence.str["meta"].str["start"]
fact_df["evidence.end"] = fact_df.evidence.str["meta"].str["end"]
fact_df["evidence.content"] = fact_df.evidence.str["content"]
fact_df["evidence.type"] = fact_df.evidence.str["type"]
fact_df["evidence.section"] = fact_df.evidence.str["meta"].str["section"]
fact_df["evidence.text"] = fact_df.evidence.apply(flatten_evidence_chunk)

print(fact_df)
print(fact_df["evidence.section"].value_counts(dropna=False))
chunks_df = read_jsonl(args.chunks)
assert chunks_df["chunk.id"].nunique() == chunks_df.shape[0], f"chunk.id must be a unique index of: {args.chunks}"
chunks_df = chunks_df[
    [
        "chunk.id",
        "chunk.start",
        "chunk.end",
        "chunk.size",
        "chunk.overlap",
        "document.total_chunks",
        "document.pmid",
        "document.content_source",
    ]
]

df = fact_df.merge(chunks_df, on="document.pmid")
df["overlap"] = df.apply(
    lambda row: compute_overlap((row["evidence.start"], row["evidence.end"]), (row["chunk.start"], row["chunk.end"])),
    axis=1,
)
df = df[df.overlap > 0]

df = df[
    [
        "chunk.id",
        "chunk.start",
        "chunk.end",
        "chunk.size",
        "chunk.overlap",
        "document.total_chunks",
        "document.pmid",
        "document.content_source",
        "evidence.start",
        "evidence.end",
        "evidence.text",
        "evidence.parts",
        "evidence.selection_method",
        "evidence.source",
        "evidence.generated",
        "evidence.section",
        "id",
        "overlap",
        "evidence.type",
        "gold_label_name",
        "claim.flat",
    ]
].drop_duplicates()

chunk_section_df = (
    df.groupby(["chunk.id"])
    .agg({"evidence.section": simplify_chunk_section})
    .reset_index()
    .rename(columns={"evidence.section": "chunk.section"})
)
print(chunk_section_df["chunk.section"].value_counts(dropna=False))

df = df.merge(chunk_section_df, how="left")

print(
    df.groupby(
        ["evidence.source", "evidence.selection_method", "evidence.start", "evidence.end", "chunk.start", "chunk.end"]
    ).agg({"claim.flat": "nunique"})
)
dump_jsonl(df, args.output)
