import argparse
import logging
import uuid

from utils import Text, concat_text, dump_jsonl, read_jsonl


def compute_overlap(range1, range2):
    overlap = min(range1[1], range2[1]) - max(range1[0], range2[0])
    if overlap < 0:
        return 0
    return overlap


def merge_overlapping_chunks(chunks, chunk_metadata_cols=["predicted_similarity.index", "predicted_similarity.score"]):
    chunks = sorted(chunks, key=lambda chunk: chunk["chunk.start"])

    for chunk in chunks:
        for col in chunk_metadata_cols:
            chunk[col] = [chunk[col]]

    merged_chunks = []

    for chunk in chunks:
        if not merged_chunks:
            merged_chunks.append(chunk)
        elif merged_chunks[-1]["chunk.end"] == chunk["chunk.start"]:
            # chunks are adjacent, in theory this should not be possible given the way we built the chunks
            raise NotImplementedError("chunks should not be possible to be adjacent")
        elif merged_chunks[-1]["chunk.end"] < chunk["chunk.start"]:
            # chunks do not overlap
            merged_chunks.append(chunk)
        elif merged_chunks[-1]["chunk.end"] >= chunk["chunk.end"]:
            # next chunk is less than overlap so it is completely contained in the first chunk
            merged_chunks[-1] = merged_chunks[-1].copy()
            merged_chunks[-1]["chunk.id"] += " " + chunk["chunk.id"]

            for col in chunk_metadata_cols:
                merged_chunks[-1][col].extend(chunk[col])
        else:
            # chunks overlap
            overlap = merged_chunks[-1]["chunk.end"] - chunk["chunk.start"]
            assert overlap > 0
            if "chunk.overlap" in chunk:
                if chunk["chunk.overlap"] != overlap:
                    raise AssertionError(
                        f'calcluated an overlap of {overlap} but expected an overlap of {chunk["chunk.overlap"]}'
                    )
            merged_chunks[-1] = merged_chunks[-1].copy()
            merged_chunks[-1]["chunk.end"] = chunk["chunk.end"]
            merged_chunks[-1]["chunk.text"] += chunk["chunk.text"][overlap:]

            merged_chunks[-1]["chunk.id"] += " " + chunk["chunk.id"]
            assert (
                len(merged_chunks[-1]["chunk.text"])
                == merged_chunks[-1]["chunk.end"] - merged_chunks[-1]["chunk.start"]
            )

            for col in chunk_metadata_cols:
                merged_chunks[-1][col].extend(chunk[col])
    return merged_chunks


def convert_chunks_to_evidence(
    chunks, chunk_metadata_cols=["predicted_similarity.index", "predicted_similarity.score"]
) -> list[Text]:

    merged_chunks = merge_overlapping_chunks(chunks, chunk_metadata_cols)

    evidence = []
    for chunk in merged_chunks:
        evidence.append(
            Text(
                chunk["chunk.text"],
                meta={
                    "start": chunk["chunk.start"],
                    "end": chunk["chunk.end"],
                    **{k: v for k, v in chunk.items() if k not in ["chunk.start", "chunk.end" "chunk.text"]},
                },
            )
        )
    return evidence


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument(
        "--ret_fail_ok",
        help="do not raise an error if not every claim has evidence retrieved, mostly only applicalbe to the oracle retrieval method",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    df = read_jsonl(args.input)

    start_claim_doc_pairs = df[["claim.flat", "document.pmid"]].drop_duplicates().shape[0]

    # add missing metadata columns (automatically exist in later versions)
    if "chunk.size" not in df.columns:
        chunk_size = (df["chunk.end"] - df["chunk.start"]).mode()[0]
        df["chunk.size"] = chunk_size
    else:
        chunk_size = df["chunk.size"].mode()[0]

    if "chunk.overlap" not in df.columns:
        chunks_df = df[df["chunk.start"] != 0].copy()
        chunks_df = chunks_df.sort_values(["chunk.start"]).drop_duplicates("document.pmid")
        overlap = (chunks_df["chunk.size"] - chunks_df["chunk.start"]).mode()[0]
        df["chunk.overlap"] = overlap

    k_value = args.length // chunk_size
    logging.info(f"using k value of {k_value} calculated based on the chunk size of {chunk_size}")
    assert k_value > 0

    # calculate the accuracy at specific k value
    similarity_function = "n/a"

    if "predicted_similarity.function" in df.columns:
        similarity_function = df["predicted_similarity.function"].unique().tolist()[0]
        assert similarity_function in ["euclidean", "cosine", "n/a"]

    df["predicted_similarity.index"] = df.groupby(["id", "claim.flat", "document.pmid"], dropna=False)[
        "predicted_similarity.score"
    ].rank(method="max", ascending=bool(similarity_function == "euclidean"))
    df["predicted_similarity.index"] = df["predicted_similarity.index"] - 1

    df = df.rename(columns={c: f"expected_{c}" for c in df.columns if c.startswith("evidence")})

    # deal with overlapping chunks
    chunk_columns = [c for c in df.columns if c.startswith("chunk") or c.startswith("predicted_similarity")]
    chunks_df = df.copy()
    chunks_df = chunks_df[chunks_df["predicted_similarity.index"] < k_value]
    chunks_df = (
        chunks_df.groupby(["id", "display_id", "document.pmid"], dropna=False)
        .apply(lambda subset_df: subset_df[chunk_columns].to_dict("records"))
        .reset_index(name="chunks")
    )
    df = df.drop(columns=chunk_columns)
    df = df.drop_duplicates(["id", "display_id", "document.pmid"])
    df = df.merge(chunks_df)
    # the id must be changed b/c it no longer applies to the same claim/evidence pair
    df["id"] = df["id"].apply(lambda _: str(uuid.uuid4()))

    # keep a subset of randomly selected examples?
    df["evidence"] = df.chunks.apply(convert_chunks_to_evidence)
    df["evidence.flat"] = df.evidence.apply(concat_text)

    end_claim_doc_pairs = df[["claim.flat", "document.pmid"]].drop_duplicates().shape[0]

    if not args.ret_fail_ok:
        assert start_claim_doc_pairs == end_claim_doc_pairs, f"{start_claim_doc_pairs} --> {end_claim_doc_pairs}"
    else:
        logging.warning(
            f"due to missing evidence the total claims has been reduced from {start_claim_doc_pairs} to {end_claim_doc_pairs}"
        )

    dump_jsonl(df, args.output)


if __name__ == "__main__":
    main()
