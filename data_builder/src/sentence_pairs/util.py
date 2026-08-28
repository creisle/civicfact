from nli.util import Text


def convert_chunks_to_evidence(chunks):
    chunks = sorted(chunks, key=lambda chunk: chunk["chunk.start"])

    for chunk in chunks:
        for col in ["predicted_similarity.index", "predicted_similarity.score", "chunk_correct"]:
            if col in chunk:
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
            merged_chunks[-1]["chunk.id"] += " " + chunk["chunk.id"]

            for col in [
                "predicted_similarity.index",
                "predicted_similarity.score",
                "chunk_correct",
            ]:
                if col in chunk:
                    merged_chunks[-1][col].extend(chunk[col])
        else:
            # chunks overlap
            overlap = merged_chunks[-1]["chunk.end"] - chunk["chunk.start"]
            assert overlap > 0
            if "chunk.overlap" in chunk:
                assert (
                    chunk["chunk.overlap"] == overlap
                ), f'calcluated an overlap of {overlap} but expected an overlap of {chunk["chunk.overlap"]}'
            merged_chunks[-1]["chunk.end"] = chunk["chunk.end"]
            merged_chunks[-1]["chunk.text"] += chunk["chunk.text"][overlap:]
            merged_chunks[-1]["chunk.id"] += " " + chunk["chunk.id"]
            assert (
                len(merged_chunks[-1]["chunk.text"])
                == merged_chunks[-1]["chunk.end"] - merged_chunks[-1]["chunk.start"]
            )

            for col in [
                "predicted_similarity.index",
                "predicted_similarity.score",
                "chunk_correct",
            ]:
                if col in chunk:
                    merged_chunks[-1][col].extend(chunk[col])

    evidence = []
    for chunk in merged_chunks:
        evidence.append(
            Text(
                chunk["chunk.text"],
                "unstructured",
                meta={
                    "start": chunk["chunk.start"],
                    "end": chunk["chunk.end"],
                    "id": chunk["chunk.id"],
                    **{
                        k: v
                        for k, v in chunk.items()
                        if k not in ["chunk.start", "chunk.end", "chunk.id", "chunk.text"]
                    },
                },
            )
        )
    return evidence
