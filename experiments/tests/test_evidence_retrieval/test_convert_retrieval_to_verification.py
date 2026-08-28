import pytest

from evidence_retrieval.convert_retrieval_to_verification import (
    convert_chunks_to_evidence,
)


@pytest.mark.parametrize(
    "ranges,merged_ranges",
    [[[(23400, 24400), (24300, 25300), (25200, 26200)], [(23400, 26200)]]],
)
def test_convert_chunks_to_evidence(ranges, merged_ranges):
    input_chunks = []
    for start, end in ranges:
        input_chunks.append(
            {"chunk.start": start, "chunk.end": end, "chunk.text": "x" * (end - start), "chunk.id": f"{start}-{end}"}
        )
    result = convert_chunks_to_evidence(input_chunks, [])
    assert [(r.meta["start"], r.meta["end"]) for r in result] == merged_ranges
