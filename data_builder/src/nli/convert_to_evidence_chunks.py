"""
For all hypothesis retrieved evidence, create a version where the evidence is all overlapping chunks instead of only the hypothesis selcted parts
this is to get the model used to longer contexts
"""

import argparse
import logging

import pandas as pd

from article_utils.util import dump_jsonl, normalize_and_expand_column, read_jsonl
from nli.util import concat_text
from sentence_pairs.generate_document_chunks import calculate_overlap

RANDOM_N = 1


def convert_chunks_to_evidence(
    chunks, chunk_metadata_cols=["predicted_similarity.index", "predicted_similarity.score"]
):
    chunks = sorted(chunks, key=lambda chunk: chunk["chunk.start"])
    # drop any duplicate chunks
    chunks_seen = set()
    temp, chunks = chunks, []
    for chunk in temp:
        if chunk['chunk.id'] in chunks_seen:
            continue
        chunks_seen.add(chunk['chunk.id'])
        chunks.append(chunk)

    for chunk in chunks:
        for col in chunk_metadata_cols:
            chunk[col] = [chunk[col]] if col in chunk and chunk[col] else []

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

    evidence = []
    for chunk in merged_chunks:
        meta = {
            k: v
            for k, v in chunk.items()
            if k not in ["chunk.start", "chunk.end", "chunk.id", "chunk.text"] and v
        }
        for k, v in meta.items():
            if isinstance(v, list):
                print(k, v)
                if len(set(v)) == 1:
                    meta[k] = v[0]
        evidence.append(
            dict(
                content=chunk["chunk.text"],
                type="unstructured",
                meta={
                    "start": chunk["chunk.start"],
                    "end": chunk["chunk.end"],
                    "id": chunk["chunk.id"],
                    **meta,
                },
            )
        )
    return evidence


def generate_new_evidence(row, chunk_mapping):
    chunks = []
    for ev in row['evidence']:
        ev_id = f"{int(row['document.pmid'])}:{int(ev['meta']['start'])}-{int(ev['meta']['end'])}"
        if ev_id not in chunk_mapping:
            logging.warning(f'{ev_id} not found')
            return None
        chunks.extend(chunk_mapping[ev_id])
    return convert_chunks_to_evidence(chunks)


def filter_pair_dups(input_df, df):
    # drop any generated entries that duplicate an existing entry
    df['_id'] = df['claim.flat'] + df['evidence.flat'] + df['document.pmid']
    input_df['_id'] = input_df['claim.flat'] + input_df['evidence.flat'] + input_df['document.pmid']
    input_df['_exists'] = True
    df = df.merge(input_df[['_id', '_exists']].drop_duplicates(), how='left')
    df = df[df._exists.isna()]
    logging.info(f'created {df.shape[0]} new pairs')

    df = pd.concat([input_df, df])
    logging.info(f'{df.shape[0]} combined examples')
    logging.info(f'{dict(df.gold_label_name.value_counts())}')
    assert df[df.claim.isna()].shape[0] == 0
    return df.drop(columns=[c for c in df.columns if c.startswith('_')])


def swap_selections_for_chunks(df, chunks_df):
    df = df.copy()
    df = df[df['evidence.source'] != 'civic']
    print(sorted(df.columns))

    print(df[['evidence.selection_method', 'evidence.source']].value_counts(dropna=False))

    ev_df = df.explode('evidence').reset_index(drop=True)
    ev_df = normalize_and_expand_column(ev_df, 'evidence')
    ev_df = ev_df[~ev_df['evidence.meta.start'].isna()]
    ev_df = ev_df[~ev_df['evidence.meta.end'].isna()]

    chunks_df = chunks_df.merge(
        ev_df[['document.pmid', 'evidence.meta.start', 'evidence.meta.end']].drop_duplicates(),
        on=['document.pmid'],
    )
    chunks_df['overlap'] = chunks_df.apply(
        lambda row: calculate_overlap(
            (row['chunk.start'], row['chunk.end']),
            (row['evidence.meta.start'], row['evidence.meta.end']),
        ),
        axis=1,
    )
    chunks_df = chunks_df[chunks_df.overlap > 0]

    evidence_to_chunk_mapping = {}
    for (pmid, ev_start, ev_end), subset_df in chunks_df.groupby(
        ['document.pmid', 'evidence.meta.start', 'evidence.meta.end']
    ):
        ev_id = f'{int(pmid)}:{int(ev_start)}-{int(ev_end)}'
        for _, row in subset_df.iterrows():
            evidence_to_chunk_mapping.setdefault(ev_id, []).append(
                {
                    c: row[c]
                    for c in [
                        'chunk.id',
                        'chunk.start',
                        'chunk.end',
                        'chunk.text',
                        'chunk.overlap',
                        'chunk.size',
                    ]
                }
            )
    df['evidence'] = df.apply(
        lambda row: generate_new_evidence(row, evidence_to_chunk_mapping), axis=1
    )
    df = df[~df.evidence.isna()]
    df['evidence.flat'] = df.evidence.apply(concat_text)

    df['id'] = df['id'] + '-chunks'
    df['evidence.selection_method'] = df['evidence.selection_method'] + ' chunks-overlap'
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('document_chunks')
    parser.add_argument('output')
    args = parser.parse_args()

    input_df = read_jsonl(args.input)
    original_columns = input_df.columns

    df = input_df.copy()
    df = df[df['evidence.source'] != 'civic']
    all_chunks_df = read_jsonl(args.document_chunks)

    df = swap_selections_for_chunks(df, all_chunks_df)
    df = filter_pair_dups(input_df, df)

    dump_jsonl(df[original_columns], args.output)
