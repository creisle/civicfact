"""
Fix the evidence chunks to match the post-merging text that was merged with an error where not the
entire chunk of text was appended and therefore incomplete chunks were added. The coordinates in the
evidence object need to reflect that.

Additionally, compile user reviews into the reviews object on a single example
"""

import glob
import logging

from article_utils.util import dump_jsonl, read_jsonl
from nli.util import concat_text
from sentence_pairs.util import convert_chunks_to_evidence


def drop_columns_if_exists(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    return df.drop(columns=cols)


def compute_overlap(range1, range2):
    overlap = min(range1[1], range2[1]) - max(range1[0], range2[0])
    if overlap < 0:
        return 0
    return overlap


def expected_evidence_coverage(row):
    coverage = 0
    for ev_group in row['all_expected_evidence']:
        for expected_chunk in ev_group:
            for obs_chunk in row['evidence']:
                overlap = compute_overlap(
                    (expected_chunk['meta']['start'], expected_chunk['meta']['end']),
                    (obs_chunk['meta']['start'], obs_chunk['meta']['end']),
                )
                coverage += overlap

    return coverage


def expected_evidence_length(row):
    total_expected = 0
    for ev in row['all_expected_evidence']:
        for expected_chunk in ev:
            total_expected += expected_chunk['meta']['end'] - expected_chunk['meta']['start']

    return total_expected


# impact of the chunk merging bug?
def fix_evidence_chunk(ev_chunk, chunks_df):
    ev_ids = set(ev_chunk['meta']['id'].split(' '))
    result = []
    text = ev_chunk['content']

    for _, chunk in (
        chunks_df[chunks_df['chunk.id'].isin(ev_ids)].sort_values(['chunk.start']).iterrows()
    ):
        if not text:
            break

        if text[: len(chunk['chunk.text'])] == chunk['chunk.text']:
            result.append(
                {
                    k: v
                    for k, v in chunk.items()
                    if k in ['chunk.start', 'chunk.end', 'chunk.text', 'chunk.id']
                }
            )
            text = text[len(chunk['chunk.text']) :]
        elif chunk['chunk.text'][900:].startswith(text[: len(chunk['chunk.text'][900:])]):
            pos = 900
            new_chunk = {
                k: v
                for k, v in chunk.items()
                if k in ['chunk.start', 'chunk.end', 'chunk.text', 'chunk.id']
            }
            chunk_text = text[: len(chunk['chunk.text'][pos:])]
            new_chunk['chunk.start'] += pos
            new_chunk['chunk.end'] = new_chunk['chunk.start'] + len(chunk_text)
            new_chunk['chunk.text'] = chunk_text
            new_chunk['chunk.derived_from'] = chunk['chunk.id']
            new_chunk['chunk.id'] = f"{chunk['pmid']}:{chunk['chunk.start']}-{chunk['chunk.end']}"
            result.append(new_chunk)
            text = text[len(chunk_text) :]
        elif text.startswith(chunk['chunk.text'][-100:]):
            pos = -100
            new_chunk = {
                k: v
                for k, v in chunk.items()
                if k in ['chunk.start', 'chunk.end', 'chunk.text', 'chunk.id']
            }
            chunk_text = chunk['chunk.text'][pos:]
            new_chunk['chunk.start'] = new_chunk['chunk.end'] - pos
            new_chunk['chunk.text'] = chunk_text
            new_chunk['chunk.derived_from'] = chunk['chunk.id']
            new_chunk['chunk.id'] = f"{chunk['pmid']}:{chunk['chunk.start']}-{chunk['chunk.end']}"
            result.append(new_chunk)
            text = text[len(chunk_text) :]
        else:
            raise NotImplementedError('TODO')

    return result


# first start by fixing the evidence coordinates to be accurate
def fix_ev_coord(row, chunks_df):
    new_chunks = []
    for ev in row.evidence:
        chunks = fix_evidence_chunk(ev, chunks_df)
        new_chunks.extend(chunks)
    ev = convert_chunks_to_evidence(new_chunks)
    return ev


def create_reviews(row):
    return [
        {
            'label': row['predicted_label_name'],
            'reason': row['reason'],
            'round': '2025.01.31_retrieval_review_civicfactpairs-2024.12.22',
            'username': row['curator'],
            'group_review_validated': False,
            **{k: v for k, v in row.items() if k.startswith('majority.')},
        }
    ]


logging.basicConfig(level=logging.INFO)

other_df = read_jsonl(
    *[f for f in glob.glob('data/*.jsonl') if f != 'data/all.jsonl'], add_filename=True
).rename(columns={'id': 'doccano.id', 'text': 'doccano.text'})

input_file_columns = other_df.columns.tolist()

meta_df = read_jsonl('data/all.jsonl').rename(
    columns={
        'model_name': 'evidence.retrieval.model_dir',
        'retrieval_limit': "evidence.retrieval.length",
    }
)
meta_df["evidence.selection_method"] = "ret_model"
meta_df["evidence.retrieval.task"] = "evidence_retrieval"
meta_df["evidence.retrieval.encoding"] = "cross"
meta_df["evidence.retrieval.length"] = 'L' + meta_df["evidence.retrieval.length"].astype(str)


df = other_df.merge(meta_df)
assert df.shape[0] == other_df.shape[0]

df['label'] = df.label.apply(lambda labels: labels[0] if labels else '')
df['predicted_label_name'] = df.label.str.extract(r'^(\w+)')
df['reason'] = df.label.str.extract(r'^\w+\.\s*(.*)$')
df['predicted_label_name'] = df.predicted_label_name.str.upper()
df['curator'] = df.filename.str.extract('data/(\w+)\.jsonl')

df['civic.evidenceDirection'] = df['civic.evidenceDirection'].fillna('SUPPORTS')
df.loc[df['civic.evidenceDirection'] == '', 'civic.evidenceDirection'] = 'SUPPORTS'
df.loc[df['civic.evidenceDirection'] == 'DOES_NOT_SUPPORT', 'civic.evidenceDirection'] = 'REFUTES'

df['freq'] = 1

agg_df = (
    df.groupby(
        [
            'document.pmid',
            'display_id',
            'civic.evidenceDirection',
            'gold_label_name',
            'correct.strict',
        ]
    )
    .agg({'predicted_label_name': list, 'freq': 'sum'})
    .reset_index()
    .rename(columns={'predicted_label_name': 'labels'})
)
agg_df['majority.predicted_label_name'] = agg_df.labels.apply(
    lambda labels: max(set(labels), key=labels.count)
)
agg_df['majority.predicted_label_support'] = agg_df.apply(
    lambda row: sum([1 for l in row.labels if l == row['majority.predicted_label_name']])
    / len(row.labels),
    axis=1,
)
agg_df['majority.num_labels'] = agg_df.labels.str.len()
agg_df.loc[agg_df['majority.predicted_label_support'] <= 0.5, 'majority.predicted_label_name'] = (
    'NO MAJORITY'
)
agg_df = agg_df.rename(columns={'predicted_label_name': 'majority.predicted_labels'}).drop(
    columns=['freq']
)

df = df.merge(agg_df)


chunks_df = read_jsonl('../../../builds/civicpairs-2024.12.22/document_chunks.jsonl.gz')
chunks_df.head()

df['evidence'] = df.apply(lambda ev: fix_ev_coord(ev, chunks_df), axis=1)
df['evidence.flat'] = df['evidence'].apply(concat_text)

# # How much of the expected evidence charaters were in the output?

build_df = read_jsonl('../../../builds/civicfact-2024.12.22/data.jsonl.gz')
build_df = (
    build_df.groupby(['claim.flat', 'document.pmid'])
    .agg({'evidence': list})
    .reset_index()
    .rename(columns={'evidence': 'all_expected_evidence'})
)
pre_size = df.shape[0]
if 'all_expected_evidence' in df.columns:
    df = df.drop(columns=['all_expected_evidence', 'ret_coverage'])
df = df.merge(build_df, on=['claim.flat', 'document.pmid'])
assert pre_size == df.shape[0]
df['expected_evidence.cover'] = df.apply(expected_evidence_coverage, axis=1).round(1)
df['expected_evidence.length'] = df.apply(expected_evidence_length, axis=1).round(1)
df['expected_evidence.coverage'] = df['expected_evidence.cover'] / df['expected_evidence.length']

df = df.merge(
    df.groupby(['display_id'])
    .agg({'expected_evidence.coverage': 'max'})
    .reset_index()
    .rename(columns={'expected_evidence.coverage': 'ret_coverage'})
)

df['reviews'] = df.apply(create_reviews, axis=1)
df = df.rename(columns={'expected_evidence.source': 'evidence.source'})
df = drop_columns_if_exists(
    df,
    [
        'all_expected_evidence',
        'chunks',
        'claim.hash',
        'curator',
        'evidence.hash',
        'freq',
        'is_expected_chunk',
        'labels',
        'predicted_label_name',
        'reason',
        'strict.correct',
        'text',
        'example',
        'id',
        'Comments',
        'conversation',
        'label',
        'filename',
        'in_table',
    ]
    + [
        c
        for c in df.columns
        if c.startswith('expected_evidence')
        or c.startswith('majority.')
        or c.startswith('doccano.')
        or c.startswith('correct.')
        or c.startswith('hypothesis.')
    ],
)
print(sorted(df.columns))
df['document.content_source'] = df['evidence.source']
df['evidence.generated'] = True

print(df)
grouping_cols = ['display_id', 'claim.flat', 'evidence.flat', 'document.pmid']
agg = {c: 'first' for c in set(df.columns) - set(grouping_cols)}
agg['reviews'] = 'sum'
agg['tracking_id'] = 'unique'
df = df.groupby(grouping_cols).agg(agg).reset_index()
print(df)
dump_jsonl(df, 'reviews.jsonl')
