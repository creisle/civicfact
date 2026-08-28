import logging
import os
import re

from article_utils.util import dump_jsonl, read_jsonl


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


def create_reviews(row):
    reviews = []
    for doc_label in row['doccano.label']:
        m = re.match(r'\s*(\S+)( \(weakly\))?\.\s*(.*)', doc_label)

        reviews.append(
            {
                'label': m.group(1),
                'reason': m.group(3),
                'round': '2025.03.21_additional_explanations_2024.03.21',
                'username': 'creisle',
                'group_review_validated': False,
                'original_label': row['gold_label_name'],
            }
        )
        if m.group(2):
            reviews[-1]['comment'] = 'weak evidence'
    return reviews


def relative_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def pick_new_gold_label(reviews):
    reviews = [r['label'] for r in reviews]
    return max(set(reviews), key=reviews.count)


logging.basicConfig(level=logging.INFO)

df = read_jsonl(relative_path('data/admin.jsonl')).rename(
    columns={k: f'doccano.{k.lower()}' for k in ['id', 'text', 'label', 'Comments']}
)

meta_df = read_jsonl(relative_path('data/subset-all.jsonl')).drop(columns=['reviews'])
df = meta_df.merge(df[['document.pmid', 'evidence.flat', 'claim.flat', 'doccano.label']])
df['reviews'] = df.apply(create_reviews, axis=1)
df['gold_label_name'] = df.reviews.apply(pick_new_gold_label)

dump_jsonl(df, relative_path('reviews.jsonl'))
