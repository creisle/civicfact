import glob
import logging
import os
import re

import pysbd
import tqdm
from article_utils.util import dump_jsonl, read_jsonl

from iaa import classify_problem

tqdm.tqdm.pandas()

RENAME_TAGS = {
    'self-contradiction': 'internal inconsistency',
    'contradicts evidence or claim': 'external inconsistency',
}
SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def rename_tag(tag):
    tag = RENAME_TAGS.get(tag, tag).lower().title()
    if tag.lower() in {
        'incoherent',
        'poor wording or grammar',
        'poor wording',
        'poor grammar',
        'poor wording or grammar or ambiguous',
    }:
        return 'Incoherent, Poor Wording, Or Poor Grammar'.title()
    return tag


def trim_trailing_sentences(generated_text: str) -> str:
    split_text = SEGMENTER.segment(generated_text)
    if not split_text[-1].strip().endswith("."):
        return " ".join(split_text[:-1])
    return generated_text


def split_responses(text):
    index = 0
    chunks = []
    for chunk in re.split(r'(^------.*)\n\n', text, flags=re.MULTILINE):
        if chunk.strip().strip().startswith('------'):
            index += len(chunk)
            continue
        trimmed_chunk = trim_trailing_sentences(chunk)
        chunks.append(
            {
                'start': index,
                'end': index + len(trimmed_chunk),
                'text': trimmed_chunk,
                'generated_text': chunk,
            }
        )
        index += len(chunk)
    return chunks


def add_untagged_spans(row, min_span_length=10):
    untagged_spans = [' ' for _ in row.response['text']]

    for tag in row.annotations:
        relative_start = tag['start_offset'] - row['response']['start']
        relative_end = tag['end_offset'] - row['response']['start']
        for pos in range(relative_start, relative_end):
            untagged_spans[pos] = '#'

    untagged_spans = ''.join(untagged_spans)
    result = []
    current_pos = 0
    for span in re.split(r'(#+)', untagged_spans):
        if all(
            [
                span.strip() == '',  # span is untagged
                len(span) > min_span_length,  # longer than <min_span_length> characters
                row.response['text'][current_pos : current_pos + len(span)].strip()
                != '',  # contains non-whitespace
                current_pos != 0,
                current_pos + len(span) != len(row.response['text']),
            ]
        ):
            result.append(
                {
                    'start_offset': current_pos + row['response']['start'],
                    'end_offset': current_pos + len(span) + row['response']['start'],
                    'labelname': 'Untagged',
                    'username': row['username'],
                }
            )
        current_pos += len(span)

    return row.annotations + result


def compute_overlap(range1, range2):
    overlap = min(range1[1], range2[1]) - max(range1[0], range2[0])
    if overlap < 0:
        return 0
    return overlap


def filter_tags(row):
    tags = []
    response_start = row.response['start']
    response_end = row.response['end']
    for annotation in row.annotations:
        if compute_overlap(
            (annotation['start_offset'], annotation['end_offset']),
            (response_start, response_end),
        ):
            tag_start = max([response_start, annotation['start_offset']])
            tag_end = min([response_end, annotation['end_offset']])
            new_annotation = {
                **annotation,
                'start_offset': tag_start,
                'end_offset': tag_end,
                'labelname': rename_tag(annotation['labelname']),
            }
            tags.append(new_annotation)
    return tags


def merge_overlapping_tags(tags):
    merged_tags = []

    for tag in tags:
        merge_pos = -1
        for i, merged_tag in enumerate(merged_tags):
            if (
                tag['labelname'] == merged_tag['labelname']
                and tag['username'] == merged_tag['username']
                and compute_overlap(
                    (tag['start_offset'], tag['end_offset']),
                    (merged_tag['start_offset'], merged_tag['end_offset']),
                )
            ):
                merge_pos = i
                break
        if merge_pos >= 0:
            start = min(merged_tags[merge_pos]['start_offset'], tag['start_offset'])
            end = max(merged_tags[merge_pos]['end_offset'], tag['end_offset'])
            new_tag = {**tag, 'start_offset': start, 'end_offset': end}
            merged_tags[merge_pos] = new_tag
        else:
            merged_tags.append(tag)
    return merged_tags


def is_committee(row):
    for comment in row['comments']:
        if 'tags: group-review' in comment['text']:
            return True
    annotations = sorted(row['annotations'], key=lambda ann: ann.get('created_at', ''))
    tag_user = None
    for ann in annotations:
        if tag_user is not None:
            if ann['username'] != tag_user:
                tag_user = None
                break
        if ann['username'] != 'admin':
            tag_user = ann['username']
    if tag_user is not None:
        return False
    return True


def pick_user(row):
    if row['project']['collaborative_annotation'] == False:
        return row['username']
    if not row['is_committee'] and len(row['assignments']) == 1:
        return row['assignments'][0]['username']
    return 'committee'


def normalize_incomplete_tagging(annotations):
    """
    the incomplete tag was used inconsistently, some people only put it on resposnes that were mostly
    helpful, others put it on all incomplete responses. Normalize this for comparison by removing the
    tag for any resposnes where there are no helpful tags
    """

    has_helpful = any([a['labelname'] == 'Helpful'] for a in annotations)
    if has_helpful:
        return annotations
    return [a for a in annotations if a['labelname'] != 'Incomplete']


logging.basicConfig(level=logging.INFO)


CURRENT_DIR = os.path.dirname(__file__)
DATA_INPUT_FILE = os.path.join(CURRENT_DIR, 'inputs/examples.jsonl.gz')

df = read_jsonl(
    *glob.glob(os.path.join(CURRENT_DIR, 'review_*/examples.jsonl.gz')), add_filename=True
)

df['review_round'] = df.filename.str.split('/').str[-2]
df = df[~df.review_round.str.contains('_2_')]  # don't need round 2 for explanation tags
logging.info(f'loaded {df.shape[0]} rows')
df = df[(df.is_confirmed == True) & (df.annotations.str.len() > 0)]
logging.info(f'{df.shape[0]} confirmed rows with annotations')

df = df.drop(
    columns=[
        'comment_count',
        'annotation_approver',
        'upload_name',
        'filename',
        'is_confirmed_by',
        'id',
        'is_confirmed',
    ]
)


df['response'] = df.text.progress_apply(split_responses)
df['is_committee'] = df.apply(is_committee, axis=1)
df['username'] = df.apply(pick_user, axis=1)
df = df.explode('response')
df['annotations'] = df.apply(filter_tags, axis=1)
df['annotations'] = df.annotations.apply(merge_overlapping_tags).apply(normalize_incomplete_tagging)
df['annotations'] = df.apply(add_untagged_spans, axis=1)
df['review_id'] = df.meta.str['review_id']
df['generated_text'] = df.response.str['generated_text']

# add the metadata from the data input file
data_df = read_jsonl(DATA_INPUT_FILE)
data_df = data_df.merge(df[['review_id', 'generated_text']].drop_duplicates()).drop(
    columns=['filename']
)
count = df.shape[0]
df = df.merge(data_df)
assert count == df.shape[0]
df['generated_text'] = df.response.str['text']
df['tags'] = df.annotations.apply(lambda ann: {t['labelname'] for t in ann})
df['tag_status'] = df.tags.apply(lambda tags: classify_problem(tags))

dump_jsonl(df, os.path.join(CURRENT_DIR, 'processed', 'explanation_tags.jsonl.gz'))
