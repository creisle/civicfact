"""
Make the combined set of human gold label + explanations from review rounds
"""

import glob
import logging
import os
import re

import pandas as pd
import pysbd
from article_utils.util import dump_jsonl, read_jsonl


def parse_comment(comment) -> tuple:
    comment_text = comment['text']
    if not comment_text or pd.isnull(comment_text):
        return {**comment, 'gold_label_name': None, 'reason': comment}
    if comment_text.startswith('REFTUES.'):
        return {
            **comment,
            'gold_label_name': 'REFUTES',
            'reason': comment_text[len('REFTUES.') :].strip(),
        }
    m = re.search(
        r'^(SUPPORT|REFUTE|NEI)S?(/(SUPPORT|REFUTE|NEI)S?)?\.?\s*(?P<reason>.*)', comment_text
    )
    if not m:
        return {**comment, 'gold_label_name': None, 'reason': comment_text.strip()}
    if m.group(2):
        if m.group(1) == 'NEI' or m.group(3) == 'NEI':
            return {**comment, 'gold_label_name': 'NEI', 'reason': m.group('reason').strip()}
        raise ValueError(f'unable to parse: {m.group(0)}')
    if m.group(1) == 'NEI':
        return {**comment, 'gold_label_name': 'NEI', 'reason': m.group('reason').strip()}
    return {**comment, 'gold_label_name': m.group(1) + 'S', 'reason': m.group('reason').strip()}


def consensus_label(labels):
    input_count = len(labels)
    counts = {}
    for label in labels:
        if label is None:
            continue
        counts[label] = counts.get(label, 0) + 1

    labels = sorted(counts, key=lambda x: counts[x], reverse=True)
    if not labels:
        return None, 0, input_count, counts
    if len(labels) == 1 or counts[labels[0]] > counts[labels[1]]:
        return labels[0], counts[labels[0]], input_count, counts
    return None, 0, input_count, counts


seg = pysbd.Segmenter(language="en", clean=False)


def parse_comment_flags(comments):
    all_flags = set()
    if not isinstance(comments, list):
        return None
    for comment in comments:
        comment = comment['text'].strip()
        m = re.match(r'^flags?\s*:\s*([^,:]+)(,\s+[^,:]+)*', comment)
        if m:
            comment = comment.split(':')[-1]
            tags = [t.strip().replace('-', ' ') for t in comment.split(',')]
            all_flags.update(tags)
    return all_flags


def add_original_comments(row, original_comments):
    all_comments = row['comments']
    if (row['review_id'], row['username']) in original_comments:
        old_comments = original_comments[(row['review_id'], row['username'])]
        current_comments = {comment['text'] for comment in row['comments']}
        old_comments = [
            {**comment, 'deleted': True}
            for comment in old_comments
            if comment['text'] not in current_comments
        ]
        if old_comments:
            all_comments = old_comments + all_comments

    for comment in all_comments:
        comment.setdefault('deleted', False)
    return all_comments


logging.basicConfig(level=logging.INFO)

CURRENT_DIR = os.path.dirname(__file__)


DATA_INPUT_FILE = os.path.join(CURRENT_DIR, 'inputs/examples.jsonl.gz')
OUTPUT_FILE = os.path.join(CURRENT_DIR, 'processed', 'gold_labels.jsonl.gz')

df = read_jsonl(os.path.join(CURRENT_DIR, 'review_1_initial/examples.jsonl.gz'), add_filename=True)
df['review_id'] = df.meta.str['review_id']

original_comments = {}

for _, row in df.iterrows():
    if row.get('comments'):
        original_comments[(row['review_id'], row['username'])] = row['comments']

# now read the other non-export review files (more data)
df = read_jsonl(os.path.join(CURRENT_DIR, 'review_2_comments/examples.jsonl.gz'), add_filename=True)
print(df.username.value_counts(dropna=False))
df['comment_count'] = df.comments.str.len()
print(df[['username', 'comment_count']].value_counts())
logging.info(f'there are {df.shape[0]} rows')
df['review_id'] = df.meta.str['review_id']

# include the original comments where changed
logging.info(
    f'prior to adding deleted comments there are {df.comments.str.len().sum()} total comments'
)
df['comments'] = df.apply(lambda row: add_original_comments(row, original_comments), axis=1)
logging.info(
    f'after adding deleted comments there are {df.comments.str.len().sum()} total comments'
)

# add flags from the admin user comments
admin_df = df[df.username == 'admin'].copy()
admin_df['admin_flags'] = admin_df.comments.apply(parse_comment_flags)
admin_df = admin_df[['admin_flags', 'review_id']].drop_duplicates(['review_id'])

df = df[df.username != 'admin'].copy()
df = pd.concat(
    [
        df,
        read_jsonl(
            os.path.join(CURRENT_DIR, 'review_3_explanations/examples.jsonl.gz'), add_filename=True
        ),
    ]
).reset_index(drop=True)
logging.info(f'combined with review 3 there are now {df.shape[0]} rows')
df['review_id'] = df.meta.str['review_id']
logging.info(f'there are {df.review_id.nunique()} unique reviews')
df = df.explode('comments')
df = df[~df.comments.str['text'].isna()]
logging.info(f'There are {df.shape[0]} comments convering {df.review_id.nunique()} reviews')
df['comment'] = df.comments.apply(parse_comment)
df['username'] = df.comment.str['username']
df["_text"] = df.comment.str['text']
df = df.drop(columns=['comments'])
df['is_deleted'] = df.comment.str['deleted']
df['comment_date'] = pd.to_datetime(df.comment.str['created_at'])

print(df[df['comment'].str['gold_label_name'].isna()])


df = df[~df['comment'].str['gold_label_name'].isna()]  # drop any unparseable non-admin comments

logging.info(
    f'After dropping non-gold-label comments there are {df.shape[0]} comments convering {df.review_id.nunique()} reviews'
)

cons_df = df.copy()
cons_df = cons_df[cons_df.is_deleted != True]
logging.info(
    f'There are {cons_df.shape[0]} non-deleted comments convering {cons_df.review_id.nunique()} reviews'
)

cons_df['comment_date'] = pd.to_datetime(cons_df.comment.str['created_at'])
cons_df['username'] = cons_df.comment.str['username']
print(cons_df[['comment_date', 'comment']])

cons_df = cons_df.sort_values('comment_date').drop_duplicates(
    ['review_id', 'username'], keep='last'
)  # keep the last gold label for any given user
cons_df['is_superceded'] = False
df = df.merge(cons_df[['review_id', 'username', 'is_superceded']], how='left')
df['is_superceded'] = df['is_superceded'].fillna(True)

cons_df['cons'] = cons_df.comment.str['gold_label_name']
print(cons_df)
cons_df = (
    cons_df.groupby(['review_id'])
    .agg({'cons': consensus_label, 'comment': list, 'username': 'unique'})
    .reset_index()
)
cons_df['consensus.gold_label_support'] = cons_df['cons'].str[1]
cons_df['consensus.gold_label_name'] = cons_df['cons'].str[0]
cons_df['consensus.total_labels'] = cons_df['cons'].str[2]
cons_df['consensus.counts'] = cons_df['cons'].str[3]
cons_df = cons_df.rename(columns={'username': 'consensus.users'})

df = df.merge(
    cons_df[['review_id'] + [c for c in cons_df.columns if c.startswith('consensus.')]], how='left'
)
df = df.merge(admin_df, how='left')
df['review_round'] = df.filename.str.split('/').str[-2]

logging.info(
    f'There are {cons_df[cons_df["consensus.gold_label_support"] == 0].review_id.nunique()} reviews that did not reach consensus: {cons_df[cons_df["consensus.gold_label_support"] == 0].review_id.unique().tolist()}'
)
unresolved_reviews = cons_df[cons_df["consensus.gold_label_support"] == 0].review_id.unique()

# who could complete these?
all_users = set(df.username.unique()) - {'admin'}
for rev_id in unresolved_reviews:
    reviewed_by = df[df['review_id'] == rev_id].username.unique()
    available = all_users - set(reviewed_by)
    logging.info(f'{rev_id} has available users: {available}')


print(df)

# now merge with the input file for adding metadata
data_df = read_jsonl(DATA_INPUT_FILE)
data_df = data_df[
    [
        'evidence.flat',
        'claim.flat',
        'review_id',
        'gold_label_name',
        'claim.source',
        'prompt',
        'evidence_chunks',
        'partition',
    ]
    + [c for c in data_df.columns if c.startswith('document.') or c.startswith('civic.')]
].drop_duplicates(['review_id'])

df = df.merge(data_df, how='left').drop(columns=['annotation_approver'])

dump_jsonl(df, OUTPUT_FILE)
