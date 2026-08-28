import argparse
import json
import logging

import pandas as pd
from article_utils.util import dump_jsonl
from doccano_client import DoccanoClient

from doccano.util import (
    get_example,
    get_project_by_name,
    list_annotations,
    list_comments,
    list_examples,
    list_members,
    list_users,
)


def extract_assignments(examples, assignments=None):
    if assignments is None:
        assignments = {}
    for example in examples:
        for assignment in example['assignments']:
            assignments.setdefault(example['meta']['review_id'], set()).add(assignment['assignee'])
    return assignments


def reformat_assignments(assignments):
    # match field IDs used everywhere else
    result = []
    for assignment in assignments:
        result.append({'user': assignment['assignee_id'], 'username': assignment['assignee']})
    return result


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('project_name')
parser.add_argument('--api_url', default='https://doccano.bcgsc.ca')
parser.add_argument(
    '--passwords',
    required=True,
    help='json file with user password pairs b/c doccano won\'t let you dump annotations properly otherwise but also excludes loads of details from the export',
)
parser.add_argument('output', help='path to the output jsonl file')
args = parser.parse_args()

with open(args.passwords, 'r') as fh:
    passwords = json.load(fh)

# instantiate a client and log in to a Doccano instance
client = DoccanoClient(args.api_url)
client.login(username='admin', password=passwords['admin'])


proj = get_project_by_name(client, args.project_name)

users = list_users(client)
members = list_members(client, proj['id'])
users_by_id = {}

for user in users:
    users_by_id[user['id']] = user

examples = list_examples(client, proj['id'])

comments = list_comments(client, proj['id'])
comments_by_example = {}
for comment in comments:
    comments_by_example.setdefault(comment['example'], []).append(comment)

assignments_by_username = {'admin': set()}
examples_by_id = {}

colab_mode = proj['collaborative_annotation']

for ex in examples:
    annotations = list_annotations(client, proj['project_type'], proj['id'], ex['id'])
    ex['annotations'] = annotations
    ex['comments'] = [
        c for c in comments_by_example.get(ex['id'], []) if (c['username'] == 'admin' or colab_mode)
    ]
    ex['assignments'] = reformat_assignments(ex['assignments'])
    for assn in ex['assignments']:
        assignments_by_username.setdefault(assn['username'], set()).add(ex['id'])
    ex['project'] = proj
    ex['username'] = 'admin'
    examples_by_id[ex['id']] = ex
    if not colab_mode:
        ex['is_confirmed_by'] = {'admin'} if ex['is_confirmed'] else set()


for ex_id, comments in comments_by_example.items():
    for comment in comments:
        if comment['username'] not in assignments_by_username:
            print('comment user not found', comment['username'])

new_annotations = []
# now get the annotaitons as each user
for username, assignments in assignments_by_username.items():
    if not assignments:
        continue
    client = DoccanoClient(args.api_url)
    client.login(username=username, password=passwords[username])
    for ex_id in assignments:
        annotations = list_annotations(client, proj['project_type'], proj['id'], ex_id)
        if colab_mode:
            examples_by_id[ex_id]['annotations'] += annotations
        else:
            example = get_example(client, proj['id'], ex_id)
            example['project'] = proj
            example['username'] = username
            example['annotations'] = annotations
            example['comments'] = [
                c for c in comments_by_example.get(ex_id, []) if c['username'] == username
            ]
            new_annotations.append(example)


for ex in examples + new_annotations:
    for ann in ex['annotations']:
        ann['username'] = users_by_id[ann['user']]['username']


df = pd.DataFrame.from_records(examples + new_annotations)

df['_comment_count'] = df.comments.str.len()
counts_df = df.groupby('id').agg({'_comment_count': 'sum'}).reset_index()
counts_df['expected_count'] = counts_df['id'].apply(lambda id: len(comments_by_example.get(id, [])))
counts_df['match'] = counts_df.expected_count == counts_df._comment_count
assert counts_df.match.all()


dump_jsonl(df[[c for c in df.columns if not c.startswith('_')]], args.output)
