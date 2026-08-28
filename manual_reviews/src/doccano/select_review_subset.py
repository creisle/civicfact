"""
Creates the subset to be reviewed picking a total number of annotations appropriate for the number of
anotators we have and the amount of overlap we want. Outputs the major fields to a file we can then use
to filter examples from other inputs files (or selectively test for some slower or resournces intensive analysis)
"""

import argparse
import logging
import math

import pandas as pd

from article_utils.util import dump_jsonl, read_jsonl


def parse_review_count(reviews):
    reasons = set()
    for rev in reviews:
        if not rev['reason'] or rev['username'] == 'bot':
            continue
        reasons.add(rev['username'])
    return len(reasons)


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--inputs',
    nargs='+',
    help='files to choose claim/evidence pairs from',
    required=True,
    action='append',
)
parser.add_argument(
    '--preferred_examples',
    help='used when regenerating from a previous selection, prefer example pairs in this file to avoid changing the subset more than you need to',
)
parser.add_argument('--ref_data', help='reference dataset to get review stats from')
parser.add_argument('--max_annotations_per_user', default=50, type=int)
parser.add_argument('--users', default=1, type=int)
parser.add_argument('--annotation_overlap', default=1, type=int)
parser.add_argument(
    '--prop_train',
    default=0.6,
    type=float,
    help='proportion of examples to choose from the training vs dev partition',
)
parser.add_argument(
    '--prop_ft',
    default=0.75,
    type=float,
    help='proportion of examples to choose from full-text document sources only',
)
parser.add_argument(
    '--errors_only',
    default=False,
    action='store_true',
    help='when true, only review examples where the label was predicted incorrectly',
)
parser.add_argument(
    '--max_review_count',
    default=None,
    type=int,
    help='when set, exclude documents that have previously been reviewed by x reviewers on any of their claims',
)
parser.add_argument('--output', required=True)
args = parser.parse_args()

if args.max_review_count is not None and args.ref_data is None:
    parser.error(f'ref_data file is required in order to filter on previously reviewed content')

dfs = []
for group_i, group in enumerate(args.inputs):
    df = read_jsonl(*group, add_filename=True)
    df['_group'] = group_i
    dfs.append(df)
df = pd.concat(dfs)
df = df.sample(frac=1)
df['document.pmid'] = df['document.pmid'].astype(str)

df = df.rename(columns={c: c[:-2] for c in df.columns if c.endswith('_x')})  # in case of bad merge

if args.errors_only:
    df = df[df.gold_label_name != df.predicted_label_name]


df['_key'] = df['claim.flat'] + ' ' + df['evidence.flat'] + ' ' + df['document.pmid']

if args.max_review_count is not None:
    # original dataset to get previously reviewed counts
    rev_df = read_jsonl(args.ref_data)
    rev_df['rev_count'] = rev_df.reviews.apply(parse_review_count)
    rev_df = rev_df.groupby(['document.pmid']).agg({'rev_count': 'max'}).reset_index()
    rev_df['document.pmid'] = rev_df['document.pmid'].astype(str)

    df = df.merge(rev_df)
    df = df[df.rev_count <= args.max_review_count]

TOTAL_ANNOTATIONS = int(
    (
        args.max_annotations_per_user
        * math.comb(args.users, args.annotation_overlap)
        / math.comb(args.users - 1, args.annotation_overlap - 1)
    )
)
logging.info(
    f"given {args.users} annotators with an overlap value of {args.annotation_overlap} and a maximum of {args.max_annotations_per_user} annotations per user results in a total coverage of {TOTAL_ANNOTATIONS} examples"
)

# select a subset to be reviewed by curators
# prefer from different publications vs the same publication
examples_df = df[
    [
        'claim.flat',
        'evidence.flat',
        'prompt',
        'document.pmid',
        'partition',
        'document.content_source',
        'gold_label_name',
        'generated',
        'claim',
        'evidence',
        'rev_count',
        '_group',
    ]
].drop_duplicates(
    [
        'claim.flat',
        'evidence.flat',
        'prompt',
        'document.pmid',
        'partition',
        'document.content_source',
        'gold_label_name',
        'generated',
    ]
)

assert (
    examples_df[
        [
            c
            for c in examples_df.columns
            if c not in {'gold_label_name', 'generated', 'claim', 'evidence'}
        ]
    ]
    .drop_duplicates()
    .shape[0]
    == examples_df.shape[0]
), 'multiple gold labels for the same input'
logging.info(
    f'there are {examples_df.shape[0]} examples to select from among {df["document.pmid"].nunique()} publications'
)

examples_df['_pub_repeat_index'] = examples_df.groupby(['document.pmid']).cumcount()
examples_df['_ft'] = examples_df['document.content_source'] != 'pubmed'
examples_df = examples_df[examples_df.partition.isin(['dev', 'train'])]

if args.preferred_examples:
    pref_df = read_jsonl(args.preferred_examples)
    pref_df['_pref_index'] = 0
    examples_df = examples_df.merge(
        pref_df[['claim.flat', 'document.pmid', 'evidence.flat', '_pref_index']], how='left'
    )
    examples_df['_pref_index'] = examples_df._pref_index.fillna(1)
else:
    examples_df['_pref_index'] = 1

subsets = [
    (False, 'dev', TOTAL_ANNOTATIONS * (1 - args.prop_ft) * (1 - args.prop_train)),
    (True, 'dev', TOTAL_ANNOTATIONS * args.prop_ft * (1 - args.prop_train)),
    (False, 'train', TOTAL_ANNOTATIONS * (1 - args.prop_ft) * args.prop_train),
    (True, 'train', TOTAL_ANNOTATIONS * args.prop_ft * args.prop_train),
]
selections = []
total_selected = 0

for label in ['NEI', 'REFUTES', 'SUPPORTS']:
    for i, (ft_status, partition, expected_examples) in enumerate(subsets):
        curr_df = (
            examples_df[
                (examples_df.partition == partition)
                & (examples_df._ft == ft_status)
                & (examples_df.gold_label_name == label)
            ]
            .copy()
            .sample(frac=1)
            .reset_index(drop=True)
            .copy()
            .sort_values(['_pref_index', '_pub_repeat_index'])
            .reset_index(drop=True)
            .copy()
        )
        if i == len(subsets) - 1 and label == 'SUPPORTS':
            expected_examples = TOTAL_ANNOTATIONS - total_selected
        else:
            expected_examples = int(expected_examples * (1 / 3))

        curr_df = curr_df[curr_df.index < expected_examples].copy()
        selections.append(curr_df)
        total_selected += curr_df.shape[0]

selected_df = pd.concat(selections)

logging.info(f'dataset statistics (n={selected_df.shape[0]})')

for col in ['rev_count', 'gold_label_name', '_ft', 'partition', '_group']:
    logging.info(f'{col}: {dict(selected_df[col].value_counts(normalize=True))}')


assert (
    selected_df[['claim.flat', 'document.pmid']].drop_duplicates().shape[0] == selected_df.shape[0]
)

dump_jsonl(
    selected_df[
        [
            'claim.flat',
            'claim',
            'document.content_source',
            'document.pmid',
            'evidence.flat',
            'evidence',
            'gold_label_name',
            'partition',
            'prompt',
        ]
    ],
    args.output,
)
