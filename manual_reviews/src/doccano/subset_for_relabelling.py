import argparse
import itertools
import logging
import os
import random

import pandas as pd

from article_utils.util import dump_jsonl


def pick_examples(df, max_examples):
    """
    First try to balance all aspects of the dataset before picking relabelling data. This is to ensure we capture each type of example that can occur.
    If there is still annotation space after we have representative examples, randomly select the remaining entries

    exclude entries which already have manually annotated review data
    """
    df = df.copy()
    df['_selected'] = False
    df['_review_count'] = df.apply(
        lambda row: sum(1 for rev in row.reviews if rev['username'] != 'bot'), axis=1
    )
    df = df[df._review_count == 0].copy()

    # try to cover each diff aspect
    # prefer examples from individual rather than repeat paper
    df['_pmid_rank'] = df.groupby(['document.pmid']).cumcount()
    df['_no_civic'] = df['civic.description'].fillna('') == ''
    df = df.sample(frac=1).sort_values(['_pmid_rank', '_no_civic']).reset_index(drop=True)

    all_partitions = ['train', 'test', 'dev']
    all_labels = df.gold_label_name.unique().tolist()

    train_ratio = 0.6
    selection_size = max_examples / len(all_labels)

    # try to make sure the partitions are balanced relative to their expected proportion
    for partition in all_partitions:
        # try to make sure the labels are balanced
        for label in all_labels:
            current_sampling = df[
                (df.partition == partition)
                & (df._pmid_rank == 0)
                & (df.gold_label_name == label)
                & (df._selected == False)
            ]
            partition_prop = train_ratio if partition == 'train' else ((1 - train_ratio) / 2)
            selection_count = min(current_sampling.shape[0], int(selection_size * partition_prop))
            logging.info(
                f'selecting {selection_count} samples for the {partition} partition and {label} label'
            )
            # already randomized the order above, take the first x samples
            selected_indices = current_sampling.head(n=selection_count).index
            df.loc[selected_indices, '_selected'] = True

    # TODO: try to ensure the user which created the example is not always the same
    # TODO: try to ensure some tabular examples are included

    remaining_examples = int(max_examples - df._selected.sum())
    assert remaining_examples >= 0, f'generated too many examples ({remaining_examples})'
    df.loc[df[~df._selected].sample(n=remaining_examples).index, '_selected'] = True
    assert df._selected.sum() <= max_examples
    df = df[df._selected].copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith('_')])
    return df


def assign_users_to_examples(
    df, users: int, coverage: int, annotations_per_user: int, flex: int, output_dir: str
):
    df = df.copy().sample(frac=1).reset_index()
    df['_total'] = 0
    user_columns = [f'_subset-{i}' for i in range(users)]
    user_combinations = list(itertools.combinations(user_columns, r=coverage))
    random.shuffle(user_combinations)
    df['_index'] = df.index
    df['_assignment'] = df._index.apply(lambda i: user_combinations[i % len(user_combinations)])

    for user_col in user_columns:
        df[user_col] = df._assignment.apply(lambda assignment: bool(user_col in assignment))

    def check_coverage(total):
        assert total >= coverage

    df['_total'] = df[user_columns].sum(axis=1)
    df['_total'].apply(check_coverage)

    if len(user_columns) > 1:
        for user_col in user_columns:
            user_total = df[user_col].sum()
            assert (
                user_total <= annotations_per_user + flex
                and user_total >= annotations_per_user - flex
            ), f'user {user_col} has been assigned {user_total} annotations'

            output_filename = os.path.join(output_dir, user_col[1:] + '.jsonl')
            dump_jsonl(
                df[df[user_col] == True][[c for c in df.columns if not c.startswith('_')]],
                output_filename,
            )
    dump_jsonl(
        df[[c for c in df.columns if not c.startswith('_')]],
        os.path.join(output_dir, 'subset-all.jsonl'),
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to the JSONL of examples to be subsetted')
    parser.add_argument('output', help='path to the directory to output each subset file')
    parser.add_argument('--users', required=True, type=int)
    parser.add_argument(
        '--flex',
        type=int,
        default=5,
        help='allowable variability in the number of annotations a user can be assigned',
    )
    parser.add_argument(
        '--coverage', default=3, type=int, help='Minimum number of users to label each example'
    )
    parser.add_argument(
        '--annotations_per_user', help='maximum annotations per user', default=100, type=int
    )
    args = parser.parse_args()

    if args.annotations_per_user < args.coverage:
        parser.error(
            f'the number of annotations per user ({args.annotation_per_user}) cannot be less than the expected coverage ({args.coverage}) per example'
        )
    if args.users < args.coverage:
        parser.error(
            f'the number of users ({args.annotations_per_user}) cannot be less than the expected coverage ({args.coverage}) per example'
        )
    max_examples = args.annotations_per_user * args.users / args.coverage
    logging.info(f'reading: {args.input}')
    df = pd.read_json(args.input, lines=True, dtype=False)
    df = pick_examples(df, max_examples)
    assign_users_to_examples(
        df, args.users, args.coverage, args.annotations_per_user, args.flex, args.output
    )


if __name__ == '__main__':
    main()
