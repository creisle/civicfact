import argparse
import itertools
import logging
import os
import random

import pandas as pd
from doccano_client import DoccanoClient

from doccano.util import (
    assign_example_to_user,
    create_labels,
    get_project_by_name,
    list_examples,
    list_members,
)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('project_name')
    parser.add_argument('--api_url', default='https://doccano.bcgsc.ca')
    parser.add_argument('--user', default='admin')
    parser.add_argument('--password_env_var', default='DOCCANO_PASSWORD')
    parser.add_argument(
        '--overlap',
        help='number of times each examples should be assigned by default',
        default=2,
        type=int,
    )
    parser.add_argument(
        '-n',
        help='dry run, just show what you would assign, do not actually post requests',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()

    # instantiate a client and log in to a Doccano instance
    client = DoccanoClient(args.api_url)
    client.login(username=args.user, password=os.environ[args.password_env_var])

    target_project = get_project_by_name(client, args.project_name)

    users = list_members(client, target_project['id'])
    assert len(set([u['username'] for u in users])) == len(
        users
    ), 'username is not a unique constraint'
    logging.info(f'found {len(users)} project members')

    users = [u for u in users if u['username'] != args.user]
    logging.info(f'found {len(users)} annotators (excluded admin user)')

    examples = list_examples(client, target_project['id'])
    logging.info(f'found {len(examples)} examples')

    # try to make sure each users gets a couple examples in common with each other user
    # attempt to evenly assign examples to users
    user_ids = [u['user'] for u in users]
    assignment_groups = list(itertools.combinations(user_ids, args.overlap))
    random.shuffle(
        assignment_groups
    )  # need to shuffle so we don't favor any particular combination

    # shuffle the examples too
    random.shuffle(examples)

    metrics = {}

    for i, example in enumerate(examples):
        for user_id in assignment_groups[i % len(assignment_groups)]:
            if not args.n:
                assign_example_to_user(client, target_project['id'], example['id'], user_id)
            metrics[user_id] = metrics.get(user_id, 0) + 1
    print(metrics)


if __name__ == '__main__':
    main()
