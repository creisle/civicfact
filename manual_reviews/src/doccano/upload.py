import argparse
import json
import logging
import os

import pandas as pd
from doccano_client import DoccanoClient

from .util import (
    create_labels,
    create_member,
    create_project,
    set_example_label,
    set_example_text,
)


def main(args):
    logging.basicConfig(level=logging.INFO)
    if not args.project_name:
        args.project_name = os.path.basename(args.project_folder.rstrip('/'))

    logging.info(f'reading: {args.user_credentials}')
    with open(args.user_credentials, 'r') as fh:
        user_credentials = json.load(fh)

    if args.superuser not in user_credentials:
        raise parser.error(f'superuser {args.superuser} must be in the credentials file')

    # instantiate a client and log in to a Doccano instance
    client = DoccanoClient(args.api_url)
    client.login(username=args.superuser, password=user_credentials[args.superuser])

    target_project = create_project(
        client, args.project_name, args.project_type, args.project_description, args.force
    )

    users = {}
    for username in args.annotators:
        user = create_member(client, target_project, username, user_credentials[username])
        users[user.username] = user

    logging.info(f'reading: {args.input}')
    df = pd.read_json(args.input, lines=True, dtype=False)

    if args.project_type == 'DocumentClassification':
        logging.info('getting project labels')
        project_labels = create_labels(
            client, target_project, df[['label']].explode('label').label.dropna().unique().tolist()
        )
    else:
        project_labels = {}

    # add examples and labels (not using upload b/c silent fail bug)
    for _, row in df.iterrows():
        example = client.create_example(target_project.id, row.text, meta=row.meta)
        for label in row.label:
            if args.project_type == 'DocumentClassification':
                set_example_label(client, target_project, example, project_labels[label])
            else:
                set_example_text(client, target_project, example, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--project_name')
    parser.add_argument(
        '--user_credentials',
        help='path to the json file with username/password pairs',
        default='.doccano_users.json',
    )
    parser.add_argument(
        '--annotators', default=[], nargs='+', help='add users to project as annotators'
    )
    parser.add_argument('--api_url', default='https://doccano.bcgsc.ca')
    parser.add_argument(
        '--project_type',
        choices=['DocumentClassification', 'Seq2seq'],
        default='DocumentClassification',
    )
    parser.add_argument(
        '--force', '-f', action='store_true', help='upload to the project even if it already exists'
    )
    parser.add_argument('--project_description', default='TBD')
    parser.add_argument('--superuser', default='admin')
    args = parser.parse_args()
    main(args)
