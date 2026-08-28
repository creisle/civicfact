import argparse
import glob
import json
import logging
import os

import pandas as pd
from doccano_client import DoccanoClient

from .util import (
    create_labels,
    create_member,
    create_project,
    set_example_comment,
    set_example_label,
)


def read_exports(jsonl_files):
    dfs = []
    for jsonl_file in jsonl_files:
        logging.info(f'reading: {jsonl_file}')
        username = os.path.basename(jsonl_file).replace('.jsonl', '')
        df = pd.read_json(jsonl_file, lines=True, dtype=False)
        df['username'] = username
        dfs.append(df)
    df = pd.concat(dfs)
    return df


# def remove_label(client: DoccanoClient, project, example, label):
#     url = f"https://doccano.bcgsc.ca/v1/projects/{project.id}/examples/{example.id}/categories/{label.id}"


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('project_folder')
    parser.add_argument('--project_name')
    parser.add_argument(
        '--user_credentials',
        help='path to the json file with username/password pairs',
        default='.doccano_users.json',
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

    df = read_exports(glob.glob(os.path.join(args.project_folder, '*.jsonl')))

    logging.info('getting project labels')
    project_labels = create_labels(
        client, target_project, df[['label']].explode('label').label.dropna().unique().tolist()
    )

    users = {}
    for username in df.username.dropna().unique().tolist():
        user = create_member(client, target_project, username, user_credentials[username])
        users[user.username] = user

    members = list(client.list_members(target_project.id))

    # add examples and labels
    examples = []
    for _, row in df[['id', 'text', 'meta']].drop_duplicates(['id', 'text']).iterrows():
        logging.debug(f'creating example: id={row.id}')
        example = client.create_example(target_project.id, row.text, meta=row.meta)
        examples.append({'id': row.id, 'text': row.text, 'example': example})

    df = df.merge(pd.DataFrame.from_records(examples), on=['id', 'text'])

    for username in df.username.dropna().unique():
        if username == args.superuser:
            user_client = client
        else:
            logging.info(f'logging in as user: {username}')
            user_client = DoccanoClient(args.api_url)
            user_client.login(username, user_credentials[username])
        for _, row in df[df.username == username].iterrows():
            for label in row.label:
                set_example_label(user_client, target_project, row.example, project_labels[label])
            for comment in row.Comments:
                set_example_comment(user_client, target_project, row.example, comment)


if __name__ == '__main__':
    main()
