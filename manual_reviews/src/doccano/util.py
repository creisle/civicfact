import logging
import os
from typing import Literal, Optional, TypedDict

import pandas as pd
from doccano_client import DoccanoClient
from doccano_client.models.example import Example
from doccano_client.models.label import Label
from doccano_client.models.project import Project
from doccano_client.models.user import User

ROLE_NAME = Literal['annotator', 'annotation_approver', 'project_admin']


class Review(TypedDict):
    reason: str
    label: str
    comment: str
    username: str
    round: str
    original_label: str
    group_review_round: Optional[str]
    group_review_tags: Optional[list[str]]
    group_review_comments: Optional[list[str]]
    group_review_validated: Optional[bool]
    group_review_label: Optional[str]


def set_example_label(
    client: DoccanoClient, project_id: int, example_id: int, user_id: int, label_id: int
):
    url = f'projects/{project_id}/examples/{example_id}/categories'
    payload = {'user': user_id, 'label': label_id, 'id': 0}
    logging.info(f'POST: {url}')
    label = client._base_repository.post(url, json=payload)
    return label


def set_example_text(client: DoccanoClient, project: Project, example: Example, text: str):
    url = f'projects/{project.id}/examples/{example.id}/texts'
    payload = {'user': 0, 'text': text, 'id': 0}
    logging.info(f'POST: {url}')
    label = client._base_repository.post(url, json=payload)
    return label


def set_example_comment(client: DoccanoClient, project_id: int, example_id: int, text: str):
    # https://doccano.bcgsc.ca/v1/projects/17/comments?example=1710
    url = f'projects/{project_id}/comments'
    payload = {'projectId': project_id, 'text': text}
    label = client._base_repository.post(url, json=payload, params={'example': example_id})
    return label


def paginate_get_request(client, url, params=None, limit=1000):
    if params is None:
        params = {}
    offset = 0

    result = []
    while True:
        resp = client._base_repository.get(url, params={**params, 'limit': limit, 'offset': offset})
        resp = resp.json()
        offset += limit
        result.extend(resp['results'])
        if resp['next'] is None:
            break
    return result


def list_annotations(client, project_type: str, project_id: int, example_id: int):
    mapping = {
        'DocumentClassification': 'categories',
        'Seq2seq': 'texts',
        'SequenceLabeling': 'spans',
    }
    item = mapping[project_type]
    annotations = client._base_repository.get(
        f'projects/{project_id}/examples/{example_id}/{item}'
    ).json()

    if project_type == 'SequenceLabeling':

        span_types = {}
        for span_type in client._base_repository.get(f'projects/{project_id}/span-types').json():
            span_types[span_type['id']] = span_type

        for ann in annotations:
            span_type = span_types[ann['label']]
            ann['labelname'] = span_type['text']
            ann['labelcolor'] = span_type['text_color']
    elif project_type == 'DocumentClassification':
        tag_types = {}
        for tag in client._base_repository.get(f'projects/{project_id}/category-types').json():
            tag_types[tag['id']] = tag

        for ann in annotations:
            tag = tag_types[ann['label']]
            ann['labelname'] = tag['text']
            ann['labelcolor'] = tag['text_color']
    else:
        raise NotImplementedError(f'not implmented yet for project type: {project_type}')
    return annotations


def get_labelname2id_mapping(client, project_type: str, project_id: int) -> dict[str, str]:

    if project_type == 'SequenceLabeling':

        span_types = {}
        for span_type in client._base_repository.get(f'projects/{project_id}/span-types').json():
            span_types[span_type['labelname']] = span_type['id']
        return span_type
    elif project_type == 'DocumentClassification':
        tag_types = {}
        for tag in client._base_repository.get(f'projects/{project_id}/category-types').json():
            tag_types[tag['text']] = tag['id']
        return tag_types
    else:
        raise NotImplementedError(f'not implmented yet for project type: {project_type}')


def list_users(client):
    # for some unknnown reason they decided not to structure this endpoint the same as all the others. no pagination options
    return client._base_repository.get('users').json()


def list_members(client, project_id):
    # for some unknnown reason they decided not to structure this endpoint the same as all the others. no pagination options
    return client._base_repository.get(f'projects/{project_id}/members').json()


def list_projects(client: DoccanoClient):
    """
    The actual code for this in the doccano python client is broken so we had to write this ourselves
    """
    return paginate_get_request(client, 'projects')


def list_comments(client: DoccanoClient, project_id):
    return paginate_get_request(client, f'projects/{project_id}/comments')


def list_assignments(client: DoccanoClient, project_id):
    # can get these via the examples
    return NotImplementedError(
        'doccano BUG here? returns empty list always, maybe it needs to be by user? can get assignments via examples'
    )
    return paginate_get_request(client, f'projects/{project_id}/assignments')


def get_project_by_name(client: DoccanoClient, name):
    target_project = None
    for project in list_projects(client):
        if project['name'] == name:
            if target_project is not None:
                raise AssertionError(f'project name is not unique ({name})')
            target_project = project
    if target_project is None:
        raise KeyError(f'failed to find project by name ({name}) in list of projects')
    return target_project


def list_examples(client: DoccanoClient, project_id):
    # will need to paginate here
    url = f'projects/{project_id}/examples'
    logging.info(f'GET {url}')

    return paginate_get_request(client, url)


def get_example(client: DoccanoClient, project_id: int, example_id: int):
    return client._base_repository.get(f'projects/{project_id}/examples/{example_id}').json()


def assign_example_to_user(client, project_id, example_id, user_id):
    return client._base_repository.post(
        f'projects/{project_id}/assignments', json={'example': example_id, 'assignee': user_id}
    ).json()


def create_project(
    client: DoccanoClient,
    project_name: str,
    project_type: str,
    project_description: str,
    force=False,
) -> Project:
    # get or create the project
    target_project = None
    projects = list(client.list_projects())
    for project in projects:
        if project.name == project_name:
            logging.info(f'found project: {project.name}[{project.id}]')
            target_project = project

    if target_project:
        if not force:
            raise ValueError(
                f'project {project_name} exists. Will not upload. use the --force option to upload to an existing project. Will delete the existing project of this name prior to upload'
            )
        logging.warning(f'deleting project: {target_project.name}[{target_project.id}]')
        client.delete_project(target_project.id)

    logging.info(f'Creating project: {project_name}')
    target_project = client.create_project(project_name, project_type, project_description)
    logging.info(f'project created: id={target_project.id}')
    return target_project


def create_member(
    admin_client: DoccanoClient,
    project: Project,
    username: str,
    password: str,
    rolename: ROLE_NAME = 'annotator',
) -> User:
    user = admin_client.search_users(username)
    if user:
        user = user[0]
    else:
        logging.info(f'creating user: {username}')
        user = admin_client.create_user(username, password)

    members = list(admin_client.list_members(project.id))

    if not any(member.user == user.id for member in members):
        logging.info(f'adding user {user.username} to project {project.name}')
        admin_client.add_member(project.id, username, rolename)

    return user


def create_labels(client: DoccanoClient, project: Project, labels: list[str]) -> dict[str, Label]:
    # get existing labels
    current_labels = list(client.list_label_types(project.id, 'category'))

    # check if all the incoming labels exist
    for label in labels:
        if not any([l.text == label for l in current_labels]):
            # add any missing labels
            new_label = client.create_label_type(project.id, 'category', label)
            current_labels.append(new_label)
    # return the labels

    return {l.text: l for l in current_labels}


def add_label_to_example(client: DoccanoClient, project_id, example_id, label_id, user_id):
    """
    https://doccano.bcgsc.ca/v1/projects/104/examples/124496/categories
    {id: 0, label: 2629, user: 0}
    """
    return client._base_repository.post(
        f'projects/{project_id}/examples/{example_id}/categories',
        json={'id': 0, 'label': label_id, 'user': user_id},
    ).json()


def label_mode(values, consensus_min=1):
    hist = {}
    for value in values:
        if not pd.isnull(value):
            hist[value] = hist.get(value, 0) + 1
    if not hist or sum(hist.values()) < consensus_min:
        return None, None

    max_value = max(hist.values())
    result = [label for label in hist if hist[label] == max_value]
    if len(result) > 1:
        return None, None
    return result[0], max_value


def fix_labels(label):
    subs = {
        'Supports': 'SUPPORTS',
        'Refutes': 'REFUTES',
        'Not Enough Information': 'NEI',
        'Does Not Support': 'DOES NOT SUPPORT',
    }
    return subs.get(label, label)


def read_relabelling_data(files, min_votes):
    dfs = []
    all_annotators = []
    for filename in files:
        logging.info(f'reading: {filename}')
        username = os.path.basename(filename).replace('.jsonl', '')

        if username != 'admin':
            all_annotators.append(username)
        df = pd.read_json(filename, lines=True, dtype=False)
        df['label'] = df.label.apply(lambda labels: '; '.join(labels)).apply(fix_labels)
        df['comment'] = df.Comments.apply(lambda comments: '; '.join(comments))
        df = df.drop(columns=['Comments'])
        df['example_id'] = df.meta.str['example_id']
        df['username'] = username
        dfs.append(df)
    df = pd.concat(dfs)
    admin_df = df[df.username == 'admin']
    admin_df = (
        admin_df.rename(columns={'label': 'gold_label_name'})
        .drop(columns=['username', 'comment'])
        .drop_duplicates(['example_id'])
    )

    df = admin_df.merge(
        df[df.username != 'admin'][['username', 'comment', 'label', 'example_id']],
        how='left',
        on=['example_id'],
    )

    # remove any examples missing labels
    df = df[df.label.fillna('') != '']

    # add the number of votes per example
    votes_df = df.groupby(['example_id', 'gold_label_name']).agg({'label': list}).reset_index()

    votes_df['example.votes'] = votes_df.label.str.len()
    votes_df['example.majority_label_name'], votes_df['example.majority_label_votes'] = zip(
        *votes_df.label.apply(lambda labels: label_mode(labels, min_votes))
    )
    votes_df['example.majority_label_name'] = votes_df['example.majority_label_name'].fillna(
        'NO MAJORITY'
    )
    votes_df['example.majority_label_votes'] = votes_df['example.majority_label_votes'].fillna(0)

    def count_gold_votes(row):
        count = 0
        for label in row.label:
            if label == row.gold_label_name:
                count += 1

        return count

    votes_df['example.gold_label_votes'] = votes_df.apply(count_gold_votes, axis=1)

    df = df.merge(
        votes_df[['example_id'] + [c for c in votes_df.columns if c.startswith('example.')]]
    )
    df['correct'] = df.label == df.gold_label_name
    df['example.majority_label_correct'] = df['example.majority_label_name'] == df.gold_label_name
    df['freq'] = 1

    return df


def read_error_review_results(files):
    """Read in the HTML JSONL file from the group error review post-relabelling"""
    results = {}
    main_df = None
    for filename in files:
        logging.info(f'reading: {filename}')
        df = pd.read_json(filename, lines=True, dtype=False)
        username = os.path.basename(filename).replace('.jsonl', '')
        df['display_id'] = df.meta.str['example_id']

        if username == 'admin':
            main_df = df
        for _, row in df[df.Comments.apply(lambda x: len(x)) > 0].iterrows():
            for comment in row.Comments:
                results.setdefault(row.display_id, []).append(
                    {'username': username, 'comment': comment, 'labels': row.label}
                )

    main_df['review.comments'] = main_df.display_id.apply(lambda id: results.get(id, []))

    return main_df[['display_id', 'review.comments', 'label']].rename(
        columns={'label': 'review.labels'}
    )
