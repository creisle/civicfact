import logging

import ratelimit
import requests

BASE_URL = 'https://api.hypothes.is/api'
LIMIT = 200
ONE_MINUTE = 60 * 5


def fetch_annotations_by_group(group_id, api_key=None):
    """"""
    offset = 0
    results = []
    headers = {}
    if api_key is not None:
        headers = {'Authorization': f'Bearer {api_key}'}
    total = None

    while total is None or offset <= total:
        logging.info(f'GET {BASE_URL}/search group={group_id} offset={offset}')
        resp = requests.get(
            f'{BASE_URL}/search',
            params={'limit': LIMIT, 'offset': offset, 'group': group_id},
            headers=headers,
        )
        resp.raise_for_status()
        content = resp.json()
        if total is None:
            total = content['total']
        results.extend(content['rows'])
        offset += LIMIT
    return results


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=1, period=1)
def delete_annotation(annotation_id, api_key):
    """"""
    headers = {'Authorization': f'Bearer {api_key}'}
    logging.info(f'DELETE {BASE_URL}/annotations/{annotation_id}')
    resp = requests.delete(
        f'{BASE_URL}/annotations/{annotation_id}',
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=1, period=1)
def post_annotation(
    group_id: str,
    api_key: str,
    uri: str,
    title: str,
    target_text: str,
    tags: list[str] = [],
    comment: str = '',
):
    """"""
    headers = {'Authorization': f'Bearer {api_key}'}
    logging.info(f'POST {BASE_URL}/annotations group={group_id} tags={tags}')
    resp = requests.post(
        f'{BASE_URL}/annotations',
        json={
            'group': group_id,
            'document': {'title': [title]},
            'uri': uri,
            'target': [{'selector': [{'type': 'TextQuoteSelector', 'exact': target_text}]}],
            'text': comment,
            'tags': tags,
            'permissions': {'read': [f'group:{group_id}']},
        },
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=1, period=1)
def update_annotation(annotation_id: str, api_key: str, body: dict):
    """"""
    headers = {'Authorization': f'Bearer {api_key}'}
    logging.info(f'PATCH {BASE_URL}/annotations/{annotation_id}')
    resp = requests.patch(
        f'{BASE_URL}/annotations/{annotation_id}',
        json=body,
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=1, period=1)
def get_annotation(annotation_id: str, api_key: str):
    """"""
    headers = {'Authorization': f'Bearer {api_key}'}
    logging.info(f'GET {BASE_URL}/annotations/{annotation_id}')
    resp = requests.get(
        f'{BASE_URL}/annotations/{annotation_id}',
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()
