import logging

import backoff
import requests

logger = logging.getLogger('elsevier')


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
def fetch_by_doi(doi) -> str:
    url = f'https://api.elsevier.com/content/article/doi/{doi}'
    logger.debug(f'GET {url}?view=FULL')
    resp = requests.get(url, params={'view': 'FULL'})

    if resp.status_code == 401 or resp.status_code == 404:
        # full text is not part of the open access subset
        return ''
    resp.raise_for_status()
    return resp.text
