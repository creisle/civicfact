import gzip
import logging
import re
from io import BytesIO, StringIO
from typing import Literal, TypedDict

import backoff
import pandas as pd
import ratelimit
import requests
import tqdm
import xmltodict
import bioc

from .author_manuscripts import get_author_xml, check_abstract_only

logger = logging.getLogger('ncbi')

ESUMMARY_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
MAX_IDS = 50
MAX_SEARCH_TERMS = 20


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=1, period=2)
def _get_pubmed_metadata(
    id_list: list[str],
    raise_on_error: bool = True,
    max_chunk_size=MAX_IDS,
    database='pubmed',
    retry=3,
) -> list[dict]:
    id_list = list(set(str(i) for i in id_list) - {''})  # remove duplicates
    result = []

    for chunk_start in tqdm.tqdm(
        range(0, len(id_list), max_chunk_size), desc='fetching article metadata'
    ):
        chunk = id_list[chunk_start : chunk_start + max_chunk_size]
        chunk_repr = f'Array[{len(chunk)}]' if len(chunk) > 5 else str(chunk)
        logger.debug(f'GET {ESUMMARY_URL}?db={database}&id={chunk_repr}')
        try:
            resp = requests.get(
                ESUMMARY_URL,
                params={'db': database, 'id': ','.join([str(c) for c in chunk]), 'retmode': 'json'},
            )
            resp.raise_for_status()
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.HTTPError,
        ) as err:
            if (
                isinstance(err, requests.exceptions.ChunkedEncodingError)
                or err.response.status_code == 500
            ):
                if max_chunk_size <= 1:
                    if retry == 0:
                        raise err
                    retry -= 1
                    logging.warning(f'retrying request. failed on error: {err}')
                    result.extend(
                        _get_pubmed_metadata(
                            chunk,
                            raise_on_error,
                            database=database,
                            max_chunk_size=1,
                            retry=retry - 1,
                        )
                    )
                    continue
                logging.warning(
                    f'failed to fetch {database} metadata for IDs. Will try in smaller chunks to narrow down problem: {chunk}'
                )
                result.extend(
                    _get_pubmed_metadata(
                        chunk, raise_on_error, database=database, max_chunk_size=max_chunk_size // 3
                    )
                )
                continue
        resp_json = resp.json()

        if 'result' not in resp_json:
            if 'error' in resp_json:
                raise AssertionError(resp_json['error'])
            raise AssertionError(f'response body has no field "result": {resp_json}')
        content = [v for k, v in resp_json['result'].items() if k != 'uids']
        if len(content) != len(chunk):
            missing_ids = set(chunk) - set(resp_json['result']['uids'])
            msg = f'number of records retrieved ({len(content)}) from {database} does not match the number of records requested ({len(chunk)}). Failed to fetch: {list(missing_ids)}'
            if raise_on_error:
                raise AssertionError(msg)
            else:
                logging.warning(msg)
        result.extend(content)

    filtered = []

    for article in result:
        if 'error' in article:
            logging.error(
                f'error retrieving metadata for article PMID:{article["uid"]}: {article["error"]}'
            )
        else:
            filtered.append(article)
    return filtered


def get_pubmed_metadata(id_list: list[str], **kwargs) -> list[dict]:
    id_list = list(set(str(i) for i in id_list))  # remove duplicates

    pubmed_ids = [i for i in id_list if not i.startswith('PMC')]
    pmc_ids = [i[3:] for i in id_list if i.startswith('PMC')]

    results = _get_pubmed_metadata(pubmed_ids, database='pubmed', **kwargs)
    results.extend(_get_pubmed_metadata(pmc_ids, database='pmc', **kwargs))
    return results


class MESHRecord(TypedDict):
    uid: str
    ds_meshterms: list[str]
    ds_scopenote: str
    ds_subheading: list[str]
    ds_meshui: str


def get_mesh_terms_from_uid(uids: list[str]) -> list[MESHRecord]:
    result = []
    for chunk_start in range(0, len(uids), MAX_IDS):
        chunk = uids[chunk_start : chunk_start + MAX_IDS]
        logger.info(f'GET {ESUMMARY_URL}?db=mesh&id=Array[{len(chunk)}]')
        resp = requests.get(
            ESUMMARY_URL,
            params={
                'db': 'mesh',
                'id': ','.join(chunk),
                'retmode': 'json',
            },
        )
        resp.raise_for_status()
        content = [v for k, v in resp.json()['result'].items() if k != 'uids']
        if len(chunk) != len(content):
            logger.warning(
                f'chunk size {(len(chunk))} does not match response size ({len(content)})'
            )
        result.extend(content)
    return result


def get_mesh_terms(id_list: list[str]) -> list[MESHRecord]:
    """
    Get full MESH records from their MESH ID ex. D055728
    """
    id_list = [str(id).strip().replace('MESH:', '') for id in id_list if id]
    uids = []
    for chunk_start in range(0, len(id_list), MAX_SEARCH_TERMS):
        chunk = id_list[chunk_start : chunk_start + MAX_SEARCH_TERMS]
        logger.info(f'GET {ESEARCH_URL}?db=mesh&term=Array[{len(chunk)}]')
        resp = requests.get(
            ESEARCH_URL, params={'db': 'mesh', 'term': '|'.join(chunk), 'retmode': 'json'}
        )
        resp.raise_for_status()
        content = resp.json()['esearchresult']['idlist']
        if len(chunk) != len(content):
            logger.debug(f'chunk size {(len(chunk))} does not match response size ({len(content)})')
        uids.extend(content)

    return get_mesh_terms_from_uid(uids)


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
def fetch_fulltext(pmc_id: str, db: Literal['pmc', 'pubmed'] = 'pmc') -> str:
    """
    https://www.ncbi.nlm.nih.gov/pmc/tools/get-full-text/
    """
    logger.debug(f'GET {EFETCH_URL}?db={db}&id={pmc_id}')
    resp = requests.get(EFETCH_URL, params={'db': db, 'id': pmc_id, 'retmode': 'xml'})
    resp.raise_for_status()
    text = resp.text

    if resp.status_code == 200:
        # can't find PMCID but still returns 200 code
        err_match = re.search(r'<Reply[^>]*error="([^"]*not available[^"]*)"\/?>', resp.text)
        if err_match:
            raise KeyError(err_match.group(1))  # force HTTP error

    if db == 'pmc':
        # make sure we got a full text and not just the abstract from PMC
        if check_abstract_only(text) == True:
            logging.warning(
                f'Got XML from PMCID ({pmc_id}) but the resulting XML contains only the abstract and not the full text'
            )
            # try to get the author manuscript
            return get_author_xml(pmc_id), 'pmc_author_fulltext'
        resp = requests.get(f'https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}')
        resp.raise_for_status()
        if 'is not Open Access' in resp.text:
            # try to get the author manuscript
            return get_author_xml(pmc_id), 'pmc_author_fulltext'
    return text, db


def fetch_biocxml(pmc_id: str, format='unicode') -> str:
    """
    ex. https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC1790863/unicode
    """
    url = (
        f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmc_id}/{format}'
    )
    logging.info(f'GET {url}')
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text


def get_fulltext_license(pmcid) -> str:
    # https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC5334499&retmode=json
    logging.debug(f'GET https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}')
    resp = requests.get('https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi', params={'id': pmcid})
    resp.raise_for_status()
    data = xmltodict.parse(resp.text)['OA']
    if 'error' in data and 'records' not in data:
        raise ValueError(str(data['error']))
    return data['records']['record']['@license']


def fetch_license_data():
    logging.info('GET https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv')
    resp = requests.get('https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv')
    license_data = resp.text
    logging.info('loading license data')
    license_df = pd.read_csv(
        StringIO(license_data), dtype={'PMID': str}, usecols=['PMID', 'License']
    )
    license_df = license_df[~license_df.PMID.isnull()]
    return license_df[['PMID', 'License']].rename(columns={'PMID': 'pmid', 'License': 'license'})


def fetch_id_mapping():
    logging.info('GET https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz')
    resp = requests.get(
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz',
    )
    data = gzip.decompress(resp.content)
    df = pd.read_csv(
        BytesIO(data),
        dtype={'PMID': str, 'PMCID': str, 'DOI': str},
        usecols=['PMID', 'PMCID', 'DOI'],
    )
    df = df[~df.PMID.isnull()]
    return df[['PMID', 'PMCID', 'DOI']].rename(
        columns={'PMID': 'pmid', 'DOI': 'doi', 'PMCID': 'pmcid'}
    )
