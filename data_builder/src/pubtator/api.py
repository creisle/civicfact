import argparse
import logging
import os
import re
import time
from urllib.parse import urlencode

import backoff
import bioc
import pandas as pd
import requests
import tqdm
from bioc import pubtator

from article_utils.db import (
    connect,
    get_pubtator_annotations,
    upsert_pubtator_annotation,
)
from article_utils.util import dump_jsonl, read_jsonl


class TooManyRequestsError(requests.exceptions.RequestException):
    pass


SEC_PER_REQ = 5  # ratelimit not working proerly. trying old school hack method


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=2)
def submit_text_request(text, concepts='All'):
    col = bioc.BioCCollection()
    doc = bioc.BioCDocument()
    col.add_document(doc)
    p = bioc.BioCPassage()
    p.text = text
    doc.add_passage(p)

    url = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/request.cgi'
    headers = {'Content-type': 'application/x-www-form-urlencoded'}

    logging.debug(f'POST: {url} len(text)={len(text)}')
    resp = requests.post(
        url, data=urlencode({'text': text, 'bioconcept': concepts}), headers=headers
    )
    resp.raise_for_status()
    session_number = resp.json()['id']
    time.sleep(SEC_PER_REQ)

    return session_number


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=2)
def retrieve_text(session_id):
    url = "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/retrieve.cgi"
    logging.debug(f'GET: {url}?id={session_id}')
    resp = requests.get(url, params={'id': session_id})
    resp.raise_for_status()

    time.sleep(SEC_PER_REQ)
    return resp.text


def regex_parse_response(response):
    # only works when the response was for submitted without a pmid and a single bit of text
    # fix new lines in text problems
    # built-in pubtator parser fails but does not throw error when the text part includes new line characters
    if not response:
        return None
    annotations = []

    for line in response.split('\n'):
        if not re.match(r'^00000\s+.*', line):
            continue
        if '\t' in line:
            row = {
                k: v
                for k, v in zip(
                    ['pmid', 'start', 'end', 'text', 'type', 'id', 'others'], line.split('\t')
                )
            }
            del row['pmid']
            annotations.append(row)
        else:
            m = re.match(
                r'00000\s+(?P<start>\d+)\s+(?P<end>\d+)\s+(?P<text>.*)\s+(?P<type>Gene|Chemical|Species|ProteinMutation|[A-Z]\w+)\s+(?P<id>\S+)$',
                line,
            )
            if not m:
                raise ValueError(f'failed to parse line from response: {repr(line)}')
            annotations.append(
                {k: (v.strip() if v is not None else v) for k, v in m.groupdict().items()}
            )
    return annotations


# def parse_response(response):
#     try:
#         col = pubtator.loads(response)[0]
#         if not col.annotations:
#             return fallback_parse_response(response)
#         return [
#             {
#                 'start': a.start,
#                 'end': a.end,
#                 'text': a.text,
#                 'id': a.id,
#                 'type': a.type,
#                 'others': a.others,
#             }
#             for a in col.annotations
#         ]
#     except ValueError:
#         return fallback_parse_response(response)


def main():
    logging.basicConfig(level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper()))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        help='Path to the CIViC JSONL dump',
    )
    parser.add_argument(
        'output',
        help='path to the JSONL output file containing the annotations pulled from pubtator',
    )
    parser.add_argument('--article_db_file', help='path to the sqlite db file', required=True)

    args = parser.parse_args()

    civic_df = read_jsonl(args.input)
    descriptions = civic_df[civic_df['description'].fillna('') != ''].description.unique().tolist()
    logging.info(
        f'There are {len(descriptions)} descriptions of civic envidence items to fetch annotations for'
    )

    db_conn = connect(args.article_db_file)
    try:
        annotations_df = pd.DataFrame.from_records(get_pubtator_annotations(db_conn))
        submitted_descriptions = (
            annotations_df.text.unique().tolist() if annotations_df.shape[0] > 0 else []
        )
        logging.info(f'{len(submitted_descriptions)} descriptions have previously been submitted')
        missing_resp = annotations_df[annotations_df.response.fillna('') == ''].shape[0]
        logging.info(f"retrieving {missing_resp} missing responses")
        # retrieve the annotations for any annotations that were submitted by there is no response
        if annotations_df.shape[0] > 0:
            tolerance = 3
            for _, row in tqdm.tqdm(
                annotations_df[annotations_df.response.fillna('') == ''].iterrows(),
                total=annotations_df[annotations_df.response.fillna('') == ''].shape[0],
                desc='retrieving responses for previously submitted text annotation requests',
            ):
                try:
                    resp = retrieve_text(row['session_id'])
                    upsert_pubtator_annotation(db_conn, {'text': row['text'], 'response': resp})
                except requests.exceptions.HTTPError as err:
                    tolerance -= 1
                    if err.response.status_code in [400, 404]:
                        logging.warning(
                            f'result is not available yet for session_id {row["session_id"]}'
                        )
                        if tolerance <= 0:
                            logging.info(f'will skip remaining ones for now')
                            break
                    else:
                        raise err

        # submit any descriptions that have not yet been submitted
        submissions = []
        logging.info(
            f'There are {len(set(descriptions) - set(submitted_descriptions))} outstanding descriptions to submit'
        )
        for text in tqdm.tqdm(
            sorted(set(descriptions) - set(submitted_descriptions), key=lambda x: len(x)),
            desc='submitted text annotation requests',
        ):
            try:
                sesh_num = submit_text_request(text)
                submissions.append({'session_id': sesh_num, 'text': text})
                upsert_pubtator_annotation(db_conn, {'session_id': sesh_num, 'text': text})
            except requests.exceptions.HTTPError as err:
                logging.warning(f'failed to submit text: {text}')
                logging.error(str(err))

        for row in tqdm.tqdm(submissions, desc='retrieving newly submitted entries'):
            try:
                resp = retrieve_text(row['session_id'])
                upsert_pubtator_annotation(db_conn, {'text': row['text'], 'response': resp})
            except requests.exceptions.HTTPError as err:
                if err.response.status_code in [400, 404]:
                    logging.warning(
                        f'result is not available yet for session_id {row["session_id"]}'
                    )
                else:
                    raise err

        annotations_df = pd.DataFrame.from_records(get_pubtator_annotations(db_conn))
        annotations_df['annotations'] = annotations_df.response.apply(regex_parse_response)
        annotations_df['pubtator'] = annotations_df.apply(
            lambda row: {'annotations': row.annotations, 'created_at': row.created_at}, axis=1
        )
        civic_df = civic_df.merge(
            annotations_df[['pubtator', 'text']].rename(columns={'text': 'description'}), how='left'
        )

        dump_jsonl(civic_df, args.output)
    finally:
        db_conn.close()


if __name__ == '__main__':
    main()
