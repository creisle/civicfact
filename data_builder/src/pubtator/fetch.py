import argparse
import logging
import re

import pandas as pd
import ratelimit
import requests
import requests_cache
from bioconverters.utils import cleanup_text

from article_utils.util import dump_jsonl
from civic.util import load_dump

from .util import BASE_URL, PubtatorAnnotation

THREE_MONTHS = 24 * 60 * 60 * 90
FETCH_LIMIT = 1000
SKIP_SECTIONS = {'REF', 'ACK_FUND', 'AUTH_CONT', 'COMP_INT'}

# globally cache requests
requests_cache.install_cache(
    '.requests_cache',
    urls_expire_after={
        '*': 4 * 60 * 60,  # 4 hours
        BASE_URL: THREE_MONTHS,
        'https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications': THREE_MONTHS,  # ~3 months
        'https://api.elsevier.com/content/article/doi/*': THREE_MONTHS,
        'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/*': THREE_MONTHS,
    },
)


def simplify_text(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return cleanup_text(text)


def fix_pubtator_offsets(bioc_json):
    try:
        assert bioc_json['passages'][0]['offset'] == 0
    except Exception as err:
        logging.warning(
            f'failed to adjust offset for article with the following top-level structure: {bioc_json.keys()}'
        )
        raise err

    for passage_i in range(1, len(bioc_json['passages'])):
        current_passage = bioc_json['passages'][passage_i]
        prev_passage = bioc_json['passages'][passage_i - 1]
        trailing_space = current_passage['offset'] - (
            prev_passage['offset'] + len(prev_passage['text'])
        )
        if trailing_space > 0:
            prev_passage['text'] = prev_passage['text'] + ' ' * trailing_space
        assert prev_passage['offset'] + len(prev_passage['text']) == current_passage['offset']


def pull_annotations_context(bioc_json, surrounding_text=50) -> list[PubtatorAnnotation]:
    result = []
    full_text = ''.join(p['text'] for p in bioc_json['passages'])

    for passage in bioc_json['passages']:
        if passage['infons'].get('section_type') in SKIP_SECTIONS:
            continue
        for annotation in passage['annotations']:
            offset = annotation['locations'][0]['offset']
            text = annotation['text']
            annotation_text = full_text[offset : offset + len(text)]
            if text != annotation_text:
                raise AssertionError(
                    f'The text at the annotation location ({annotation_text}) does not match the text provided by the Bioc annotation ({text})'
                )
            # find suffix and prefix positions, extend until you hit whitespace to make sure word-matching will work
            prefix_start = max([0, offset - surrounding_text])
            suffix_end = offset + len(text) + surrounding_text

            while suffix_end < len(full_text) and full_text[suffix_end - 1] != ' ':
                suffix_end += 1

            while prefix_start > 0 and full_text[prefix_start] != ' ':
                prefix_start -= 1

            result.append(
                PubtatorAnnotation(
                    **{
                        f'passage.{k}': v
                        for k, v in passage['infons'].items()
                        if k in ['section_type', 'type']
                    },
                    offset=offset,
                    **annotation['infons'],
                    pmid=bioc_json['pmid'],
                    pmcid=bioc_json['pmcid'],
                    text=annotation['text'],
                    prefix=full_text[prefix_start:offset],
                    suffix=full_text[offset + len(text) : suffix_end],
                )
            )
    return result


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=1, period=1)
def article_request(url, param_name, article_list) -> "list[dict]":
    list_repr = (
        f'[{len(article_list)}]'
        if len(article_list) > 5
        else ','.join([str(i) for i in article_list])
    )
    logging.info(f'POST {url}?{param_name}={list_repr}')
    resp = requests.post(url, json={param_name: article_list})

    if resp.status_code == 400 and len(article_list) > 5:
        content = []
        mid_point = len(article_list) // 2
        logging.info(f'debugging bad request. Splitting fetch list by 2')
        content.extend(article_request(url, param_name, article_list[:mid_point]))
        content.extend(article_request(url, param_name, article_list[mid_point:]))
    else:
        if resp.status_code > 400:
            logging.error(f'failed to retreive the following articles: {article_list}')
            logging.error(resp.text)
        resp.raise_for_status()
        content = resp.json()['PubTator3']
    return content


def fetch_pubtator_annotations(id_list, id_type) -> "list[PubtatorAnnotation]":
    if id_type == 'pmc':
        url = f'{BASE_URL}/publications/pmc_export/biocjson'
        param_name = 'pmcids'

    else:
        url = f'{BASE_URL}/publications/export/biocjson'
        param_name = 'pmids'

    queue = [
        i for i in id_list if i != ''
    ]  # empty string will throw errors, make sure none in list

    result = []
    while queue:
        articles = article_request(url, param_name, queue[:FETCH_LIMIT])
        for pubtator in articles:
            fix_pubtator_offsets(pubtator)
            result.extend(pull_annotations_context(pubtator))

        if len(queue) <= FETCH_LIMIT:
            break
        else:
            queue = queue[FETCH_LIMIT:]
    return result


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        help='Path to the CIViC JSONL dump',
    )
    parser.add_argument(
        'output',
        help='path to the JSONL output file containing the annotations pulled from pubtator and matched to our local BIOC files',
    )

    args = parser.parse_args()

    civic_df = load_dump(args.input)
    civic_df['source.pmcId'] = civic_df['source.pmcId'].fillna('')
    civic_df['pmid'] = civic_df.pmid.fillna('')
    civic_df = civic_df[civic_df.pmid != '']
    articles_df = civic_df[['pmid', 'source.pmcId']].drop_duplicates()
    logging.info(f'There are {articles_df.shape[0]} articles to fetch annotations for')

    # try by PMC ID first
    annotations = fetch_pubtator_annotations(
        articles_df[articles_df['source.pmcId'] != '']['source.pmcId'].tolist(), 'pmc'
    )
    pmcids_fetched = {annotation['pmcid'] for annotation in annotations if annotation['pmcid']}
    # fetch by pubmed ID if they do not have a PMC ID or if the PMC ID did not work
    annotations.extend(
        fetch_pubtator_annotations(
            articles_df[articles_df['source.pmcId'] == '']['pmid'].tolist(), 'pmid'
        )
    )
    annotations.extend(
        fetch_pubtator_annotations(
            articles_df[
                (articles_df['source.pmcId'] != '')
                & (~articles_df['source.pmcId'].isin(pmcids_fetched))
            ]['pmid'].tolist(),
            'pmid',
        )
    )
    logging.info(f'collected {len(annotations)} annotations')
    ann_df = pd.DataFrame.from_records(annotations)
    ann_df['pmid'] = ann_df['pmid'].astype(str)
    ann_df['source'] = ''
    ann_df['identifier'] = ann_df.identifier.fillna('')
    ann_df = ann_df.rename(
        columns={c: f'pubtator.{c}' for c in ann_df.columns if c not in ['pmid', 'pmcid']}
    )
    ann_df['text'] = (
        ann_df['pubtator.prefix'] + ann_df['pubtator.text'] + ann_df['pubtator.suffix']
    ).apply(simplify_text)
    ann_df['is_species'] = ann_df['pubtator.biotype'] == 'species'
    ann_df['in_table'] = ann_df['pubtator.passage.section_type'] == 'TABLE'
    logging.warning(f'dropping {ann_df.is_species.sum()} species annotations')
    ann_df = ann_df[~ann_df.is_species].drop(columns=['is_species'])
    dump_jsonl(ann_df, args.output)


if __name__ == '__main__':
    main()
