import argparse
import logging
import os
import re
import subprocess
import urllib
from io import StringIO

import backoff
import pandas as pd
import requests
import requests_cache
import tqdm
from bioconverters.pmcxml import pmcxml2bioc

BASE_URL = 'https://ftp.ncbi.nlm.nih.gov/pub/pmc/manuscript/xml'
DATA_FOLDER = 'data/pmc/author_manuscripts'
LISTING_FILE = os.path.join(DATA_FOLDER, 'listing.jsonl')


@backoff.on_exception(backoff.expo, urllib.error.HTTPError, max_time=5)
def get_listing_chunk(link, url=BASE_URL):
    full_link = f'{url}/{link}'
    logging.info(f'reading: {full_link}')
    df = pd.read_csv(full_link)
    tar_file = link.replace('.filelist.csv', '.tar.gz')
    tar_link = f'{BASE_URL}/{tar_file}'
    df['tar_link'] = tar_link
    df['date'] = link.split('.')[-3]
    df = df.rename(columns={'PMID': 'pmid'})
    return df


def get_listing(url=BASE_URL):
    logging.info(f'GET: {url}')
    html = requests.get(url).text
    links = []

    for line in html.split():
        if 'filelist.csv' not in line:
            continue

        m = re.search(r'href="([^"]+)"', line)
        if m:
            links.append(m.group(1))

    dfs = []
    for link in links:
        dfs.append(get_listing_chunk(link, url=url))
        # subprocess.check_call(['wget', tar_link, '-O', tar_file], cwd=data_folder)
    df = pd.concat(dfs)
    df = df.sort_values(['date']).drop_duplicates(['AccessionID', 'pmid'], keep='last')
    df['pmid'] = df.pmid.astype(str).fillna('')
    return df


def create_listing(output_file):
    logging.info(f'creating listing: {output_file}')
    from .util import dump_jsonl

    df = get_listing()
    dump_jsonl(df, output_file)
    return df


def pick_article_group(pmc_id):
    num = pmc_id.strip('PMC')
    if len(num) > 6:
        return num[:-6].zfill(3)
    return '000'


def check_abstract_only(xml: str):
    sections = set()
    for doc in pmcxml2bioc(
        StringIO(xml),
        all_xml_path_infon=False,
        mark_citations=False,
        trim_sentences=False,
        normalize_tables=False,
    ):
        for passage in doc.passages:
            sections.add(passage.infons['section'])
    if sections - {'title', 'abstract'}:
        return False
    return True


def get_author_xml(pmc_id, listing_file=LISTING_FILE, data_folder=DATA_FOLDER):
    if not os.path.exists(listing_file):
        create_listing(listing_file)
    df = pd.read_json(listing_file, lines=True, dtype=False)
    df = df[df.AccessionID == pmc_id]
    if df.shape[0] == 0:
        raise KeyError(f'XML does not exist in the author manuscripts for {pmc_id}')

    row = list(df.iterrows())[0][1]
    local_tar_file = os.path.join(data_folder, row.tar_link.split('/')[-1])

    if not os.path.exists(local_tar_file):
        subprocess.check_call(['wget', row.tar_link, '-O', local_tar_file])
        # tar --extract --file={tarball.tar} {file}

    local_xml_file = os.path.join(data_folder, row['Article File'])
    if not os.path.exists(local_xml_file):
        logging.info(f'extracting author manuscripts file: {row["Article File"]}')
        subprocess.check_call(
            ['tar', '--extract', f'--file={os.path.basename(local_tar_file)}', row['Article File']],
            cwd=data_folder,
        )

    with open(local_xml_file, 'r') as fh:
        logging.debug(f'reading: {local_xml_file}')
        xml = fh.read()
        if check_abstract_only(xml):
            raise ValueError(f'PMC Author Fulltext but only got an abstract')
        return xml


def upload_listing(args):
    # avoids the circular import issue
    from civic.util import load_dump

    from .db import batch_get_article_metadata, connect, load_metadata_from_api

    author_df = pd.read_json(args.listing, dtype=False, lines=True)
    civic_df = load_dump(args.civic)
    # only load author metadata for civic PMIDs to start
    author_df = author_df.merge(civic_df)

    # extract any tarfiles from these
    for tar_link, subset_df in author_df.groupby('tar_link'):
        archive = tar_link.split('/')[-1]
        target_files = subset_df['Article File'].unique().tolist()
        cwd = os.path.dirname(args.listing)
        if not os.path.exists(os.path.join(cwd, archive)):
            logging.info(f'archive does not exist, fetching archive')
            subprocess.check_call(
                ['wget', tar_link],
                cwd=cwd,
            )
        logging.info(
            f'extracting {len(target_files)} xml files from {os.path.join(cwd, archive)} archive'
        )
        subprocess.check_call(
            ['tar', '--extract', f'--file={archive}', *target_files],
            cwd=cwd,
        )

    conn = connect(args.db_file)
    cursor = None
    try:
        logging.info(f'load missing metadata into the DB')
        meta_df = load_metadata_from_api(conn, author_df.pmid.unique().tolist(), strict=False)
        logging.info(f'now fetch metadata from the DB')
        meta_df = pd.DataFrame.from_records([dict(r) for r in batch_get_article_metadata(conn)])
        meta_df = meta_df.merge(author_df[['pmid', 'Article File']], how='left')
        meta_df['Article File'] = meta_df['Article File'].fillna('')

        # only update those that change
        meta_df = meta_df[meta_df['Article File'] != meta_df['author_fulltext']]
        meta_df['author_fulltext'] = meta_df['Article File']

        cursor = conn.cursor()

        logging.info(f'updating {meta_df.shape[0]} article metadata entries')

        for _, row in tqdm.tqdm(
            meta_df.iterrows(), total=meta_df.shape[0], desc='updating author_fulltext status'
        ):
            cursor.execute(
                '''UPDATE article_metadata SET author_fulltext = :author_fulltext
                WHERE id = :id''',
                {'id': row.id, 'author_fulltext': row.author_fulltext},
            )
        conn.commit()
    finally:
        if cursor:
            cursor.close()
        conn.close()


if __name__ == '__main__':

    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'info').upper())

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='action', dest='action')
    create_parser = subparsers.add_parser('create')
    create_parser.add_argument('listing', default=LISTING_FILE)

    upload_parser = subparsers.add_parser('upload')
    upload_parser.add_argument('db_file')
    upload_parser.add_argument('listing', default=LISTING_FILE)
    upload_parser.add_argument('civic', help='jsonl dump of civic evidence items')

    args = parser.parse_args()
    if args.action == 'create':
        create_listing(args.listing)
    else:
        requests_cache.install_cache('.requests_cache')
        upload_listing(args)

    args = parser.parse_args()
