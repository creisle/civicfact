import glob
import json
import logging
import os
import time

import pandas as pd
import requests_cache

from article_utils.author_manuscripts import get_listing
from article_utils.ncbi import fetch_license_data
from article_utils.util import NpEncoder
from civic.download import get_evidence_items
from civic.util import normalize_evidence_rows
from hypothesis.download import fetch_annotations_by_group
from hypothesis.util import parse_hypothesis_annotation
from nli.util import SELECTION_DELIMITER

# globally cache requests
requests_cache.install_cache(
    '.requests_cache',
    urls_expire_after={
        'https://api.hypothes.is/api*': 30 * 60,  # 30m
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/manuscript/xml': 0,
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/manuscript/xml/index.html': 0,
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/manuscript/xml/*': 60 * 60 * 20,  # 20h
        'https://civicdb.org/api/graphql': 60 * 60 * 20,  # 20h
        'https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv': 60 * 60 * 20,  # 20h
    },
)


ASCII_MAX = 127


def flatten(t):
    return [item for sublist in t for item in sublist]


def agg_text(series):
    return '; '.join([s for s in series.unique() if not pd.isnull(s)])


def write_data_summary(output_file, hyp_annotations_df, civic_data_df):
    df = hyp_annotations_df.copy().rename(
        columns={'author': 'username', 'status': 'status_label', 'text': 'selected_text'}
    )
    print(df.columns)
    print(df)
    df['civic_eid'] = df.civic_eid.astype(str)
    pmc_check = civic_data_df.copy()[['civic_eid', 'source.pmcId', 'pmid', 'license']].rename(
        columns={'source.pmcId': 'civic_pmc_id', 'pmid': 'civic_pubmed_id'}
    )
    pmc_check['civic_eid'] = pmc_check.civic_eid.astype(str)
    df = df.merge(pmc_check, how='left').fillna('')

    for col in ['pubmed_id', 'civic_pubmed_id', 'pmc_id', 'civic_pmc_id']:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('^PMC', '', regex=True)
            .str.replace(r'\.0$', '', regex=True)
        )

    df['username'] = df.username.str.replace(r'^acct:', '', regex=True)
    df['username'] = df.username.str.replace(r'@hypothes.is$', '', regex=True)

    def group_text(text):
        return SELECTION_DELIMITER.join([f'{t.strip()}' for t in text])

    df.loc[
        (df.pubmed_id != '') & (df.civic_pubmed_id != '') & (df.pubmed_id != df.civic_pubmed_id),
        'errors',
    ] = (
        'PubMed mismatch: ' + df.civic_pubmed_id + ' vs ' + df.pubmed_id
    )
    df.loc[
        (df.pmc_id != '') & (df.civic_pmc_id != '') & (df.pmc_id != df.civic_pmc_id),
        'errors',
    ] = (
        'PMC mismatch: ' + df.civic_pmc_id + ' vs ' + df.pmc_id
    )
    df.loc[df.errors != '', 'fix'] = True

    df = (
        df.groupby(
            [
                'username',
                'civic_eid',
                'example',
                'status_label',
                'pmc_id',
                'pubmed_id',
                'source',
                'license',
            ]
        )
        .agg({'selected_text': group_text, 'fix': 'any', 'errors': agg_text})
        .reset_index()
    )
    df['status_label'] = df.status_label.str.upper()
    df['user_sort'] = df.username.apply(lambda name: 0 if name == 'creisle' else 1)
    df = df.sort_values(['user_sort'])
    df = df.drop(columns=['selected_text'])

    df = df[df.civic_eid != '']
    df['civic_eid'] = df.civic_eid.astype(int)
    logging.info(f'writing: {output_file}')
    with open(output_file, 'w') as fh:
        fh.write(json.dumps({'records': df.fillna('').to_dict(orient='records')}, cls=NpEncoder))


def find_fix_flagged_annotations(raw_annotations) -> dict[str, str]:
    annotations_by_id = {}
    fix_flagged = []
    for annotation in raw_annotations:
        annotations_by_id[annotation['id']] = annotation
        tags = {t.lower().strip() for t in annotation['tags']}
        if "fix" in tags:
            fix_flagged.append((annotation['id'], annotation['text']))

    result = {}
    while fix_flagged:
        current_id, comment = fix_flagged.pop(0)
        if current_id in result or current_id not in annotations_by_id:
            continue
        current = annotations_by_id.get(current_id)
        result[current_id] = comment
        for ref in current.get('references', []):
            fix_flagged.append((ref, comment))

    return result


def get_license_info():
    author_df = get_listing()
    author_df['license'] = 'Author Manuscript Available'
    logging.info(f'There are {author_df.shape[0]} author manuscripts available')
    license_df = fetch_license_data()
    license_df = license_df[(license_df.pmid != '') & (license_df.license != '')]
    logging.info(f'There are {license_df.shape[0]} articles in the OA dataset')
    print(license_df.groupby(['license']).agg({'pmid': 'nunique'}))

    df = pd.concat([license_df[['license', 'pmid']], author_df[['license', 'pmid']]])
    df = df.sort_values(['pmid', 'license']).drop_duplicates(['pmid'], keep='last')
    print(df.groupby(['license']).agg({'pmid': 'nunique'}))
    logging.info(f'There are {df.shape[0]} articles in the combined dataset')
    return df


def main(output_folder):
    annotations = []
    raw_annotations = fetch_annotations_by_group('JRwd19vg', os.environ['HYPOTHESIS_API_KEY'])
    flagged_annotations = find_fix_flagged_annotations(raw_annotations)
    logging.info(f'fetched {len(raw_annotations)} annotation objects')
    for ann in raw_annotations:
        if ann.get('tags') == ['todo'] or ann.get('tags') == [] or ann.get('tags') == ['acronym']:
            continue

        try:
            parsed = parse_hypothesis_annotation(ann)
            if not parsed.tags:
                raise ValueError(f'Unable to parse any tags: {ann.get("tags")}')
            for tag in parsed.tags:
                annotations.append(
                    {
                        'civic_eid': tag.civic_eid,
                        'status': tag.status,
                        'author': parsed.author,
                        'pmc_id': parsed.pmc_id,
                        'pubmed_id': parsed.pubmed_id,
                        'text': parsed.text,
                        'example': tag.example_number,
                        'annotation_id': parsed.id,
                        'errors': flagged_annotations.get(parsed.id, ''),
                        'fix': parsed.id in flagged_annotations,
                        'source': parsed.uri,
                    }
                )
        except (KeyError, ValueError) as err:
            logging.error(f'error in parsing the annotation {ann["id"]} by {ann["user"]}')
            logging.error(str(err))

    annotations_df = pd.DataFrame.from_records(annotations)
    annotations_summary_df = (
        annotations_df.groupby(['civic_eid'])
        .agg(
            {
                'author': 'nunique',
                'fix': 'any',
                'annotation_id': 'nunique',
                'errors': lambda x: ';'.join(x.unique()),
            }
        )
        .reset_index()
        .rename(columns={'author': 'author_count', 'annotation_id': 'annotations'})
    )
    annotations_summary_df = annotations_summary_df[annotations_summary_df.civic_eid != '']
    annotations_summary_df['civic_eid'] = annotations_summary_df['civic_eid'].astype(int)

    df = (
        normalize_evidence_rows(get_evidence_items())
        .rename(columns={'id': 'civic_eid'})
        .merge(annotations_summary_df, how='left', on=['civic_eid'])
    )

    def filter_relevant_revisions(revisions):
        filtered = []
        for rev in revisions:
            if rev['fieldName'] not in {'evidence_level', 'description', 'rating'}:
                filtered.append(rev)
        return filtered

    df['revisions.nodes'] = df['revisions.nodes'].apply(filter_relevant_revisions)

    df['pending_revisions'] = df['revisions.nodes'].apply(
        lambda revs: any([rev['status'] == 'NEW' for rev in revs])
    )

    df['revised_fields'] = df['revisions.nodes'].apply(
        lambda revs: '; '.join(sorted({rev['fieldName'] for rev in revs if rev['status'] == 'NEW'}))
    )

    license_df = get_license_info()
    df = df.merge(license_df[['pmid', 'license']], how='left')
    df['license'] = df.license.fillna('')
    df['source.pmcId'] = df['source.pmcId'].replace('None', '').fillna('')
    non_pmc_full_text = df[(df.license != '') & (df['source.pmcId'] == '')].shape[0]
    logging.warning(f'there are {non_pmc_full_text} full text articles without a PMC ID')
    # df.loc[
    #     df['source.pmcId'] == '', 'license'
    # ] = ''  # do not both with full-text without PMC version

    logging.info(f'found {annotations_df.shape[0]} annotations')
    write_data_summary(os.path.join(output_folder, 'nli.json'), annotations_df, df)

    output_file = os.path.join(output_folder, 'civic.json')
    logging.info(f'writing: {output_file}')
    df['therapies'] = df.therapies.apply(lambda therapies: ';'.join(d['name'] for d in therapies))
    df['civic_eid'] = df.civic_eid.astype(int)
    # df['errors'] = df['errors'].apply(lambda errors: ';'.join(e for e in errors if e))
    with open(output_file, 'w') as fh:
        df['annotations'] = df['annotations'].fillna(0)
        output_records = (
            df.fillna('')
            .rename(
                columns={
                    'source.pmcId': 'pmc_id',
                    'pmid': 'pubmed_id',
                    'evidenceDirection': 'evidence_direction',
                    'evidenceType': 'evidence_type',
                    'significance': 'clinical_significance',
                    'molecularProfile.name': 'molecularProfile',
                }
            )[
                [
                    'civic_eid',
                    'pmc_id',
                    'pubmed_id',
                    'evidence_direction',
                    'evidence_type',
                    'clinical_significance',
                    'molecularProfile',
                    'therapies',
                    'license',
                    'annotations',
                    'fix',
                    'errors',
                    'status',
                    'submittedBy',
                    'acceptedBy',
                    'pending_revisions',
                    'revised_fields',
                ]
            ]
            .to_dict(orient='records')
        )
        for row in output_records:
            try:
                json.dumps(row)
            except TypeError as err:
                print(row)
                raise err
        fh.write(json.dumps({'records': output_records}))


def generate_docs(**kwargs):
    logging.basicConfig(level=logging.INFO)
    output_folder = os.path.join(kwargs['config']['docs_dir'], 'data')
    json_data_files = glob.glob(os.path.join(output_folder, '*.json'))

    ONE_HOUR = 60 * 60
    if (
        json_data_files
        and time.time() - max(os.path.getmtime(f) for f in json_data_files) <= ONE_HOUR
        and not kwargs.get('force')
    ):
        # otherwise mkdocs serve will infinite loop reload
        logging.warning(f'JSON File(s) exist under {output_folder}. Will not re-generate')
    else:
        main(output_folder)


if __name__ == '__main__':
    generate_docs(config={'docs_dir': 'docs'}, force=True)
