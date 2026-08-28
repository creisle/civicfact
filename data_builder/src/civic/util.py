import json
import logging

import pandas as pd

# pubmed IDs in civic that do not exist and should be replaced with the appropriate ID
from article_utils.util import INVALID_PUBMED_IDS

from .download import civic_request


def filter_casei_dups(input: list[str]) -> list[str]:
    """
    filter a list and remove any duplicates, assume case insensitivity
    """
    seen = set()
    result = []
    for item in input:
        if item.lower().strip() not in seen:
            result.append(item)
            seen.add(item.lower().strip())
    return result


def float_to_intstr(float_value):
    if pd.isnull(float_value):
        return ''
    try:
        return str(int(float_value))
    except ValueError as err:
        logging.warning(err)
        return ''


def normalize_evidence_rows(rows, max_level=2):
    df = pd.json_normalize(rows, max_level=max_level)
    # drop object columns that get added for nulls
    if 'disease' in df.columns:
        df = df.drop(columns=['disease'])
    df['pmid'] = ''
    df.loc[df['source.sourceType'] == 'PUBMED', 'pmid'] = df[df['source.sourceType'] == 'PUBMED'][
        'source.citationId'
    ].apply(float_to_intstr)
    # fix invalid pubmed ID
    df['pmid'] = df['pmid'].apply(lambda pmid: INVALID_PUBMED_IDS.get(pmid, pmid))

    df['significance'] = (
        df.significance.str.replace('_', ' ', regex=False)
        .str.replace('SENSITIVITYRESPONSE', 'SENSITIVITY/RESPONSE')
        .str.lower()
    )
    df.loc[df.significance.isin({'na', 'n/a'}), 'significance'] = 'N/A'

    def get_event_user(events, event_type='SUBMITTED'):
        submitter = ''
        for event in events:
            if event['action'] == event_type:
                submitter = event['originatingUser']['displayName']
        return submitter

    def get_event_time(events, event_type='SUBMITTED'):
        submitter = ''
        for event in events:
            if event['action'] == event_type:
                submitter = event['createdAt']
        return submitter

    for event in ['SUBMITTED', 'ACCEPTED', 'REJECTED']:
        df[f'{event.lower()}By'] = df['events.nodes'].apply(
            lambda events: get_event_user(events, event)
        )
        df[f'{event.lower()}At'] = df['events.nodes'].apply(
            lambda events: get_event_time(events, event)
        )
    return df


def load_dump(
    dump_file,
    max_level=2,
    prefix='',
):
    logging.info(f'reading: {dump_file}')
    df = pd.read_json(dump_file, dtype=False, lines=True)
    # df = df.drop(columns=[c for c in exclude_fields if c in df.columns])

    df = normalize_evidence_rows(df.to_dict('records'), max_level=max_level)
    df = df.rename(columns={'events.nodes': 'events'})
    df['evidenceDirection'] = df['evidenceDirection'].str.upper().str.replace(' ', '_')
    df.loc[df.evidenceDirection == 'NA', 'evidenceDirection'] = 'N/A'

    if prefix:
        df = df.rename(columns={c: f'{prefix}.{c}' for c in df.columns})
    return df


def get_item_by_id(item_id, item_type: str):
    if item_type == 'disease':
        query_body = '''\
query BrowseDisease($id: Int!) {
  disease(id: $id) {
      id
      name
      displayName
      doid
      diseaseAliases
  }
}
'''
    elif item_type == 'therapy':
        query_body = '''\
query BrowseTherapy($id: Int!) {
  therapy(id: $id) {
      id
      name
      ncitId
      therapyAliases
  }
}
'''
    elif item_type == 'phenotype':
        query_body = '''\
query BrowsePhenotype($id: Int!) {
  phenotype(id: $id) {
      id
      name
      hpoId
      description
  }
}
'''
    elif item_type == 'molecularProfile':
        query_body = '''\
query BrowseMolecularProfile($id: Int!) {
  molecularProfile(id: $id) {
    name
    id
    variants {
        name
        variantAliases
        id

        feature {
        id
        name
        fullName
        featureAliases
        featureInstance {
            __typename
            ... on Gene {
                entrezId
            }
            ... on Factor {
                ncitId
            }
            ... on Fusion {
                fivePrimeGene {
                    entrezId
                    id,
                    name,
                    featureAliases
                }
                fivePrimePartnerStatus
                threePrimeGene {
                    entrezId
                    id,
                    name,
                    featureAliases
                }
                threePrimePartnerStatus
            }
        }
        }
    }
  }
}
'''
    else:
        raise NotImplementedError(item_type)
    content = civic_request(
        'https://civicdb.org/api/graphql', json={"query": query_body, "variables": {"id": item_id}}
    )
    return content['data'][item_type]
