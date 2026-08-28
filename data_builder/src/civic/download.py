import argparse
import json
import logging
import re
import time
from http.client import HTTPException
from typing import Optional

import backoff
import pandas as pd
import requests
import requests_cache

from article_utils.util import INVALID_PUBMED_IDS, dump_jsonl
from figutils import package_path

QUERY_BODY = '''\
query evidenceItems($after: String) {
  evidenceItems(after: $after, status: ALL) {
    totalCount
    pageInfo {
      endCursor
      hasNextPage
    }
    nodes {
      id
      status
      significance
      molecularProfile {
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
      variantOrigin
      evidenceType
      evidenceLevel
      evidenceRating
      evidenceDirection
      status
      flagged
      source {
        citationId
        name
        sourceType
        pmcId
      }
      description
      disease {
        id
        doid
        name
        diseaseAliases
      }
      therapies {
        id
        ncitId
        name
        therapyAliases
      }
      therapyInteractionType
      events {
        nodes {
          action
          originatingUser {
            displayName
            role
          }
          createdAt
        }
      }
      phenotypes {
        hpoId
        name
        id
      }
      revisions {
        nodes {
          currentValue
          suggestedValue
          status
          revisionSetId
          id
          updatedAt
          fieldName
          creationActivity {
            createdAt
            user {
              displayName
              role
            }
            note
          }
          linkoutData {
            currentValue {
              ... on ObjectField  {
                objects {
                  id
                  entityType
                  deprecated
                  deleted
                }
              }
              ... on ScalarField {
                value
              }
            }
            diffValue {
              __typename
              ... on ObjectFieldDiff {
                suggestedObjects {
                  displayName
                  id
                  entityType
                }
                removedObjects {
                  displayName
                  id
                  entityType
                }
              }
            }
          }
        }
      }
    }
  }
}
'''


class TooManyRequestsError(requests.exceptions.RequestException):
    pass


@backoff.on_exception(backoff.expo, TooManyRequestsError, max_time=10)
def civic_request(*pos, **kwargs):
    resp = requests.post(*pos, **kwargs)
    if resp.status_code == 429:
        raise TooManyRequestsError(resp.text)
    try:
        resp.raise_for_status()
    except Exception as err:
        logging.error(resp.text)
        raise err
    content = resp.json()
    if hasattr(resp, 'from_cache') and not resp.from_cache:
        time.sleep(0.5)  # TODO: fix with real rate limiting

    return content


def fix_molecular_profile_name(mp_name):
    match = re.match(
        r'^(([A-Z][A-Z0-9-]*[A-Z0-9])\s+)((\2::[A-Z][A-Z0-9-]*[A-Z0-9])|([A-Z][A-Z0-9-]*[A-Z0-9]::\2))\b.*',
        mp_name,
    )
    if match:
        mp_name = mp_name[len(match.group(1)) :]
    return mp_name


def page_civic_nodes(
    queryType,
    node_query_body,
    additional_query_filters='',
    url='https://civicdb.org/api/graphql',
    max_records: Optional[int] = None,
):
    if additional_query_filters:
        filters = ", ".join([''] + [f'{k}: {v}' for k, v in additional_query_filters.items()])
    else:
        filters = ''
    query_body = f'''\
query {queryType}($after: String) {{
  {queryType}(after: $after{filters}) {{
    totalCount
    pageInfo {{
      endCursor
      hasNextPage
    }}
    nodes {{
      {node_query_body}
    }}
  }}
}}
'''
    result = []
    page = 1
    has_next_page = True
    cursor = ""

    while has_next_page and (max_records is None or len(result) < max_records):
        logging.info(f'fetching {url} page={page} cursor={cursor}')
        content = civic_request(url, json={"query": query_body, "variables": {"after": cursor}})
        if content.get('errors', []):
            raise HTTPException(str(content['errors']))
        try:
            has_next_page = content['data'][queryType]['pageInfo']['hasNextPage']
            cursor = content['data'][queryType]['pageInfo']['endCursor']

            result.extend(content['data'][queryType]['nodes'])
            page += 1
        except Exception as err:
            logging.error(content)
            raise err
    return result


def get_evidence_items(
    url='https://civicdb.org/api/graphql',
    max_records: Optional[int] = None,
    query_body=QUERY_BODY,
):
    result = []
    page = 1
    has_next_page = True
    cursor = ""

    while has_next_page and (max_records is None or len(result) < max_records):
        logging.info(f'fetching {url} page={page} cursor={cursor}')
        content = civic_request(url, json={"query": query_body, "variables": {"after": cursor}})
        if content.get('errors', []):
            raise HTTPException(str(content['errors']))
        try:
            has_next_page = content['data']['evidenceItems']['pageInfo']['hasNextPage']
            cursor = content['data']['evidenceItems']['pageInfo']['endCursor']

            result.extend(content['data']['evidenceItems']['nodes'])
            page += 1
        except Exception as err:
            logging.error(content)
            raise err

    for row in result:
        try:
            # reformat_old_variant_style(row)
            # row['variant']['name'] = row['variant']['name'].replace('DEPRECATED ', '')
            row['source']['citationId'] = INVALID_PUBMED_IDS.get(
                str(row['source']['citationId']), row['source']['citationId']
            )
            # fix fusion molecular profile name
            row['molecularProfile']['name'] = fix_molecular_profile_name(
                row['molecularProfile']['name']
            )
            # differentiate non pubmed sources by prefixing citation IDs with their source type when it isn't pubmed
            row['source']['citationId'] = (
                str(row['source']['citationId'])
                if row['source']['sourceType'] == 'PUBMED'
                else f"{row['source']['sourceType']}:{row['source']['citationId']}"
            )

        except KeyError as err:
            logging.error(str(row))
            raise err
    return result


def rename_field(item, old_name, new_name):
    item[new_name] = item[old_name]
    del item[old_name]


def coerce_string(row, *fields):
    for field in fields:
        row[field] = str(row[field])


if __name__ == '__main__':
    # globally cache requests
    requests_cache.install_cache(
        '.requests_cache',
        urls_expire_after={
            '*': 4 * 60 * 60,  # 4 hours
        },
    )
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_path', default=package_path('data/civic-dump.jsonl'))
    args = parser.parse_args()
    content = sorted(get_evidence_items(), key=lambda row: row['id'])
    logging.info(f'writing: {args.output_path}')

    df = pd.DataFrame.from_records(content)
    df['id'] = df['id'].astype(str)
    dump_jsonl(df, args.output_path)

    pass
