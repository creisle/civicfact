import argparse
import logging
import re

import pandas as pd

from article_utils.util import dump_jsonl

from .util import filter_casei_dups

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('civic')
parser.add_argument('drugs')
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.civic}')
civic_df = pd.read_json(args.civic, lines=True, dtype=False)
logging.info(f'reading: {args.drugs}')
drugs_df = pd.read_json(args.drugs, lines=True, dtype=False).fillna('')

drugs_lookup = {}

for _, row in drugs_df.iterrows():
    drugs_lookup[row['id']] = {
        'name': row['name'],
        'therapyAliases': row.synonyms,
        'ncitId': row['id'],
    }


def augment_drugs(therapies):
    result = []
    for therapy in therapies:
        code = therapy['ncitId']
        synonyms = filter_casei_dups(
            sorted(
                set(
                    [therapy['name']]
                    + therapy['therapyAliases']
                    + drugs_lookup.get(code, {}).get('therapyAliases', [])
                )
            )
        )
        therapy['therapyAliases'] = synonyms
        result.append(therapy)
    return result


civic_df['therapies'] = civic_df['therapies'].apply(augment_drugs)

# filter any predictive entries without drugs
civic_df['_flagged'] = (civic_df['evidenceType'] == 'PREDICTIVE') & (
    (civic_df['therapies'].str.len() == 0) | (civic_df['therapies'].isna())
)
logging.warning(
    f'dropping {civic_df._flagged.sum()} predictive entries missing drug information: {civic_df[civic_df._flagged]["id"].tolist()}'
)
civic_df = civic_df[civic_df._flagged == False].drop(columns=['_flagged'])

dump_jsonl(civic_df, args.output)
