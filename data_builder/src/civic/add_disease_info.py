import argparse
import logging
import re

import pandas as pd

from article_utils.util import dump_jsonl, listify


def augment_disease(row):
    if pd.isnull(row['disease']):
        return row['disease']
    disease = row['disease']
    if 'name' not in disease:
        if 'displayName' in disease:
            disease['name'] = disease['displayName']
        else:
            raise ValueError('no name for diseae object: {row["disease"]}')
    synonyms = set(disease['diseaseAliases']) | {disease['name']}
    for name in listify(row['do.synonyms']) + listify(row['do.mesh']):
        synonyms.add(name)
    if not pd.isnull(row['do.name']):
        synonyms.add(row['do.name'])

    for syn in list(synonyms):
        m = re.match(r'(.*) (disease|syndrome)$', syn, re.IGNORECASE)
        if m:
            synonyms.add(m.group(1))
    disease['diseaseAliases'] = sorted(synonyms)
    return disease


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('civic')
parser.add_argument('diseases')
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.civic}')
civic_df = pd.read_json(args.civic, lines=True, dtype=False)
logging.info(f'reading: {args.diseases}')
diseases_df = pd.read_json(args.diseases, lines=True, dtype=False).fillna('')
diseases_df = diseases_df.rename(columns={c: f'do.{c}' for c in diseases_df.columns})
print(diseases_df)
diseases_df['_doid'] = diseases_df['do.id'].str.replace('DOID:', '')
civic_df['_doid'] = civic_df.disease.apply(
    lambda disease: disease['doid'] if not pd.isnull(disease) else None
)

civic_df = civic_df.merge(diseases_df, on=['_doid'], how='left', validate='many_to_one')


civic_df['disease'] = civic_df.apply(augment_disease, axis=1)
civic_df = civic_df.drop(
    columns=[c for c in civic_df.columns if c.startswith('_') or c.startswith('do.')]
)

dump_jsonl(civic_df, args.output)
