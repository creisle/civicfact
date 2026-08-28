import argparse
import logging
import re

import pandas as pd

from article_utils.util import dump_jsonl, listify

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('civic')
parser.add_argument('cells')
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.civic}')
civic_df = pd.read_json(args.civic, lines=True, dtype=False)

logging.info(f'reading: {args.cells}')
cells_df = pd.read_json(args.cells, lines=True, dtype=False).fillna('')

cells_lookup = {}

for disease_name, subset_df in cells_df.explode('derives_from').groupby('derives_from'):
    names = set()
    names.update(subset_df['cellosaurus.id'].tolist())
    names.update(subset_df['clo.id'].tolist())
    names.update(subset_df['name'].tolist())
    names.update(subset_df['ncit.id'].tolist())
    for synonym_set in subset_df['synonyms'].tolist():
        names.update(synonym_set)


def augment_disease(disease):
    if pd.isnull(disease):
        return disease
    new_aliases = set()
    for name in listify(disease['diseaseAliases']):
        new_aliases.update(cells_lookup.get(name, set()))
    for field in ['name', 'displayName']:
        if disease.get(field):
            name = disease[field]
            new_aliases.add(name)
            new_aliases.update(cells_lookup.get(name, set()))
    new_aliases.update(listify(disease['diseaseAliases']))
    disease['diseaseAliases'] = new_aliases

    if 'name' not in disease and 'displayName' in disease:
        disease['name'] = disease['displayName']
    return disease


civic_df['disease'] = civic_df['disease'].apply(augment_disease)

dump_jsonl(civic_df.drop(columns=[c for c in civic_df.columns if c.startswith('_')]), args.output)
