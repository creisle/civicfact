import argparse
import logging
import re

import pandas as pd

from article_utils.util import dump_jsonl

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('civic')
parser.add_argument('phenotypes')
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.civic}')
civic_df = pd.read_json(args.civic, lines=True, dtype=False)
logging.info(f'reading: {args.phenotypes}')
phenotypes_df = pd.read_csv(args.phenotypes).fillna('')

phenotypes_lookup = {}

for _, row in phenotypes_df.iterrows():
    phenotypes_lookup[row['id']] = row['synonyms'].split(';')


def augment_phenotypes(phenotypes):
    result = []
    for phenotype in phenotypes:
        phenotype['phenotypeAliases'] = phenotypes_lookup.get(phenotype['hpoId'], [])
        result.append(phenotype)
    return result


civic_df['phenotypes'] = civic_df['phenotypes'].apply(augment_phenotypes)

dump_jsonl(civic_df, args.output)
