import argparse
import logging

import pandas as pd

from article_utils.util import dump_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.input}')
names = [
    'id',
    'concept IRI',
    'parents',
    'synonyms',
    'definition',
    'display_name',
    'concept status',
    'type',
    'concept in subset',
]

df = pd.read_csv(args.input, sep='\t', names=names).fillna('')
df['synonyms'] = df['display_name'] + '|' + df.synonyms
df['synonyms'] = df.synonyms.str.split('|').apply(lambda names: [n for n in names if n.strip()])
df['name'] = df['display_name']
df.loc[df.name == '', 'name'] = df.synonyms.str[0]
df['synonyms'] = df.synonyms.apply(lambda names: sorted(set(names) - {''}))
df['parents'] = df.parents.str.split('|').apply(lambda names: [n for n in names if n.strip()])

dump_jsonl(df[['id', 'name', 'synonyms', 'type', 'parents']], args.output)
