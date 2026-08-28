import argparse
import json

import pandas as pd

from civic.util import filter_casei_dups

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

with open(args.input, 'r') as fh:
    data = json.load(fh)

rows = []
for graph in data['graphs']:
    for node in graph['nodes']:
        id = node['id'].split('/')[-1]
        if not id.startswith('HP_'):
            continue
        synonyms = []
        for syn in node.get('meta', {}).get('synonyms', []):
            synonyms.append(syn['val'])

        rows.append(
            {
                'name': node['lbl'],
                'id': node['id'].split('/')[-1].replace('_', ':'),
                'synonyms': ';'.join(filter_casei_dups(sorted(set(synonyms) | {node['lbl']}))),
            }
        )


df = pd.DataFrame.from_records(rows)

df.to_csv(args.output, index=False)
