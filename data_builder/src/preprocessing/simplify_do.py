import argparse
import itertools
import json
import logging
import re

import pandas as pd
import requests_cache

from article_utils.ncbi import get_mesh_terms
from article_utils.util import dump_jsonl

requests_cache.install_cache('.requests_cache')


def agg_string(series):
    return series.dropna().str.join(';')


def agg_lists(series):
    result = []
    series.dropna().apply(lambda item: result.extend(item))
    return result


def expand_synonyms(names):
    # fix her2-receptor positive issue, receptor is generally inferred when not given
    subs = [(r'-receptor\b', ''), (r'\bpositive\b', 'amplified'), (r'\s\bpositive\b', '')]
    for name in list(names):
        for ordering in itertools.permutations(subs, r=len(subs)):
            for target, replacement in ordering:
                names.add(re.sub(target, replacement, name, re.IGNORECASE))
    return sorted(names)


def process_node(node):
    mesh = []
    ncit = []
    for xref in node['meta'].get('xrefs', []):
        if xref['val'].startswith('MESH:'):
            mesh.append(xref['val'])
        if xref['val'].startswith('NCI:'):
            ncit.append(xref['val'].replace('NCI:', ''))
    synonyms = []
    for syn in node['meta'].get('synonyms', []):
        synonyms.append(syn['val'])

    doids = [node['id'].split('/')[-1].replace('_', ':')]
    for val in node.get('basicPropertyValues', []):
        if val['pred'].endswith('hasAlternativeId'):
            doids.append(val['val'])
        elif val['pred'] == 'hasExactSynonym':
            synonyms.append(val['val'])
    definition = ''
    if node['meta'].get('definition'):
        definition = node['meta']['definition']['val']

    return {
        'name': node['lbl'],
        'id': doids,
        'synonyms': sorted(set(synonyms)),
        'mesh': sorted(set(mesh)),
        'ncit': sorted(set(ncit)),
        'definition': definition,
    }


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('ncit')
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.input}')
with open(args.input, 'r') as fh:
    data = json.load(fh)

nodes = []
parents = {}

for graph in data['graphs']:
    for node in graph['nodes']:
        id = node['id'].split('/')[-1]
        if not id.startswith('DOID_'):
            continue
        nodes.append(process_node(node))

    for edge in graph['edges']:
        sub = edge['sub'].split('/')[-1].replace('_', ':')
        obj = edge['obj'].split('/')[-1].replace('_', ':')
        if edge['pred'] == 'is_a' and sub.startswith('DOID') and obj.startswith('DOID'):
            parents.setdefault(sub, []).append(obj)

df = pd.DataFrame.from_records(nodes)
df = df.explode('id')

# augment this data with MESH
mesh_ids = set(df.explode('mesh').fillna('').mesh.unique().tolist()) - {''}
mesh_records = get_mesh_terms(sorted(mesh_ids))
mesh_df = pd.DataFrame.from_records(mesh_records)
mesh_df['mesh'] = mesh_df['ds_meshui'].apply(lambda mid: f'MESH:{mid}'.upper())
mesh_df = mesh_df[['mesh', 'ds_meshterms']].merge(df.explode('mesh'), on=['mesh'])
mesh_df = mesh_df.groupby(['id']).agg({'mesh': set, 'ds_meshterms': agg_lists}).reset_index()

df = df.drop(columns=['mesh']).merge(
    mesh_df[['id', 'mesh', 'ds_meshterms']], how='left', validate='one_to_one'
)

# Add NCIt synonyms/names
logging.info(f'reading: {args.ncit}')
ncit_df = pd.read_json(args.ncit, lines=True, dtype=False)
ncit_df = ncit_df.rename(columns={c: f'ncit.{c}' for c in ncit_df.columns})
ncit_df = ncit_df.merge(
    df.explode('ncit')[['name', 'ncit', 'id']], left_on=['ncit.id'], right_on=['ncit']
).fillna('')
ncit_df = ncit_df.groupby(['id']).agg({'ncit': set, 'ncit.synonyms': agg_lists}).reset_index()
df = df.drop(columns=['ncit']).merge(
    ncit_df[['id', 'ncit', 'ncit.synonyms']], how='left', validate='one_to_one'
)


def gather_synonyms(row):
    names = [row['name']]
    for field_name in ['ncit.synonyms', 'ds_meshterms', 'synonyms']:
        try:
            names.extend(row[field_name])
        except TypeError:
            pass
    return set(names) - {''}


df['synonyms'] = df.apply(gather_synonyms, axis=1).apply(expand_synonyms)
df = df.drop(columns=['ds_meshterms', 'ncit.synonyms'])
df['parents'] = df['id'].apply(lambda id: parents.get(id, []))

dump_jsonl(df, args.output)
