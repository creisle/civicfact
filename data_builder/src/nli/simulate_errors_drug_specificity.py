import argparse
import logging
import os
import random
import re
from copy import deepcopy

import networkx as nx
import pandas as pd

from article_utils.util import dump_jsonl, read_jsonl
from civic.util import load_dump
from nli.util import NEUTRAL_LABEL, concat_text

NUM_DRUG_PICKS = 2


def create_reasoning(row):
    old_therapy = row['original_therapy']
    new_therapy = row['new_therapy']
    if row['claim.source'].count('+') > 2:
        # too hard to manage explanations after this point
        return []
    if row['distance'] > 0:
        # made claim more general
        reason = f'The therapy in the evidence ({old_therapy["name"]}) is more specific than the therapy in the claim ({new_therapy["name"]}).'
    else:
        reason = f'The therapy in the claim ({new_therapy["name"]}) is more specific than the therapy in the evidence ({old_therapy["name"]})'

    if (
        isinstance(row['reviews'], list)
        and len(row['reviews']) == 1
        and row['gold_label_name'] == 'NEI'
        and row['reviews'][0]['username'] == 'bot'
    ):
        return [
            {
                'reason': re.sub(r'\.\s*$', '', row['reviews'][0]['reason'])
                + '. Additionally, t'
                + reason[1:],
                'username': 'bot',
                'label': NEUTRAL_LABEL,
                'script': os.path.basename(__file__),
            }
        ]
    return [
        {
            'reason': reason,
            'username': 'bot',
            'label': NEUTRAL_LABEL,
            'script': os.path.basename(__file__),
        }
    ]


def get_nodes_by_distance(
    g: nx.DiGraph, start_node: str, min_distance: int, max_distance: int
) -> set[tuple[int, str]]:
    """
    Given some starting nodes get nodes between min_distance and max_distance away, return their distance and the node ID.
    The distance will be negative for predecessors of the current node
    """
    picked_nodes = []
    assert max_distance >= min_distance
    paths = [[start_node]]

    while paths:
        current = paths.pop(0)
        if len(current) - 1 >= min_distance:
            picked_nodes.append((len(current) - 1, current[-1]))
        if len(current) - 1 == max_distance:
            continue

        for child in g.successors(current[-1]):
            paths.append(current + [child])

    paths = [[start_node]]

    while paths:
        current = paths.pop(0)
        if len(current) - 1 >= min_distance:
            picked_nodes.append((-1 * (len(current) - 1), current[-1]))
        if len(current) - 1 == max_distance:
            continue

        for parent in g.predecessors(current[-1]):
            paths.append(current + [parent])

    return set(picked_nodes)


def replace_drug_in_claim(row):
    """
    modify the claim object to use the new disease term, also modify the synonyms table in the metadata associated with the claim object
    """
    new_drug = row['new_therapy']

    claim = deepcopy(row['claim'])

    prev_drug_names = {d.strip().lower() for d in claim['content']['therapies']}
    claim['content']['therapies'] = [row['new_therapy']['name']]

    new_synonyms = [{'name': new_drug['name'], 'synonyms': new_drug['synonyms']}]
    for synonym in claim.get('meta', {}).get('synonyms', []):
        if synonym['name'].lower().strip() not in prev_drug_names or synonym['type'] == 'therapy':
            new_synonyms.append(synonym)
    claim['meta']['synonyms'] = new_synonyms
    return claim


logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('--ncit', required=True)
parser.add_argument('--min_path_distance', type=int, default=1)
parser.add_argument('--max_path_distance', type=int, default=6)
parser.add_argument('--civic_dump', required=True)
args = parser.parse_args()

assert args.max_path_distance >= args.min_path_distance
assert args.min_path_distance >= 1

ncit_df = read_jsonl(args.ncit)
g = nx.DiGraph()

parent_mapping = {}
data_mapping = {}
drug_name_to_id = {}

for _, row in ncit_df.iterrows():
    parent_mapping.setdefault(row['id'], []).extend(row['parents'])
    data_mapping[row['id']] = {k: v for k, v in row.items() if k not in ['id', 'parents']}
    for name in [row['name']] + row['synonyms']:
        drug_name_to_id.setdefault(name, set()).add(row['id'])

drug_name_to_id = {name: list(ids)[0] for name, ids in drug_name_to_id.items() if len(ids) == 1}

civic_df = load_dump(args.civic_dump)
drug_df = civic_df.explode('therapies')
drug_df = drug_df[~drug_df['therapies'].isna()]
drug_df['ncitId'] = drug_df['therapies'].str['ncitId']
drug_df['therapyName'] = drug_df['therapies'].str['name']

# get a set of all possible therapy names or codes associated with a PMID
therapies_by_pmid = {}
for _, row in drug_df.drop_duplicates(['ncitId', 'therapyName', 'document.pmid']).iterrows():
    if not pd.isnull(row['ncitId']):
        therapies_by_pmid.setdefault(row['document.pmid'], set()).add(row['ncitId'])
    if not pd.isnull(row['therapyName']):
        therapies_by_pmid.setdefault(row['document.pmid'], set()).add(row['therapyName'])
    therapies_by_pmid.setdefault(row['document.pmid'], set()).update(
        row['therapies']['therapyAliases']
    )

drug_df = drug_df[~drug_df['ncitId'].isna()]
civic_therapies = set(drug_df['ncitId'].unique().tolist())

for code in civic_therapies:
    for parent in parent_mapping[code]:
        g.add_edge(parent, code)

# come up with a list of possible therapy swaps
swaps = {}

for code in civic_therapies:
    nodes = get_nodes_by_distance(g, code, args.min_path_distance, args.max_path_distance)
    swaps[code] = list(nodes)

# now extract the therpy elements from the various claims
input_df = read_jsonl(args.input)
original_columns = [c for c in input_df.columns]

df = input_df.copy()
df = df[df['claim.flat'].str.contains('therapies')]
df['civic_id_null'] = df['civic.id'].fillna('') == ''
df['original_therapy'] = (
    df.claim.str['content']
    .str['therapies']
    .apply(lambda therapies: {t.lower().strip() for t in therapies})
)
df = df[df.original_therapy.str.len() == 1]
df['original_therapy'] = df.original_therapy.apply(lambda t: list(t)).str[0]
df['ncitId'] = df.original_therapy.apply(lambda name: drug_name_to_id.get(name))
df = df[~df.ncitId.isna()]

df['original_therapy'] = df.ncitId.apply(lambda ot: data_mapping[ot])
df['new_therapy'] = df.ncitId.apply(lambda code: swaps.get(code))
df = df[~df.new_therapy.isna()]


def filter_document_drugs(row):
    # filter out any drugs that are already referenced by the same document in CIViC (to avoid creating "real" claims)
    drugs = []
    doc_drugs = therapies_by_pmid.get(row['document.pmid'], set())

    for dist, drug_code in row['new_therapy']:
        names = [drug_code, data_mapping[drug_code]['name']] + data_mapping[drug_code]['synonyms']
        if any([name in doc_drugs for name in names]):
            continue
        drugs.append((dist, drug_code))
    return drugs


df['new_therapy'] = df.apply(filter_document_drugs, axis=1)

# pick X random possible choices
df['new_therapy'] = df.new_therapy.apply(
    lambda t: random.choices(t, k=NUM_DRUG_PICKS) if NUM_DRUG_PICKS < len(t) else t
)
df = df.explode('new_therapy')
df['distance'] = df.new_therapy.str[0]
df['new_therapy'] = df.new_therapy.str[-1].apply(lambda nt: data_mapping[nt])

df['claim'] = df.apply(replace_drug_in_claim, axis=1)
df['claim.flat'] = df.claim.apply(concat_text)
df['gold_label_name'] = NEUTRAL_LABEL
df['reviews'] = df.apply(create_reasoning, axis=1)
df['claim.generated'] = True
df['generated'] = True
df['claim.source'] = df['claim.source'] + ' + modified drug specificity'

df['id'] = df['id'] + '-drs'

print(df)

# drop any generated entries that duplicate an existing entry
df['civic.ground_truth'] = False
df['_id'] = df['claim.flat'] + df['evidence.flat']
input_df['_id'] = input_df['claim.flat'] + input_df['evidence.flat']
input_df['_exists'] = True
df = df.merge(input_df[['_id', '_exists']].drop_duplicates(), how='left')
df = df[df._exists.isna()]
assert df[df.claim.isna()].shape[0] == 0
logging.info(f'created {df.shape[0]} new claims')

dump_jsonl(pd.concat([input_df, df])[original_columns], args.output)
