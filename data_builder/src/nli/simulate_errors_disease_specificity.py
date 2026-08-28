import argparse
import logging
import random
from copy import deepcopy

import networkx as nx
import pandas as pd

from article_utils.util import dump_jsonl, read_jsonl
from nli.util import NEUTRAL_LABEL, concat_text


def pick_new_disease(graph: nx.DiGraph, doid, distance) -> list[str]:
    """
    Given a starting disease and some distance, gather terms x nodes away from the current term
    """
    if distance <= 0:
        return [doid]

    result = []
    for child in graph.successors(doid):
        result.extend(pick_new_disease(graph, child, distance - 1))

    for parent in graph.predecessors(doid):
        result.extend(pick_new_disease(graph, parent, distance - 1))
    return result


def replace_disease_in_claim(graph, row):
    """
    modify the claim object to use the new disease term, also modify the synonyms table in the metadata associated with the claim object
    """
    disease = graph.nodes[row['new_disease']]
    prev_disease = row['disease']

    claim = deepcopy(row['claim'])

    prev_disease = claim['content']['disease']
    claim['content']['disease'] = disease['name']

    for synonym in claim.get('meta', {}).get('synonyms', []):
        if (
            synonym['name'].lower().strip() == prev_disease.lower().strip()
            and synonym['type'] == 'disease'
        ):
            synonym['name'] = disease['name']
            synonym['synonyms'] = disease['synonyms']

    return claim


def create_reasoning(row):
    claim_disease = row['claim']['content']['disease']
    evidence_disease = row['disease']
    if row['claim.source'].count('+') > 3:
        # too hard to manage explanations after this point
        return []
    if row['distance'] > 0:
        # made claim more general
        reason = f'The disease in the evidence ({evidence_disease}) is more specific than the disease in the claim ({claim_disease}).'
    else:
        reason = f'The disease in the claim ({claim_disease}) is more specific than the disease in the evidence ({evidence_disease}).'
    if row['gold_label_name'] != 'NEI':
        if not row['reviews']:
            return [{'reason': reason, 'username': 'bot', 'label': NEUTRAL_LABEL}]
        elif len(row['reviews']) == 1 and row['reviews'][0]['username'] == 'bot':
            return [
                {
                    'reason': row['reviews'][0] + ' Also, t' + reason[1:],
                    'username': 'bot',
                    'label': NEUTRAL_LABEL,
                }
            ]
    return []


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('--do', required=True)
parser.add_argument('--distance', type=int, default=2)
args = parser.parse_args()

logging.info(f'reading: {args.do}')
do_df = pd.read_json(args.do, lines=True, dtype=False)
graph = nx.DiGraph()

for _, row in do_df.iterrows():
    graph.add_node(row['id'], **dict(row))
    for parent in row['parents']:
        graph.add_edge(row['id'], parent, pred='is_a')

logging.info(f'reading: {args.input}')
input_df = read_jsonl(args.input)
original_columns = input_df.columns.tolist()
df = input_df[input_df.gold_label_name != NEUTRAL_LABEL].copy()

# discard any rows without a disease
print(df[['claim.source', 'claim']])
df['disease'] = df['claim'].str['content'].str['disease'].str.lower()
print(df)
df = df[~df.disease.isna()]
print(df)
do_df = do_df.rename(columns={'id': 'doid', 'name': 'disease'})
do_df['disease'] = do_df.disease.str.lower()
df = df.merge(do_df)
print(df.columns.to_list())
# randomly choose a distance away from the target at least 2 hops away
distances = list(range(2, args.distance + 1))
df['distance'] = random.choices(distances, k=df.shape[0])
df['new_disease'] = df.apply(
    lambda row: random.choices(pick_new_disease(graph, row['doid'], row['distance']), k=3), axis=1
)
df = df[df.new_disease != '']
df = df.explode('new_disease')
df['_drop'] = df.new_disease == 'disease'
logging.info(f'dropping rows (n={df._drop.sum()}) where new selected disease is "disease"')
df = df[df._drop == False]

df['claim'] = df.apply(lambda row: replace_disease_in_claim(graph, row), axis=1)
df['claim.flat'] = df.claim.apply(concat_text)
df['gold_label_name'] = NEUTRAL_LABEL
# add the reasoning as a "review"
df['reviews'] = df.apply(create_reasoning, axis=1)
df['claim.generated'] = True
df['generated'] = True
df['claim.source'] = df['claim.source'] + ' + modified disease specificity'

df['id'] = df['id'] + '-dis'

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
