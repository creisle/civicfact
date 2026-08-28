import argparse
import logging
import os
import random
import re
from copy import deepcopy

import pandas as pd

from article_utils.util import dump_jsonl, read_jsonl
from nli.util import concat_text


def replace_mp_in_claim(row):
    """
    modify the claim object to use the new disease term, also modify the synonyms table in the metadata associated with the claim object
    """
    new_mut = row['new_mp']

    claim = deepcopy(row['claim'])
    prev_mut = claim['content']['molecularProfile']
    claim['content']['molecularProfile'] = new_mut

    synonyms = []

    for synonym in claim.get('meta', {}).get('synonyms', []):
        if synonym['name'] == prev_mut:
            continue
        synonyms.append(synonym)

    claim['meta']['synonyms'] = synonyms

    return claim


def create_new_mp_states(row):
    claim = row['claim']['content']
    if any([char in claim['molecularProfile'] for char in '()[]/;!']):
        # skip any super complex MPs as we are likely to create errors modiying them
        return None
    words = set(claim['molecularProfile'].lower().split(' '))
    if len(words & {'and', 'or'}) == 2:
        # skip any super complex MPs as we are likely to create errors modiying them
        return None
    new_mps = []

    if 'and' in words:
        new_mps.append(re.sub(r'\band\b', 'OR', claim['molecularProfile'], re.IGNORECASE))
    elif 'or' in words:
        new_mps.append(re.sub(r'\bor\b', 'AND', claim['molecularProfile'], re.IGNORECASE))

    mp = re.sub(r'(^|\b)NOT\s+', '', claim['molecularProfile'], re.IGNORECASE)
    if mp != claim['molecularProfile']:
        new_mps.append(mp)

    variant_list = [
        v.strip()
        for v in re.split(r'\b(and|or)\b', claim['molecularProfile'], re.IGNORECASE)
        if v.strip()
    ]
    new_variants = {}
    for v in variant_list:
        numbers = re.search(r'\d{3,}', v)
        if numbers:
            numbers = numbers.group(0)
            new_numbers = list(numbers)
            random.shuffle(new_numbers)
            new_numbers = ''.join(new_numbers)
            if numbers != new_numbers:
                assert len(numbers) == len(new_numbers)
                new_variants[v] = v.replace(numbers, new_numbers)

    if new_variants:
        pick = random.choice(list(new_variants))
        new_mp = claim['molecularProfile'].replace(pick, new_variants[pick])
        assert claim['molecularProfile'] != new_mp
        new_mps.append(new_mp)

    return new_mps


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.input}')
input_df = read_jsonl(args.input)
original_columns = input_df.columns.tolist()
df = input_df.copy()

# print(df)
df['new_mp'] = df.apply(create_new_mp_states, axis=1)
df = df.explode('new_mp')
df = df[~df.new_mp.isna()]
df['claim.source'] = df['claim.source'] + ' + modified mp'

df['id'] = df['id'] + '-mp'
df['gold_label_name'] = 'REVISIONS'

df['claim'] = df.apply(replace_mp_in_claim, axis=1)
df['claim.flat'] = df.claim.apply(concat_text)
df['claim.generated'] = True
df['generated'] = True

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
