import argparse
import logging
import os
import re
from copy import deepcopy

from article_utils.util import dump_jsonl, read_jsonl
from nli.util import concat_text


def ablate_claim_element(claim, element):
    content = []
    matched = False
    for col, val in claim['content']:
        if col == element:
            matched = True
        else:
            content.append([col, val])
    if not matched:
        return claim
    claim = deepcopy(claim)
    claim['content'] = content
    return claim


def reduce_claim(claim, element):
    content = []
    for col, val in claim['content']:
        if col == element:
            content.append([col, val])
    claim = deepcopy(claim)
    claim['content'] = content
    return claim


def empty_claim(_):
    return [{'content': [], 'meta': {}, 'type': 'claim'}]


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('--force', action='store_true', default=False)
args = parser.parse_args()


m = re.match(r'(?P<build_dir>.*builds)/civicfact-(?P<version>[^/]+)/data.jsonl.gz$', args.input)
assert m
build_dir = m.group('build_dir')
version = m.group('version')
df = read_jsonl(args.input)
claim_columns = set()


def extract_header(claim: dict):
    claim_columns.update(claim)


df['claim'].apply(extract_header)
logging.info(f'claim header elements: {claim_columns}')


for element in claim_columns:
    output_file = os.path.join(build_dir, f'civicfact_ab_{element}-{version}', 'data.jsonl.gz')
    if os.path.exists(output_file) and not args.force:
        logging.warning(f'file already exists. Use --force to overwrite: {output_file}')
        continue
    temp_df = df.copy()
    temp_df['claim'] = temp_df.claim.apply(lambda claims: ablate_claim_element(claims, element))
    temp_df['claim.flat'] = temp_df.claim.apply(concat_text)
    temp_df['ablation'] = element
    dump_jsonl(temp_df, output_file)

for element in claim_columns:
    output_file = os.path.join(build_dir, f'civicfact_only_{element}-{version}', 'data.jsonl.gz')
    if os.path.exists(output_file) and not args.force:
        logging.warning(f'file already exists. Use --force to overwrite: {output_file}')
        continue
    temp_df = df.copy()
    temp_df['claim'] = temp_df.claim.apply(lambda claims: reduce_claim(claims, element))
    temp_df['claim.flat'] = temp_df.claim.apply(concat_text)
    temp_df['only'] = element
    dump_jsonl(temp_df, output_file)

output_file = os.path.join(build_dir, f'civicfact_empty-{version}', 'data.jsonl.gz')
if os.path.exists(output_file) and not args.force:
    logging.warning(f'file already exists. Use --force to overwrite: {output_file}')
else:
    temp_df = df.copy()
    temp_df['claim'] = temp_df.claim.apply(lambda claims: empty_claim(claims))
    temp_df['claim.flat'] = temp_df.claim.apply(concat_text)
    temp_df['ablation'] = 'all'
    dump_jsonl(temp_df, output_file)


output_file = os.path.join(build_dir, f'civicfact_dummy-{version}', 'data.jsonl.gz')
if os.path.exists(output_file) and not args.force:
    logging.warning(f'file already exists. Use --force to overwrite: {output_file}')
else:
    temp_df = df.copy()
    temp_df['claim'] = temp_df.claim.apply(lambda claims: empty_claim(claims))
    temp_df['claim.flat'] = temp_df.gold_label_name
    temp_df['ablation'] = 'dummy'
    dump_jsonl(temp_df, output_file)
