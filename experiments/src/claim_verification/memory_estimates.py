import argparse
import os
import re
import subprocess

import pandas as pd


def get_model_name(text):
    m = re.search(r'Memory Usage for loading `([^`]+)`', text)
    return m.group(1)


def replace_unicode_range(text, ord_min, ord_max, replacement_char):
    return ''.join(
        [replacement_char if ord(c) >= ord_min and ord(c) <= ord_max else c for c in text]
    )


def parse_memory_table(text):
    header = None
    rows = []

    text = replace_unicode_range(text, 9474, 9500, '|')
    # text = re.sub(u'[\u9474-\u9500]', '|', text)

    for line in text.split('\n'):
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if not cells:
            continue
        if cells[0] == 'dtype':
            header = cells
        elif header and len(cells) == len(header):
            rows.append({h: c for h, c in zip(header, cells)})
    df = pd.DataFrame.from_records(rows)
    df['model'] = get_model_name(text)
    return df


parser = argparse.ArgumentParser()
parser.add_argument('model_name')
parser.add_argument('--hf_token', default=os.environ.get('HF_TOKEN'))
parser.add_argument('output')
args = parser.parse_args()

if not args.hf_token:
    parser.error('hf_token was not given and the HF_TOKEN environment variable is not defined')

output = subprocess.check_output(
    ['accelerate', 'estimate-memory', args.model_name],
    env={**os.environ, 'HF_TOKEN': args.hf_token},
).decode()

df = parse_memory_table(output)
df
output = subprocess.check_output(['accelerate', 'estimate-memory', args.model_name], env=os.environ)

df.to_csv(args.output, index=False)
