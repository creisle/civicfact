import argparse
import json
import logging
import os

import pandas as pd

from .api import post_annotation


def upload_example(row, group_id, api_key, tags=[]):
    if not pd.isnull(row["document.pmcid"]) and row["document.pmcid"] != '':
        url = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{row["document.pmcid"]}'
    else:
        assert (
            row["document.pmid"] or ''
        ) != '', 'each uploaded entry must have a valid PMCID or PMID'
        url = f'https://pubmed.ncbi.nlm.nih.gov/{row["document.pmid"]}'
    tag = row["claim.id"] + '_ex:' + row['evidence.id']

    for i, selection in enumerate(row['evidence']):
        logging.info(f'uploading {tag} {i + 1}/{len(row["evidence"])} to {url}')
        selection_metadata = {
            k: v for k, v in selection['meta'].items() if not k.startswith('hypothesis.')
        }  # exclude hypothesis metadata
        post_annotation(
            group_id,
            api_key,
            url,
            row['document.title'],
            selection['content']
            if selection['type'] != 'table'
            else ' '.join([v for _, v in selection['content']]),
            [tag, *tags],
            json.dumps({'claim': row['claim'], 'meta': selection_metadata}),
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api_key',
        default=os.environ.get('BOT_HYPOTHESIS_API_KEY'),
        help='API key to access hypothesis content. Defaults to the ENV variable BOT_HYPOTHESIS_API_KEY',
    )
    parser.add_argument(
        '--group',
        help='The annotation group you wish to download annotations from',
        default='ADmgdiEi',
    )
    parser.add_argument(
        'input',
        help='The input JSONL file with the generated examples. Any non-neutral or examples with existing hypothesis ids will be ignored.',
    )
    parser.add_argument(
        '--no_skip',
        default=False,
        help='upload all entries, do not skip non-generated or entries which already contain hypothesis IDs',
        action='store_true',
    )
    args = parser.parse_args()

    logging.info(f'reading: {args.input}')
    df = pd.read_json(args.input, lines=True, dtype=False)

    skipped = 0
    loaded = 0

    for _, row in df.iterrows():
        if not row['evidence.generated'] and not args.no_skip:
            skipped += 1
            continue

        hypothesis_selections = 0
        for selection in row['evidence']:
            if 'hypothesis.id' in selection['meta']:
                hypothesis_selections += 1

        if hypothesis_selections > 0 and not args.no_skip:
            if hypothesis_selections < len(row['evidence']):
                logging.error(
                    f'evidence ({row["evidence.id"]}) selections do not all have hypothesis IDs'
                )
            skipped += 1
            continue
        upload_example(row, args.group, args.api_key)
        loaded += 1

    logging.info(f'skipped {skipped} examples')
    logging.info(f'loaded {loaded} examples')
