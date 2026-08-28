"""
Code to interact with the hypothes.is API
"""
import argparse
import json
import logging
import os

from .api import fetch_annotations_by_group

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api_key',
        default=os.environ.get('HYPOTHESIS_API_KEY'),
        help='API key to access hypothesis content. Defaults to the ENV variable HYPOTHESIS_API_KEY',
    )
    parser.add_argument('group', help='The annotation group you wish to download annotations from')
    parser.add_argument('output', help='Path to the JSONL output file to write annotations to')
    args = parser.parse_args()

    results = fetch_annotations_by_group(args.group, args.api_key)

    with open(args.output, 'w') as fh:
        logging.info(f'writing: {args.output}')
        for ann in results:
            fh.write(json.dumps(ann, sort_keys=True) + '\n')
