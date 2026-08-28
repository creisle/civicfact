import argparse
import logging
import os

import pandas as pd

from article_utils.acronyms import sub_acronym
from article_utils.util import dump_jsonl


def sub_acronyms(row, acronyms_df, text_col='text.extracted', pmid_col='pmid'):
    text = row[text_col]
    changes = 0
    if not text:
        return text, changes
    for _, ac_row in acronyms_df[acronyms_df.pmid == row[pmid_col]].iterrows():
        prev_text = text
        text = sub_acronym(text, ac_row.acronym, ac_row.defn)
        if text != prev_text:
            changes += 1
    return text, changes


def main():
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'info').upper())
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to the JSONL file with the text to be subbed')
    parser.add_argument('--sub_column', default='evidence.flat')
    parser.add_argument(
        '--acronyms',
        required=True,
        help='path to the JSONL file containing the acronyms parsed out for each publications',
    )
    parser.add_argument('output', help='path to the JSONL file with the acronyms added')
    args = parser.parse_args()

    logging.info(f'reading: {args.acronyms}')
    acronyms_df = pd.read_json(args.acronyms, lines=True, dtype=False)
    logging.info(f'reading: {args.input}')
    df = pd.read_json(args.input, lines=True, dtype=False)
    logging.info('inserting acronyms')
    df[args.sub_column], df['acronyms_inserted'] = zip(
        *df.apply(
            lambda row: sub_acronyms(row, acronyms_df, args.sub_column, 'document.pmid'), axis=1
        )
    )
    logging.info(f'Inserted {df.acronyms_inserted.sum()} acronyms into {df.shape[0]} examples')
    dump_jsonl(df, args.output)


if __name__ == '__main__':
    main()
