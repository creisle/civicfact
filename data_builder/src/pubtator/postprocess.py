import argparse
import itertools
import logging

import numpy as np
import pandas as pd

from article_utils.db import TextMatch
from article_utils.match_text import exhaustive_match
from article_utils.util import dump_jsonl
from hypothesis.util import is_junk


def get_pubtator_representations(pub):
    result = [
        pub.get('pubtator.name', ''),
        str(pub['pubtator.identifier']),
        pub['pubtator.text'],
        str(pub.get('pubtator.normalized_id', '')),
        str(pub.get('text.extracted', '')),
    ]
    result = [r for r in result if r is not None and not pd.isnull(r) and not r.strip()]
    return {p.lower() for p in set(result) - {'None', '', 'null'}}


def non_null(value):
    if hasattr(value, '__len__'):
        return True
    return not pd.isnull(value)


def simplify_pubtator_elements(row) -> set[str]:
    biotype = row['pubtator.biotype']
    elements = {}
    if biotype == 'gene':
        elements['genes'] = get_pubtator_representations(row)
        if non_null(row['pubtator.normalized_id']):
            elements['genes'].add(row['pubtator.normalized_id'])
    elif biotype == 'variant':
        if non_null(row.get('pubtator.gene_id')):
            elements['genes'] = [str(int(row['pubtator.gene_id']))]
        variants = set()
        if non_null(row.get('pubtator.hgvs')):
            variants.update([row['pubtator.text'], row['pubtator.hgvs']])
        if non_null(row.get('pubtator.normalized_id')):
            variants.add(row['pubtator.normalized_id'].strip('#'))
        elements['variants'] = variants
    elif biotype in {'disease', 'celline', 'cellline'}:
        diseases = set()
        for col in ['do.id', 'cell.derives_from', 'do.synonyms', 'do.name']:
            if non_null(row[col]):
                diseases.update(row[col])
        diseases.update(get_pubtator_representations(row))
        elements['diseases'] = diseases
    elif biotype == 'chemical':
        elements['drugs'] = get_pubtator_representations(row)
    else:
        logging.debug(f'skipping not implemented type. biotype={biotype}')
    return elements


def compute_submatch(row):
    if row['context.match_error']:
        return TextMatch(
            start_pos=None,
            end_pos=None,
            query_text=row['pubtator.text'],
            extracted_text=None,
            score=None,
            table_header='',
            section='',
            error_message='context matching failed so did not attempt submatching annotation text',
        )
    try:
        m = exhaustive_match(
            row['context.extracted'],
            row['pubtator.text'],
            min_aln_score=0.75,
            ref_gap_penalty=-0.5,
            gap_exception=lambda char: is_junk(char) or '-' == char,
        )

        return m
    except ValueError as err:
        return TextMatch(
            start_pos=None,
            end_pos=None,
            query_text=row['pubtator.text'],
            extracted_text=None,
            score=None,
            table_header='',
            section='',
            error_message=str(err),
        )


def flag_obvious_errors(row):
    """
    pubtator makes some pretty obvious mistakes in some cases, ex NM_01234 which would obviously a transript to a human gets labelled a a gene called NM
    """
    if row['pubtator.text'] == 'NM' and row['pubtator.suffix'].startswith('_'):
        return 'marked NM_ transcript as a gene named NM'

    return ''


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument(
        '--disease_ontology',
        required=True,
        help='path to the disease ontology JSONL file containing simplified DO records',
    )
    parser.add_argument(
        '--cells',
        required=True,
        help='path to the cell line JSONL file containing simplified cellosaurus/clo/ncit records',
    )
    parser.add_argument(
        'output',
        help='path to the JSONL output file containing the processed annotations',
    )

    args = parser.parse_args()

    logging.info(f'reading: {args.cells}')
    cell_df = pd.read_json(args.cells, lines=True, dtype=False).fillna('')
    cell_df = cell_df[cell_df['cellosaurus.id'] != '']
    cell_df = cell_df.rename(columns={c: f'cell.{c}' for c in cell_df.columns})
    cell_df = cell_df.rename(columns={'cell.cellosaurus.id': 'pubtator.identifier'})
    # pubtator uses CVCL:### whereas cellosaurus has it as CVCL_###
    cell_df['pubtator.identifier'] = cell_df['pubtator.identifier'].str.replace('_', ':')
    cell_df['pubtator.database'] = 'cvcl'

    logging.info(f'reading: {args.disease_ontology}')
    mesh_df = pd.read_json(args.disease_ontology, lines=True, dtype=False)
    mesh_df = mesh_df.explode('mesh')
    mesh_df = (
        mesh_df.groupby(['mesh'])
        .agg(
            {
                'id': 'unique',
                'synonyms': lambda series: list(
                    set(itertools.chain.from_iterable(series.tolist()))
                ),
                'parents': lambda series: list(set(itertools.chain.from_iterable(series.tolist()))),
                'name': 'unique',
            }
        )
        .reset_index()
    )
    mesh_df = mesh_df.rename(
        columns={
            'mesh': 'pubtator.identifier',
            **{c: f'do.{c}' for c in mesh_df.columns if c != 'mesh'},
        }
    )
    mesh_df['pubtator.database'] = 'ncbi_mesh'

    logging.info(f'reading: {args.input}')
    df = pd.read_json(args.input, lines=True, dtype=False)
    errors_df = df[df["text.match_error"].fillna("") != ""]
    print(errors_df.columns)
    print(errors_df.groupby(['pubtator.biotype', 'pubtator.text']).agg({'pmid': 'nunique'}))
    logging.info(f'dropping failed primary matches: {errors_df.shape[0]}')

    df['error'] = df.apply(flag_obvious_errors, axis=1)
    logging.info(f'dropping obvious pubtator errors: {df[df.error != ""].shape[0]}')
    df = df[df.error == '']

    df = df[df["text.match_error"].fillna('') == '']
    df = df.rename(
        columns={c: c.replace('text.', 'context.') for c in df.columns if c.startswith('text.')}
    )

    df['text'] = df.apply(compute_submatch, axis=1)

    for col_name, match_prop_name in {
        'text.start': 'start_pos',
        'text.end': 'end_pos',
        'text.score': 'score',
        'text.extracted': 'extracted_text',
        'text.match_error': 'error_message',
        'text.table_header': 'table_header',
        'text.section': 'section',
    }.items():
        df[col_name] = df['text'].str[match_prop_name]

    errors_df = df[df["text.match_error"].fillna("") != ""]
    print(errors_df.groupby(['pubtator.biotype', 'pubtator.text']).agg({'pmid': 'nunique'}))
    logging.info(f'dropping failed secondary matches: {errors_df.shape[0]}')
    df = df[df['text.match_error'].fillna('') == '']

    # update the positions for the global position of the match
    df['text.start'] = df['context.start'] + df['text.start']
    df['text.end'] = df['context.start'] + df['text.end']

    logging.info(f'adding alias terms from the disease ontology to pubtator disease annotations')
    df = df.merge(mesh_df, how='left', on=['pubtator.identifier', 'pubtator.database'])
    df = df.merge(cell_df, how='left', on=['pubtator.identifier', 'pubtator.database'])

    df['pubtator.elements'] = df.apply(simplify_pubtator_elements, axis=1)

    def sanity_check(row):
        if len(row['text.extracted']) != (row['text.end'] - row['text.start']):
            print({k: v for k, v in row.items() if k.startswith('text') or k.startswith('context')})
            raise AssertionError(f'failed length check')

    df.apply(sanity_check, axis=1)
    df = df[
        [c for c in df.columns if c.startswith('context.') or c.startswith('text.')]
        + [
            'pmcid',
            'pmid',
            'pubtator.elements',
            'pubtator.text',
            'pubtator.offset',
            'pubtator.biotype',
            'pubtator.prefix',
            'pubtator.suffix',
        ]
    ]

    dump_jsonl(df, args.output)


if __name__ == '__main__':
    main()
