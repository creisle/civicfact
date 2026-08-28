import argparse
import logging

import pandas as pd
from article_utils.acronyms import has_acronym
from article_utils.util import dump_jsonl
from civic.util import load_dump
from nli.util import (
    LABEL_COL,
    SELECTION_DELIMITER,
    Text,
    convert_col_to_words,
    convert_enum_to_words,
)


def build_alias_table(claim) -> str:
    aliases = []

    for component in claim:
        if 'meta' not in component:
            logging.warning(f'claim component has no metadata: {component}')
            continue
        if 'synonyms' not in component['meta'] or not component['meta']['synonyms']:
            logging.warning(f'claim metadata has no synonyms: {component}')
            continue
        for synonym in component['meta']['synonyms']:
            if synonym['name'] and synonym['synonyms']:
                aliases.append(
                    {
                        'term': synonym['name'],
                        'term_type': synonym['type'],
                        'term_aliases': set(synonym['synonyms']) - {synonym['name']},
                    }
                )
    if not aliases:
        return ''

    aliases_df = pd.DataFrame.from_records(aliases)
    aliases_df['term_type'] = aliases_df.term_type.apply(
        lambda tt: tt if tt != 'molecularProfile' else 'variant'
    )
    aliases_df = aliases_df.drop_duplicates(['term', 'term_type']).fillna('')
    aliases_df = aliases_df[aliases_df.term != '']
    aliases_df = aliases_df[aliases_df.term_aliases.apply(lambda aliases: bool(aliases))]

    rows = ['<tr><th>Term</th><th>Type</th><th>Aliases</th></tr>']
    for _, row in aliases_df.iterrows():
        term_aliases = ' '.join(
            [f'<div class="chip" title="{a}">{a}</div>' for a in row.term_aliases]
        )
        rows.append(f'<tr><td>{row.term}</td><td>{row.term_type}</td><td>{term_aliases}</td></tr>')
    if len(rows) < 2:
        logging.warning(f'no non-name aliases found: {aliases}')
        return ''
    return '<table>' + " ".join(rows) + '</table>'


def create_hyp_links(row):
    links = []
    base_url = f'https://pubmed.ncbi.nlm.nih.gov/{row["document.pmid"]}'
    if row['document.pmcid']:
        base_url = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{row["document.pmcid"]}'
    for ev in row.evidence:
        if ev['meta'].get('hypothesis.id'):
            link = f'{base_url}/#annotations:{ev["meta"]["hypothesis.id"]}'
            links.append(f'<a href={link}>{link}</a>')

    return '<br>'.join(links)


def build_acronym_html(df, acronyms_df):
    df = df.merge(acronyms_df, on=['document.pmid'])
    df['_has_acronym'] = df.apply(
        lambda row: has_acronym(row['evidence.flat'], row.acronym), axis=1
    )
    df = df[df._has_acronym]
    df = (
        df.groupby(['id', 'document.pmid'])
        .apply(lambda group_df: group_df[['acronym', 'defn']].to_dict('records'))
        .rename('acronyms')
        .reset_index()
    )

    def build_html(acronyms):
        if len(acronyms) == 0:
            return ''
        html = ['<ul class="acronyms">']
        for item in acronyms:
            html.append('<li>{acronym}: {defn}</li>'.format(**item))
        html.append('</ul>')
        return '<br/><h3>Acronyms</h3>' + ''.join(html)

    # now build the list for the acronyms
    df['acronym_html'] = df['acronyms'].apply(build_html)
    return df


def build_html(df, acronyms_df, selection_delimiter=SELECTION_DELIMITER, debug=False):
    df = df.copy()

    def build_html_component(selections: list[Text]) -> str:
        paragraphs = []
        for selection in selections:
            if selection['type'] == 'table':
                # new paragraph
                table = []
                for col, val in selection['content']:
                    table.append(
                        f'<tr><th>{convert_col_to_words(col)}</th><td>{convert_enum_to_words(val)}</td></tr>'
                    )

                paragraphs.append('<table>' + " ".join(table) + '</table>')
            elif paragraphs:
                paragraphs[-1] += selection_delimiter + selection['content']
            else:
                paragraphs.append(selection['content'])
        return ' '.join([f'<p style="white-space: pre-wrap">{p}</p>' for p in paragraphs])

    df['evidence_html'] = df.evidence.apply(build_html_component)
    df['claim_html'] = df.claim.apply(build_html_component)
    df['info_html'] = df['claim'].apply(build_alias_table)

    acronym_html_df = build_acronym_html(df, acronyms_df)

    df = df.merge(
        acronym_html_df[['id', 'document.pmid', 'acronym_html']],
        on=['id', 'document.pmid'],
        validate='one_to_one',
        how='left',
    )

    df['info_html'] = df.info_html.fillna('') + df.acronym_html.fillna('')

    if debug:
        df['hyp_links'] = df.apply(create_hyp_links, axis=1)

    for field in ['claim_html', 'evidence_html', 'info_html']:
        if df[df[field].fillna('') == ''].shape[0] != 0:
            logging.warning(
                f'some rows ({df[df[field].fillna("") == ""].shape[0]}) do not have {field}'
            )

    df.loc[df.info_html != '', 'info_html'] = (
        '<div class="ad_info"><h2>Additional Information</h2>'
        + df[df.info_html != ''].info_html
        + '</div>'
    )

    df['html'] = (
        '<div class="example">'
        + (df['hyp_links'] if debug else '')
        + '<div class="question"><h2>Given a claim and some evidence, determine the stance of the evidence toward the claim.</h2></div>'
        + '<div class="claim"><h2>Claim</h2>'
        + df.claim_html
        + '</div><div class="evidence"><h2>Evidence</h2>'
        + df.evidence_html
        + '</div>'
        + df.info_html
        + '<br><div class="question"><h2>Does the evidence support the claim? Use of the following labels: SUPPORTS, REFUTES, NEI. Give your reasoning</h2></div>'
        + '</div>'
    )

    assert df[df.html.fillna('') == ''].shape[0] == 0, 'some rows do not have html'

    return df


def columns_are_unique_key(df, columns):
    total = df.shape[0]
    unique = df.drop_duplicates(columns).shape[0]

    if unique != total:
        dups_df = df.copy()
        dups_df['freq'] = 1
        dups_df = dups_df.groupby(columns).agg({'freq': 'sum'}).reset_index()
        dups_df = dups_df[dups_df.freq > 1]
        print(dups_df)
        raise AssertionError(f'columns ({columns}) are not a unique index ')


def main(args):
    civic_df = load_dump(args.civic_dump)
    civic_df = civic_df.rename(columns={c: f'civic.{c}' for c in civic_df.columns})
    civic_df = civic_df.rename(columns={'civic.id': 'claim.id', 'civic.pmid': 'document.pmid'})
    civic_df['claim.id'] = civic_df['claim.id'].astype(str)
    civic_df['document.pmid'] = civic_df['document.pmid'].astype(str)

    logging.info(f'reading: {args.acronyms}')
    acronyms_df = pd.read_json(args.acronyms, lines=True, dtype=False).rename(
        columns={'pmid': 'document.pmid'}
    )

    logging.info(f'reading: {args.input}')
    df = pd.read_json(args.input, lines=True, dtype=False)

    total_examples = df.shape[0]
    logging.info(f'read {total_examples} input examples')

    columns_are_unique_key(df, ['id', 'document.pmid'])

    df = df.merge(civic_df, on=['claim.id', 'document.pmid'], how='left')
    assert (
        total_examples == df.shape[0]
    ), f'examples in {total_examples} does not match examples out {df.shape[0]}'
    # also dump an html version of examples for loading in doccano
    html_df = build_html(df, acronyms_df, debug=args.debug)
    assert html_df[html_df.html.fillna('') == ''].shape[0] == 0

    assert (
        total_examples == html_df.shape[0]
    ), f'examples in {total_examples} does not match examples out {html_df.shape[0]}'

    # html_df['metadata'] = html_df.apply(pull_metadata, axis=1)
    html_df['fix_flagged'] = html_df.flagged.fillna(False).astype(bool)

    def collect_labels(row):
        if 'reviews' not in row:
            return [row[LABEL_COL]]
        labels = []
        for review in row['reviews']:
            if review['reason']:
                labels.append(row[LABEL_COL].upper() + '. ' + review['reason'])
        if not labels:
            labels = [row[LABEL_COL]]
        return labels

    if args.label_column == 'gold_label_name':
        html_df['label'] = html_df.apply(collect_labels, axis=1)
    elif args.label_column is None:
        html_df['label'] = html_df.apply(lambda _: [], axis=1)
    else:
        html_df['label'] = html_df[args.label_column].apply(
            lambda l: l if isinstance(l, list) else ([l] if l else [])
        )

    logging.info(f'writing {html_df.shape[0]} examples')

    html_df = html_df.rename(columns={'html': 'text'})
    assert html_df[html_df.text.fillna('') == ''].shape[0] == 0

    dump_jsonl(
        html_df[
            ['text', 'label', 'display_id', 'id', 'claim.flat', 'evidence.flat', 'document.pmid']
            + args.metadata
        ].sort_values(['document.pmid']),
        args.output,
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--civic_dump', help='path to the JSONL civic dump', required=True)
    parser.add_argument('input', help='path to the JSONL of examples to be partitioned')
    parser.add_argument('output', help='path to the output HTML JSONL file')
    # parser.add_argument('--elements', required=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--label_column')
    parser.add_argument(
        '--acronyms',
        required=True,
        help='path to the file containing the extracted acronym definitions by publication ID',
    )
    parser.add_argument(
        '--metadata',
        nargs='+',
        default=[],
        help='addition fields to put into the metadata dict. default includes: id, display_id',
    )
    args = parser.parse_args()
    main(args)
