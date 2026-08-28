"""
Verify the annotation in our biocXML version of the article and extract the actual text from the XML
add context elements such as the table header where applicable
"""

import argparse
import logging

import bioc
import pandas as pd
import tqdm

from article_utils.db import DbTextMatch, connect, fetch_xml, get_match, get_matches
from article_utils.match_text import match_text_to_document
from article_utils.util import dump_jsonl, read_jsonl

tqdm.tqdm.pandas()


def match_row(row, article_content, db_conn, min_aln_score) -> DbTextMatch:
    if article_content is None:
        return None
    # try to get from db
    prev_match = get_match(db_conn, article_content['id'], row.text)
    if prev_match:
        return prev_match

    # logging.info(f'matching "{row.text[:50]}..." to PMID {row.pmid}')
    collection = bioc.loads(article_content['content'])
    last_err = None
    match_args = {'header_lines': 2}
    if row.get('in_table'):
        match_args['ref_gap_penalty'] = -0.5
        match_args['tgt_gap_penalty'] = -1
    try:
        match = DbTextMatch(
            **match_text_to_document(
                row.text, collection.documents[0], min_aln_score, **match_args
            ),
            content_id=article_content['id'],
        )

        return match
    except ValueError as err:
        if 'too many' in str(err):
            raise err
        last_err = err

    # add to the db\
    err_match = DbTextMatch(
        content_id=article_content['id'],
        query_text=row.text,
        extracted_text=None,
        start_pos=None,
        end_pos=None,
        score=None,
        table_header=None,
        error_message=str(last_err),
    )
    return err_match


def extract_abstract_from_fulltext(fulltext):
    doc = bioc.loads(fulltext).documents[0]
    content = []
    found_abstract = False
    for passage in doc.passages:
        if passage.infons['section'] == 'title':
            content.append(passage.text)
        if passage.infons['section'] == 'abstract':
            found_abstract = True
            content.append(passage.text)
    content = ''.join(content)
    if not found_abstract:
        return ''
    return content


def fetch_abstracts(articles, pmid_list) -> dict[str, str]:
    abstracts = {}
    for pmid in pmid_list:
        if articles[pmid]['content_source'] == 'pubmed':
            abstracts[pmid] = articles[pmid]['content']
        else:
            abstract = extract_abstract_from_fulltext(articles[pmid]['content'])
            abstracts[pmid] = abstract
    return abstracts


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--article_db_file',
        help='path to the sqlite3 file which contains article data',
        default='data/articles_db.sqlite',
    )
    parser.add_argument(
        'input',
        help='path to the JSON file downloaded from hypothesis',
    )
    parser.add_argument(
        '--min_aln_score',
        help='minimum score for alignments of the annotation text back to the full text of the article',
        default=0.8,
        type=float,
    )
    parser.add_argument(
        'output',
        help='path to the output JSONL file with the added XML match data',
    )
    parser.add_argument(
        '--match_output',
        help='path to the JSONL file with the matches to be uploaded',
        required=True,
    )
    args = parser.parse_args()

    df = read_jsonl(args.input, required_columns=['pmid', 'not_in_main_text'])
    print(df.columns.tolist())

    if df.shape[0] == 0:
        # if the input file is empty, re-output an empty file
        dump_jsonl(df, args.output)
        return
    FIX_DUP_DELETIONS = {'22713161': '25998099'}
    df['pmid'] = df.pmid.apply(lambda pmid: FIX_DUP_DELETIONS.get(str(pmid), pmid)).astype(str)

    db_conn = connect(args.article_db_file)
    try:
        articles = fetch_xml(db_conn, df)

        df['_has_content'] = df.pmid.apply(lambda pmid: pmid in articles)

        text_df = df[['text', 'pmid']].drop_duplicates()
        text_df['_has_content'] = text_df.pmid.apply(lambda pmid: pmid in articles)

        text_df = text_df[(text_df.pmid != '')]

        if 'not_in_main_text' in df.columns:
            abstract_pair_df = df[(df.not_in_main_text == True)].copy()

            abstracts = fetch_abstracts(articles, abstract_pair_df.pmid.unique().tolist())
            abstract_pair_df['text'] = abstract_pair_df['pmid'].apply(lambda pmid: abstracts[pmid])
            abstract_pair_df = abstract_pair_df[abstract_pair_df.text != '']
            abstract_pair_df['username'] = 'abstracts'

            df = pd.concat([df, abstract_pair_df]).reset_index(drop=True)

        text_df = df[['text', 'pmid']].drop_duplicates()
        text_df['_has_content'] = text_df.pmid.apply(lambda pmid: pmid in articles)
        text_df = text_df[(text_df.pmid != '') & (text_df.text != '')]

        logging.info(
            f'matching {text_df.shape[0]} text selections across {text_df.pmid.nunique()} articles'
        )
        # split dataframes so that the progress calculation works correctly
        text_df['_match'] = text_df.progress_apply(
            lambda row: match_row(row, articles.get(row.pmid), db_conn, args.min_aln_score),
            axis=1,
        )
        new_matches = text_df['_match'].dropna().tolist()

        for col_name, match_prop_name in {
            'text.start': 'start_pos',
            'text.end': 'end_pos',
            'text.score': 'score',
            'text.extracted': 'extracted_text',
            'text.match_error': 'error_message',
            'text.table_header': 'table_header',
            'db.content_id': 'content_id',
            'text.section': 'section',
        }.items():
            text_df[col_name] = text_df['_match'].str[match_prop_name]

        text_df = text_df.drop(columns=[c for c in text_df.columns if c.startswith('_')])

        text_df['evidence.source'] = text_df.pmid.apply(
            lambda pmid: dict(articles.get(pmid, {})).get('content_source', '')
        )

        df = df.merge(text_df, on=['text', 'pmid'], how='left').rename(
            columns={'text': 'text.query'}
        )
        df['text.extracted'] = df['text.extracted'].fillna('')
        df['text.match_error'] = df['text.match_error'].fillna('')
        df.loc[
            (df['text.extracted'] == '')
            & (df['text.match_error'] == '')
            & (df['_has_content'] == False),
            'text.match_error',
        ] = 'missing article content'
        df.loc[
            (df['text.extracted'] == '')
            & (df['text.match_error'] == '')
            & (df['text.query'] == ''),
            'text.match_error',
        ] = 'page note, nothing to match'
        unexplained_errors = df[(df['text.extracted'] == '') & (df['text.match_error'] == '')]
        if unexplained_errors.shape[0] > 0:
            print(
                df[(df['text.extracted'] == '') & (df['text.match_error'] == '')][
                    ['text.query', 'document.title', 'pmid', 'pmcid', 'db.content_id']
                ]
            )
            raise AssertionError(
                f'Some match errors ({unexplained_errors.shape[0]}) do not have a listed reason'
            )

        df['in_table'] = df.in_table | df['text.table_header'].fillna('') != ''

        dump_jsonl(df, args.output)
        new_matches_df = pd.DataFrame.from_records(new_matches)
        dump_jsonl(new_matches_df, args.match_output)
    finally:
        db_conn.close()


if __name__ == '__main__':
    main()
