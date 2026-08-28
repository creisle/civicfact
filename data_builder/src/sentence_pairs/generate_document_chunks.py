import argparse
import logging
import re
from dataclasses import dataclass

import bioc
import pandas as pd
import tqdm

from article_utils.db import connect, fetch_xml
from article_utils.match_text import regex_match_substring
from article_utils.util import dump_jsonl, listify, normalize_and_expand_column
from civic.util import load_dump
from nli.util import Text


@dataclass
class DocumentChunk:
    start: int
    text: str


def expand_pubtator_annotations(df, min_length):
    matches = []
    logging.info('expanding pubtator annotations to entire article using regex matching')
    for _, full_row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        if len(full_row['text.extracted']) < min_length:
            # skip re-matching any super short annotations
            row = {
                k: v for k, v in full_row.items() if not k.startswith('text') and k != 'plaintext'
            }
            row.update(
                {
                    'match.start': full_row['text.start'],
                    'match.end': full_row['text.end'],
                    'match.text': full_row['text.extracted'],
                }
            )
            matches.append(row)
            continue
        original_found = False
        row_matches = regex_match_substring(
            full_row['plaintext'],
            full_row['text.extracted'],
            add_word_boundaries=bool(len(full_row['text.extracted']) < 3),
            case_sensitive=bool(len(full_row['text.extracted']) < 3),
        )

        for i in range(len(row_matches)):
            row = {
                k: v
                for k, v in full_row.items()
                if (not k.startswith('text') and k != 'plaintext') or k.endswith('.score')
            }
            for k, v in row_matches[i].items():
                row[f'match.{k}'] = v
            row_matches[i] = row

        original_found = any(
            [
                m['match.start'] == full_row['text.start']
                and m['match.end'] == full_row['text.end']
                for m in row_matches
            ]
        )

        if not original_found:
            logging.warning(
                f'dropping annotation {repr(full_row["text.extracted"])} (biotype={full_row["pubtator.biotype"]}) '
                + f'from {full_row["pmcid"]}/PMID:{full_row["document.pmid"]}. regex matching ({len(row_matches)}) failed to re-create original annotation: ({int(full_row["text.start"])}-{int(full_row["text.end"])}). '
                + f'{repr(full_row["pubtator.prefix"])} {repr(full_row["pubtator.text"])} {repr(full_row["pubtator.suffix"])}'
            )
        else:
            matches.extend(row_matches)

    match_df = pd.DataFrame.from_records(matches).drop_duplicates(
        ['document.pmid', 'match.start', 'match.text']
    )
    # now recollapse these and merge any identifiers
    logging.info(f'expanded from {df.shape[0]} to {match_df.shape[0]} annotations')
    return match_df


def chunk_document(doc: str, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    chunks = []
    for chunk_i in range(0, len(doc), chunk_size - chunk_overlap):
        chunks.append(dict(start=chunk_i, text=doc[chunk_i : chunk_i + chunk_size]))
    return chunks


def convert_biocxml_to_text(xml_bioc_content):
    col = bioc.loads(xml_bioc_content)
    return ''.join([p.text for p in col.documents[0].passages])


def get_passage_ranges_from_biocxml(xml_bioc_content):
    col = bioc.loads(xml_bioc_content)
    rows = []
    for passage in col.documents[0].passages:
        rows.append(
            {'start': passage.offset, 'end': passage.offset + len(passage.text), **passage.infons}
        )

    return rows


def calculate_overlap(range1: tuple[int, int], range2: tuple[int, int]) -> int:
    overlap = min([range1[1], range2[1]]) - max([range1[0], range2[0]])
    if overlap < 0:
        return 0
    return overlap


def check_is_null(value):
    if not hasattr(value, '__len__') and pd.isnull(value):
        return True
    return False


def compact_records(df):
    records = []
    for record in df.dropna(how='all').to_dict('records'):
        new_record = {}
        for k, v in record.items():
            if not check_is_null(v):
                new_record[k] = v
        records.append(new_record)
    return records


def overlap_chunks_by_range(
    chunks_df,
    df,
    prefix,
    new_col_name,
    min_overlap,
    additional_keep_columns=None,
    max_rows=10000,
    filter_by_percent=False,
):
    if additional_keep_columns is None:
        additional_keep_columns = []

    overlap_dfs = []

    for i in tqdm.tqdm(range(0, df.shape[0], max_rows)):
        overlap_df = chunks_df.merge(
            df[i : min([i + max_rows, df.shape[0]])], on=['document.pmid'], how='inner'
        )
        overlap_df['overlap'] = overlap_df.apply(
            lambda row: calculate_overlap(
                (row['chunk.start'], row['chunk.end']),
                (row[f'{prefix}.start'], row[f'{prefix}.end']),
            ),
            axis=1,
        )
        overlap_df['overlap.percent'] = overlap_df.overlap / (
            overlap_df[f'{prefix}.end'] - overlap_df[f'{prefix}.start']
        )
        if filter_by_percent:
            overlap_df = overlap_df[overlap_df['overlap.percent'] >= min_overlap]
        else:
            overlap_df = overlap_df[overlap_df.overlap >= min_overlap]
        overlap_dfs.append(overlap_df)

    keep_columns = [
        c for c in df.columns if c.startswith(prefix) or c.startswith('overlap')
    ] + additional_keep_columns
    keep_columns = list(set(keep_columns))

    overlap_df = (
        pd.concat(overlap_dfs)
        .groupby(['chunk.start', 'chunk.end', 'document.pmid'])
        .apply(lambda subset_df: compact_records(subset_df[keep_columns]))
        .rename(new_col_name)
        .reset_index()
    )
    chunks_df = chunks_df.merge(overlap_df, how='left')
    return chunks_df


def filter_matching_acronyms(row):
    result = []

    for acronym in listify(row['acronyms']):
        names = [acronym['name']]
        if names[-1].endswith('s'):
            names.append(names[-1][:-1])
        else:
            names.append(names[-1] + 's')
        for name in names:
            if regex_match_substring(
                row['chunk.text'], name, add_word_boundaries=True, case_sensitive=True
            ):
                result.append(acronym)
    return result


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--partitions', required=True)
    parser.add_argument('--civic_dump', default='./data/civic-dump.jsonl')
    parser.add_argument('--article_db_file', default='./data/articles_db.sqlite')
    parser.add_argument('--pubtator_annotations', required=True)
    parser.add_argument('--chunk_size', default=1000, type=int)
    parser.add_argument('--chunk_overlap', default=100, type=int)
    parser.add_argument('--output', required=True, help='path to output document chunks to')
    parser.add_argument(
        '--acronyms', required=True, help='path to the jsonl final containing acronym definitions'
    )
    parser.add_argument('--min_pubtator_overlap', type=float, default=0.95)
    parser.add_argument('--min_selection_overlap', type=float, default=0.75)
    args = parser.parse_args()

    logging.info(f'reading: {args.partitions}')
    part_df = pd.read_csv(args.partitions).rename(columns={'pmid': 'document.pmid'})
    part_df['document.pmid'] = part_df['document.pmid'].astype(str)

    logging.info(f'reading: {args.acronyms}')
    acronyms_df = pd.read_json(args.acronyms, lines=True, dtype=False).rename(
        columns={'pmid': 'document.pmid'}
    )
    acronyms_df = (
        acronyms_df.groupby(['document.pmid'])
        .apply(
            lambda s: s[['acronym', 'defn']].rename(columns={'acronym': 'name'}).to_dict('records')
        )
        .reset_index(name='acronyms')
    )

    # # are there any claims that are more than 1024 characters?
    # logging.info(f'reading: {args.nli_data}')
    # df = pd.read_json(args.nli_data, dtype=False, lines=True)
    # logging.info(f'loaded {df.shape[0]} total rows from the nli data file')
    # df['claim.length'] = df['claim.flat'].str.len()
    # df = df[
    #     (df.generated == False)
    #     & (df['evidence.source'] != 'civic')
    #     & (df['gold_label_name'] != 'NEI')
    # ]
    # logging.info(f'there are {df.shape[0]} manual evidence/claim pairs')

    # selections_df = df.explode('evidence').reset_index()
    # selections_df = normalize_and_expand_column(selections_df, 'evidence')
    # selections_df['evidence.content'] = selections_df['evidence.content'].apply(
    #     lambda content: Text(content=content).concat()
    # )
    # selections_df = (
    #     selections_df.groupby(
    #         ['evidence.meta.start', 'evidence.meta.end', 'document.pmid', 'evidence.content']
    #     )
    #     .agg({'claim.flat': 'unique', 'civic.id': 'unique'})
    #     .reset_index()
    # )

    civic_df = load_dump(args.civic_dump)

    db_conn = connect(args.article_db_file)
    try:
        articles = fetch_xml(db_conn, civic_df, 'pmid', strict=False)
    finally:
        db_conn.close()

    logging.info(f'loaded {len(articles)} articles')
    articles_df = pd.DataFrame.from_records(list(articles.values()))
    articles_df['plaintext'] = articles_df.content.apply(convert_biocxml_to_text)
    articles_df['document.length'] = articles_df.plaintext.str.len()
    articles_df['chunk'] = articles_df.plaintext.apply(
        lambda text: chunk_document(text, args.chunk_size, args.chunk_overlap)
    )
    articles_df = articles_df.rename(columns={'pmid': 'document.pmid'})
    articles_df['document.pmid'] = articles_df['document.pmid'].astype(str)

    articles_df = articles_df.merge(part_df[['document.pmid', 'partition']], how='left')
    if articles_df[articles_df.partition.fillna('') == ''].shape[0] != 0:
        logging.warning(
            f'partition not defined for all articles: {articles_df.partition.isna().sum()} adding to training partition by default'
        )
        articles_df.loc[articles_df.partition.isna(), 'partition'] = 'train'

    chunks_df = (
        articles_df[
            ['document.pmid', 'chunk', 'document.length', 'license', 'content_source', 'partition']
        ]
        .explode('chunk')
        .reset_index(drop=True)
    )
    chunks_df = normalize_and_expand_column(chunks_df, 'chunk')
    chunks_df['chunk.end'] = chunks_df['chunk.start'] + chunks_df['chunk.text'].str.len()
    chunks_df['chunk.id'] = (
        chunks_df['document.pmid'].astype(str)
        + ":"
        + chunks_df['chunk.start'].astype(str)
        + '-'
        + chunks_df['chunk.end'].astype(str)
    )
    logging.info(f'split documents into {chunks_df.shape[0]} chunks')

    chunks_df = chunks_df.merge(acronyms_df, on=['document.pmid'], how='left')
    chunks_df['acronyms'] = chunks_df.apply(filter_matching_acronyms, axis=1)

    passages_df = articles_df[['document.pmid', 'content']].copy()
    passages_df['passage'] = passages_df.content.apply(get_passage_ranges_from_biocxml)
    passages_df = passages_df.explode('passage').reset_index(drop=True)
    passages_df = normalize_and_expand_column(passages_df, 'passage').drop(columns=['content'])

    logging.info(f'overlapping passage metadata with document chunks')
    chunks_df = overlap_chunks_by_range(
        chunks_df,
        passages_df,
        prefix='passage',
        new_col_name='passages',
        min_overlap=args.chunk_overlap,
    )

    # extend pubtator annotations
    logging.info(f'reading: {args.pubtator_annotations}')
    pub_df = pd.read_json(args.pubtator_annotations, lines=True, dtype=False).rename(
        columns={'pmid': 'document.pmid'}
    )
    logging.info(f'loaded {pub_df.shape[0]} pubtator annotations')
    pub_df = pub_df[
        pub_df['text.match_error'].fillna('') == ''
    ]  # ignore elements with matching errors
    logging.info(f'there are {pub_df.shape[0]} pubtator annotations with valid matching')

    pub_df = expand_pubtator_annotations(
        pub_df.merge(articles_df[['plaintext', 'document.pmid']]), min_length=4
    )
    ### now calculate overlap with pubtator annotations

    logging.info(f'calculating pubtator annotations that overlap with document chunks')

    chunks_df = overlap_chunks_by_range(
        chunks_df,
        pub_df,
        prefix='match',
        new_col_name='pubtator',
        min_overlap=args.min_pubtator_overlap,
        filter_by_percent=True,
        additional_keep_columns=[
            c for c in pub_df.columns if c.startswith('pubtator.') or c.startswith('match.')
        ],
    )

    # logging.info('calculating manual evidence selection overlap with document chunks')
    # chunks_df = overlap_chunks_by_range(
    #     chunks_df,
    #     selections_df,
    #     prefix='evidence.meta',
    #     new_col_name='selections',
    #     min_overlap=args.min_selection_overlap,
    #     filter_by_percent=True,
    #     additional_keep_columns=['claim.flat', 'civic.id'],
    # )

    chunks_df['chunk.overlap'] = args.chunk_overlap
    chunks_df['chunk.size'] = args.chunk_size
    print(chunks_df)
    print(chunks_df.dtypes)

    total_chunks = (
        chunks_df.groupby(['document.pmid'])
        .agg({'chunk.id': 'nunique'})
        .reset_index()
        .rename(columns={'chunk.id': 'document.total_chunks'})
    )
    chunks_df = chunks_df.merge(total_chunks)
    chunks_df = chunks_df.rename(columns={'content_source': 'document.content_source'})

    dump_jsonl(chunks_df, args.output)


if __name__ == '__main__':
    main()
