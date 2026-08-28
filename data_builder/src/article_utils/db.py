import argparse
import logging
import os
import sqlite3
from io import StringIO
from typing import Literal, Optional, TypedDict
from xml.etree.ElementTree import ParseError

import bioc
import pandas as pd
import requests_cache
import tqdm
from bioconverters.pmcxml import pmcxml2bioc
from bioconverters.pubmedxml import pubmedxml2bioc

from .elsevier import fetch_by_doi
from .ncbi import (
    fetch_fulltext,
    get_author_xml,
    get_fulltext_license,
    get_pubmed_metadata,
)

Partition = Literal['train', 'test', 'dev', 'unassigned']


class ArticleMetadata(TypedDict):
    id: int
    doi: Optional[str]
    pmid: Optional[str]
    pmcid: Optional[str]
    license: Optional[str]
    author_fulltext: str
    journal: Optional[str]
    pubtype: Optional[str]


class ArticleContent(TypedDict):
    id: int
    article_id: int
    content: str
    content_type: Literal['bioc', 'xml']
    content_source: str


class TextHash(TypedDict):
    id: int
    text: str
    meta: Optional[dict]
    updated_at: int


class AcronymDefinition(ArticleMetadata):
    article_id: int
    acronym: str
    definition: str


class TextMatch(TypedDict):
    start_pos: Optional[int]
    end_pos: Optional[int]
    query_text: str
    extracted_text: Optional[str]
    score: Optional[float]
    table_header: Optional[str]
    section: Optional[str]
    error_message: Optional[str]


class DbTextMatch(TextMatch):
    content_id: int


class PubtatorAnnotation(TypedDict):
    id: int
    text: str
    created_at: int
    updated_at: int
    response: Optional[str]


class PartitionAssignment(TypedDict):
    partition: Partition
    pmid: str


def connect(db_file, read_only=False, **kwargs) -> sqlite3.Connection:
    if read_only:
        db_file = f'file:{db_file}?mode=ro'
        kwargs['uri'] = True
    db_conn = sqlite3.connect(db_file, **kwargs)
    db_conn.row_factory = sqlite3.Row
    return db_conn


def load_metadata_from_api(
    db_conn: sqlite3.Connection, article_ids: list[str], strict: bool = True, missing_only=True
):
    if not article_ids:
        # nothing to do
        return

    cursor = db_conn.cursor()
    try:
        # update ones where they already exist, insert them if they don't
        existing_pmids = {
            r['pmid'] for r in cursor.execute('SELECT pmid from article_metadata').fetchall()
        }
    finally:
        cursor.close()

    if missing_only:
        article_ids = set(article_ids) - existing_pmids

    if not article_ids:
        return

    df = pd.json_normalize(
        get_pubmed_metadata(article_ids, raise_on_error=strict),
        record_path='articleids',
        meta=['uid', 'pubtype', 'source'],
    )
    if not df.shape[0]:
        if not strict:
            # nothing to do, ignorning failed metadata retrieval
            return
        raise ValueError(f'failed to find metadata for artcile ids: {article_ids}')
    df['pubtype'] = df.pubtype.apply(lambda pubtype: ';'.join(pubtype))
    df = (
        df.drop_duplicates()
        .pivot(index=['uid', 'source', 'pubtype'], columns='idtype', values='value')
        .reset_index()
    )

    # gets called pmid/pmcid when you fetch from pmc and pubmed/pmc when you fetch from pubmed
    if 'pmid' in df.columns and 'pubmed' not in df.columns:
        df = df.rename(columns={'pmid': 'pubmed'})
    if 'pmcid' in df.columns and 'pmc' not in df.columns:
        df = df.rename(columns={'pmcid': 'pmc'})
    for col in ['doi', 'pmc', 'pubmed']:
        if col not in df.columns:
            df[col] = ''
    df = df.fillna('')[['doi', 'pmc', 'pubmed', 'source', 'pubtype']].drop_duplicates(['pubmed'])
    data = []

    for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='fetching licensing data'):
        license = ''
        if row.pmc:
            try:
                license = get_fulltext_license(row.pmc)
                if license == '':
                    raise NotImplementedError(
                        f'did not expect an OA article to ever return empty license data: {row.pmc}. Violates some assumptions'
                    )
            except ValueError:
                license = ''
        data.append(
            {
                'doi': row.doi,
                'pmid': row.pubmed,
                'pmcid': row.pmc,
                'license': license,
                'journal': row.source,
                'pubtype': row.pubtype,
            }
        )

    cursor = db_conn.cursor()
    try:
        # author fulltext information is from separate files
        cursor.executemany(
            'INSERT INTO article_metadata(pmid, doi, pmcid, license, pubtype, journal) VALUES (:pmid, :doi, :pmcid, :license, :pubtype, :journal)',
            [d for d in data if d['pmid'] not in existing_pmids],
        )
        cursor.executemany(
            'UPDATE article_metadata set doi=:doi, pmcid=:pmcid, license=:license, pubtype=:pubtype, journal=:source WHERE pmid=:pmid',
            [d for d in data if d['pmid'] in existing_pmids],
        )
        db_conn.commit()
    finally:
        cursor.close()


def get_article_metadata(
    db_conn: sqlite3.Connection, article_id: str, id_type: str = 'pmc'
) -> ArticleMetadata:
    result = []
    article_id = str(article_id)  # force data type
    assert article_id
    if id_type == 'pmc':
        result = db_conn.execute(
            'SELECT * FROM article_metadata WHERE pmcid=:article_id', {'article_id': article_id}
        ).fetchall()
    elif id_type in {'pubmed', 'pmid'}:
        result = db_conn.execute(
            'SELECT * FROM article_metadata WHERE pmid=:article_id', {'article_id': article_id}
        ).fetchall()

        if not result:
            logging.info(f'missing metadata for PMID:{article_id}. Fetching from API')
            # try to fetch from API
            load_metadata_from_api(db_conn, [article_id])

            result = db_conn.execute(
                'SELECT * FROM article_metadata WHERE pmid=:article_id', {'article_id': article_id}
            ).fetchall()
    elif id_type == 'doi':
        result = db_conn.execute(
            'SELECT * FROM article_metadata WHERE doi=:article_id', {'article_id': article_id}
        ).fetchall()
    else:
        raise NotImplementedError(f'unsupported article id type {id_type}')

    if not result:
        raise KeyError(f'article not found in metadata table (id={article_id}, id_type={id_type})')
    elif len(result) > 1:
        raise ValueError(
            f'article in metadata table (id={article_id}, id_type={id_type}) is not unique [{len(result)}]'
        )
    return dict(result[0])


def batch_get_article_metadata(
    db_conn: sqlite3.Connection,
    article_ids: Optional[list[str]] = None,
    id_type: Literal['pmc', 'pubmed', 'pmid'] = 'pmc',
    strict=False,
) -> list[ArticleMetadata]:
    result = []

    if article_ids is None:
        # select all articles
        result.extend(db_conn.execute('SELECT * FROM article_metadata').fetchall())
        return [dict(r) for r in result]

    batch_size = 999
    index = 0
    result = []
    while index < len(article_ids):
        batch = article_ids[index : index + batch_size]
        id_list_placeholder = ', '.join('?' for _ in batch)
        if id_type == 'pmc':
            result.extend(
                db_conn.execute(
                    f'SELECT * FROM article_metadata WHERE pmcid IN ({id_list_placeholder})',
                    batch,
                ).fetchall()
            )
        elif id_type in {'pubmed', 'pmid'}:
            result.extend(
                db_conn.execute(
                    f'SELECT * FROM article_metadata WHERE pmid IN ({id_list_placeholder})',
                    batch,
                ).fetchall()
            )
        elif id_type == 'doi':
            result.extend(
                db_conn.execute(
                    f'SELECT * FROM article_metadata WHERE doi IN ({id_list_placeholder})',
                    batch,
                ).fetchall()
            )
        else:
            raise NotImplementedError(f'unsupported article id type {id_type}')
        index += batch_size

    if len(result) != len(article_ids) and strict:
        raise KeyError(
            f'Unexpected number of article(s)({len(result)}) found/not found in metadata table (ids={article_ids}, id_type={id_type}). expected ({len(article_ids)}'
        )
    return [dict(r) for r in result]


def _get_fulltext_xml(
    metadata, ignore_oa_errors: bool = True
) -> tuple[str, Literal['pubmed', 'pmc', 'elsevier', '', 'pmc_author_fulltext']]:
    if metadata['pmcid']:
        if not metadata['license']:
            # fetch from PMCID
            try:
                # can only use directly from PMC if it has a license, it is doesn't can still use author-provided fulltext where available
                xml = get_author_xml(metadata['pmcid'])
                return xml, 'pmc_author_fulltext'
            except KeyError as err:
                if not ignore_oa_errors:
                    raise err
            except ValueError:
                pass  # happens when PMC returns just an abstract
        else:
            # fetch from PMCID
            try:
                xml, source = fetch_fulltext(metadata['pmcid'], 'pmc')
                return xml, source
            except KeyError as err:
                if not ignore_oa_errors:
                    raise err
            except ValueError:
                pass  # happens when PMC returns just an abstract
    if metadata['doi'] and not metadata['pmcid']:
        xml = fetch_by_doi(metadata['doi'])
        if xml:
            return xml, 'elsevier'
    xml, source = fetch_fulltext(metadata['pmid'], 'pubmed')
    if xml:
        return xml, source
    return '', ''


class MetaArticleContent(ArticleMetadata, ArticleContent):
    pass


def get_fulltext(db_cursor: sqlite3.Cursor, db_article_id: int) -> MetaArticleContent:
    return [
        dict(r)
        for r in db_cursor.execute(
            '''SELECT article_content.*,
                article_metadata.pmid,
                article_metadata.pmcid,
                article_metadata.doi,
                article_metadata.license,
                article_metadata.author_fulltext
            FROM article_content
            JOIN article_metadata on article_content.article_id = article_metadata.id
            WHERE article_id=:article_id
            ''',
            {'article_id': db_article_id},
        ).fetchall()
    ]


def get_all_fulltext(
    db_cursor: sqlite3.Cursor, content_type: Literal['xml', 'bioc']
) -> MetaArticleContent:
    return [
        dict(r)
        for r in db_cursor.execute(
            '''SELECT article_content.*,
                article_metadata.pmid,
                article_metadata.pmcid,
                article_metadata.doi,
                article_metadata.license,
                article_metadata.author_fulltext
            FROM article_content
            JOIN article_metadata on article_content.article_id = article_metadata.id
            WHERE article_content.content_type = :content_type
            ''',
            {'content_type': content_type},
        ).fetchall()
    ]


def add_article_content(
    db_conn: sqlite3.Connection, db_article_id: int, content: str, source_db: str, content_type: str
):
    cursor = db_conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO article_content (article_id, content, content_source, content_type) values (?, ?, ?, ?)",
            (db_article_id, content, source_db, content_type),
        )
        db_conn.commit()
    finally:
        cursor.close()


def get_xml(
    db_conn: sqlite3.Connection,
    article_id: str,
    id_type: str = 'pmc',
    ignore_oa_errors: bool = True,
) -> list[ArticleContent]:
    cursor = db_conn.cursor()
    try:
        article_metadata = get_article_metadata(db_conn, article_id, id_type)
        # check if already exists
        content = get_fulltext(cursor, article_metadata['id'])
        content = [c for c in content if c['content_type'] == 'xml']

        if content:
            return content

        xml, xml_source = _get_fulltext_xml(article_metadata, ignore_oa_errors=ignore_oa_errors)
        add_article_content(db_conn, article_metadata['id'], xml, xml_source, 'xml')
        return [
            c for c in get_fulltext(cursor, article_metadata['id']) if c['content_type'] == 'xml'
        ]
    finally:
        cursor.close()


def convert_xml_to_bioc(xml, xml_source):
    col = bioc.BioCCollection()
    if xml_source.startswith('pmc'):
        for doc in pmcxml2bioc(
            StringIO(xml),
            all_xml_path_infon=False,
            mark_citations=False,
            trim_sentences=False,
            normalize_tables=False,
        ):
            col.add_document(doc)
    elif xml_source == 'pubmed':
        for doc in pubmedxml2bioc(StringIO(xml), trim_sentences=False):
            col.add_document(doc)
    else:
        raise NotImplementedError(f'unable to convert to bioc for source type ({xml_source})')
    return bioc.dumps(col)


def get_bioc(
    db_conn: sqlite3.Connection,
    article_id: str,
    id_type: Literal['pmc', 'pubmed', 'doi'] = 'pmc',
    ignore_oa_errors=True,
) -> list[MetaArticleContent]:
    cursor = db_conn.cursor()
    try:
        article_metadata = get_article_metadata(db_conn, article_id, id_type)
        fulltext = get_fulltext(cursor, article_metadata['id'])
        xml_ft = [doc for doc in fulltext if doc['content_type'] == 'xml']
        bioc_ft = [doc for doc in fulltext if doc['content_type'] == 'bioc']

        if bioc_ft:
            return bioc_ft

        if not xml_ft:
            xml, xml_source = _get_fulltext_xml(article_metadata, ignore_oa_errors=ignore_oa_errors)
            add_article_content(db_conn, article_metadata['id'], xml, xml_source, 'xml')
        else:
            xml = xml_ft[0]['content']
            xml_source = xml_ft[0]['content_source']

        if not bioc_ft:
            try:
                bioc_text = convert_xml_to_bioc(xml, xml_source)
            except ParseError as err:
                logging.error(f'error parsing {id_type}:{article_id}: {err}')
                raise err
            add_article_content(db_conn, article_metadata['id'], bioc_text, xml_source, 'bioc')

        fulltext = get_fulltext(cursor, article_metadata['id'])
        return [f for f in fulltext if f['content_type'] == 'bioc']
    finally:
        cursor.close()


def check_match(text_match: DbTextMatch) -> DbTextMatch:
    value = {  # defaults
        'error_message': '',
        'section': '',
        'start_pos': None,
        'end_pos': None,
        'score': None,
        'extracted_text': '',
        'table_header': '',
    }
    value.update(text_match)

    for col in ['start_pos', 'end_pos']:
        assert (
            value[col] is None or isinstance(value[col], int) or isinstance(value[col], float)
        ), f'{col} ({repr(value[col])}, {type(value[col])}) must be numeric'

    assert value["score"] is None or isinstance(
        value["score"], float
    ), f'score ({repr(value["score"])}) must be numeric'

    return value


def add_matches(db_conn: sqlite3.Connection, text_matches: list[DbTextMatch]):
    cursor = db_conn.cursor()

    column_order = [
        'query_text',
        'extracted_text',
        'start_pos',
        'end_pos',
        'content_id',
        'score',
        'table_header',
        'error_message',
        'section',
    ]
    # values = []
    # for text_match in text_matches:
    #     text_match = check_match(text_match)
    #     values.append((text_match[c] for c in column_order))
    try:
        db_conn.executemany(
            f'INSERT INTO text_matches({", ".join(column_order)}) VALUES ({", ".join([":" + c for c in column_order])})',
            [check_match(m) for m in text_matches],
        )
        db_conn.commit()
    finally:
        cursor.close()


def add_match(db_conn: sqlite3.Connection, text_match: DbTextMatch):
    add_matches(db_conn, [text_match])


def cleanup_broken_links(db_conn, table, attr, foreign_table, foreign_attr):
    cursor = db_conn.cursor()
    try:
        cursor.execute(
            f'''DELETE FROM {table} WHERE NOT EXISTS (SELECT * FROM {foreign_table} WHERE {foreign_attr} = {table}.{attr})''',
        )
        db_conn.commit()
    finally:
        cursor.close()


def delete_article(db_conn: sqlite3.Connection, article_id: str, **kwargs):
    metadata = get_article_metadata(db_conn, article_id, **kwargs)
    cursor = db_conn.cursor()
    try:
        cursor.execute(
            '''DELETE FROM article_metadata WHERE id = :id''',
            {'id': metadata["id"]},
        )
        db_conn.commit()
    finally:
        cursor.close()
    cleanup_broken_links(db_conn, 'article_content', 'article_id', 'article_metadata', 'id')
    cleanup_broken_links(db_conn, 'acronyms', 'article_id', 'article_metadata', 'id')
    cleanup_broken_links(db_conn, 'text_matches', 'content_id', 'article_content', 'id')


def delete_article_content(db_conn: sqlite3.Connection, content_id: str):
    cursor = db_conn.cursor()
    try:
        cursor.execute(
            '''DELETE FROM article_content WHERE id = :id''',
            {'id': content_id},
        )
        db_conn.commit()
    finally:
        cursor.close()
    cleanup_broken_links(db_conn, 'acronyms', 'article_id', 'article_metadata', 'id')
    cleanup_broken_links(db_conn, 'text_matches', 'content_id', 'article_content', 'id')


def delete_parsed_articles(db_conn: sqlite3.Connection):
    cursor = db_conn.cursor()
    try:
        cursor.execute(
            '''DELETE FROM article_content WHERE content_type = :content_type''',
            {'content_type': 'bioc'},
        )
        db_conn.commit()
    finally:
        cursor.close()
    cleanup_broken_links(db_conn, 'acronyms', 'article_id', 'article_metadata', 'id')
    cleanup_broken_links(db_conn, 'text_matches', 'content_id', 'article_content', 'id')


class MetaTextMatch(ArticleContent, ArticleMetadata, DbTextMatch):
    pass


def get_matches(db_conn: sqlite3.Connection) -> list[MetaTextMatch]:
    cursor = db_conn.cursor()
    try:
        acronyms = cursor.execute(
            '''SELECT *
            FROM text_matches
            JOIN article_content ON (text_matches.content_id = article_content.id)
            JOIN article_metadata ON (article_content.article_id = article_metadata.id)
            '''
        ).fetchall()
        return [dict(r) for r in acronyms]
    finally:
        cursor.close()


def get_match(db_conn: sqlite3.Connection, content_id: int, query_text: str) -> MetaTextMatch:
    cursor = db_conn.cursor()
    try:
        result = cursor.execute(
            '''SELECT *
            FROM text_matches
            JOIN article_content ON (text_matches.content_id = article_content.id)
            JOIN article_metadata ON (article_content.article_id = article_metadata.id)
            WHERE text_matches.content_id = :content_id AND text_matches.query_text = :query_text
            ''',
            {'query_text': query_text, 'content_id': content_id},
        ).fetchone()
        if result is None:
            return None
        return dict(result)
    finally:
        cursor.close()


def get_acronyms(db_conn: sqlite3.Connection) -> list[AcronymDefinition]:
    cursor = db_conn.cursor()
    try:
        acronyms = cursor.execute(
            'SELECT * FROM acronyms JOIN article_metadata ON (acronyms.article_id = article_metadata.id)'
        ).fetchall()
        return acronyms
    finally:
        cursor.close()


def backfill_bioc(db_conn: sqlite3.Connection, pubmed_ids: set[str], **kwargs):
    metadata_pmids = {
        r['pmid'] for r in db_conn.execute('SELECT * FROM article_metadata').fetchall()
    }
    if pubmed_ids - metadata_pmids:
        load_metadata_from_api(db_conn, pubmed_ids - metadata_pmids - {''}, **kwargs)
    articles_missing_xml = db_conn.execute(
        '''SELECT * FROM article_metadata
        LEFT JOIN article_content ON (article_content.article_id = article_metadata.id)
        WHERE content IS NULL''',
    ).fetchall()
    selected_articles = [a for a in articles_missing_xml if a['pmid'] in pubmed_ids]
    logging.info(
        f'{len(selected_articles)} selected articles do not have BIOC full text or abstract versions'
    )
    for article in tqdm.tqdm(selected_articles):
        get_bioc(db_conn, article['pmid'], id_type='pubmed')


def fetch_xml(
    db_conn: sqlite3.Connection, annotations_df, article_id_col='pmid', strict: bool = True
) -> dict[str, MetaArticleContent]:
    articles: dict[str, MetaArticleContent] = {}
    pubmed_ids = (
        annotations_df[annotations_df[article_id_col] != ''][article_id_col].unique().tolist()
    )
    logging.info(f'fetching xml records for {len(pubmed_ids)} articles')
    for pmid in tqdm.tqdm(pubmed_ids):
        logging.debug(f'fetching xml data for {pmid}')
        try:
            article = get_bioc(db_conn, pmid, 'pubmed')[0]
            articles[str(pmid)] = article
        except (KeyError, AssertionError, ParseError, ValueError) as err:
            logging.error(err)
            if strict:
                raise err
    return articles


def get_or_create_textid(
    db_conn: sqlite3.Connection, text: str, create_ok: bool = True, metadata: Optional[str] = None
) -> int:
    cursor = db_conn.cursor()
    assert text.strip()

    try:
        record = cursor.execute(
            'SELECT id FROM text_hashing WHERE text = :text', {'text': text}
        ).fetchone()
        if record:
            return record[0]
        if not create_ok:
            raise ValueError(f'hash record not found: {text}')

        cursor.execute(
            'INSERT INTO text_hashing(text, meta) values (:text, :meta)',
            {'text': text, 'meta': metadata},
        )
        db_conn.commit()
        return cursor.lastrowid
    finally:
        cursor.close()


def upsert_texthash(
    db_conn: sqlite3.Connection, text: str, meta: Optional[str] = None, update_ok: bool = True
) -> TextHash:
    cursor = db_conn.cursor()
    assert text.strip()

    try:
        record = cursor.execute(
            'SELECT * FROM text_hashing WHERE text = :text', {'text': text}
        ).fetchone()
        if record and not update_ok:
            return TextHash(record)
        elif record:
            # update existing meta object and timestamp
            cursor.execute(
                'UPDATE text_hashing SET meta = :meta, last_updated = CURRENT_TIMESTAMP WHERE text = :text ',
                {
                    'text': text,
                    'meta': meta,
                },
            )

        else:
            # insert new record
            cursor.execute(
                'INSERT INTO text_hashing(text, meta, last_updated) values (:text, :meta, CURRENT_TIMESTAMP)',
                {
                    'text': text,
                    'meta': meta,
                },
            )

        db_conn.commit()
        record = cursor.execute(
            'SELECT * FROM text_hashing WHERE text = :text', {'text': text}
        ).fetchone()
        return TextHash(record)
    finally:
        cursor.close()


def get_textids(db_conn: sqlite3.Connection) -> list[TextHash]:
    cursor = db_conn.cursor()

    try:
        records = cursor.execute('SELECT * FROM text_hashing ').fetchall()
        return [dict(r) for r in records]
    finally:
        cursor.close()


def set_partition_assignment(db_conn: sqlite3.Connection, partition: Partition, pmid: str) -> None:
    cursor = db_conn.cursor()
    assert partition.strip()
    assert pmid.strip()

    try:
        cursor.execute(
            'INSERT INTO partition_assignment(partition, pmid) values (:partition, :pmid)',
            {'partition': partition, 'pmid': pmid},
        )
        db_conn.commit()
    finally:
        cursor.close()


def set_partition_assignments(
    db_conn: sqlite3.Connection, assignments: list[PartitionAssignment]
) -> None:
    cursor = db_conn.cursor()

    try:
        cursor.executemany(
            'INSERT INTO partition_assignment(partition, pmid) values (:partition, :pmid)',
            assignments,
        )
        db_conn.commit()
    finally:
        cursor.close()


def get_partition_assignments(db_conn: sqlite3.Connection) -> list[PartitionAssignment]:
    cursor = db_conn.cursor()

    try:
        records = cursor.execute('SELECT * FROM partition_assignment').fetchall()
        return [dict(r) for r in records]
    finally:
        cursor.close()


def upsert_partition_assignment(
    db_conn: sqlite3.Connection, pmid: str, partition: Partition
) -> PartitionAssignment:
    cursor = db_conn.cursor()

    try:
        record = cursor.execute(
            'SELECT * FROM partition_assignment WHERE pmid = :pmid', {'pmid': pmid}
        ).fetchone()
        if not record:
            set_partition_assignment(db_conn, partition, pmid)
        else:
            record = dict(record)
            if record['partition'] != partition:
                cursor.execute(
                    'UPDATE partition_assignment SET partition = :partition WHERE pmid = :pmid',
                    {'pmid': pmid, 'partition': partition},
                )
                db_conn.commit()

    finally:
        cursor.close()


def upsert_pubtator_annotation(db_conn: sqlite3.Connection, annotation: PubtatorAnnotation):
    cursor = db_conn.cursor()
    assert annotation['text']

    try:
        record = cursor.execute(
            'SELECT * FROM pubtator_annotations WHERE text = :text', annotation
        ).fetchone()
        if not record:

            assert annotation['session_id']
            if annotation.get('response'):
                cursor.execute(
                    'INSERT INTO pubtator_annotations(text, session_id, response) values (:text, :session_id, :response)',
                    annotation,
                )
            else:
                cursor.execute(
                    'INSERT INTO pubtator_annotations(text, session_id) values (:text, :session_id)',
                    annotation,
                )
        else:
            input = {**annotation}
            input['id'] = record['id']
            cursor.execute(
                'UPDATE pubtator_annotations SET response = :response, updated_at = current_timestamp where id = :id',
                input,
            )
        db_conn.commit()
    finally:
        cursor.close()


def get_pubtator_annotations(db_conn: sqlite3.Connection) -> list[PubtatorAnnotation]:
    cursor = db_conn.cursor()

    try:
        records = cursor.execute('SELECT * FROM pubtator_annotations').fetchall()
        return [dict(r) for r in records]
    finally:
        cursor.close()


if __name__ == '__main__':
    # globally cache requests
    requests_cache.install_cache(
        '.requests_cache',
        urls_expire_after={
            'https://api.hypothes.is/api*': 5 * 60,  # 5m
            'https://civicdb.org/api/*': 60 * 60,  # 1 hour
            '*': 30 * 60 * 60 * 24,  # 1 month,
            'https://eutils.ncbi.nlm.nih.gov/entrez': 10 * 60 * 60 * 24,  # 10 days,
        },
    )
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'info').upper())
    parser = argparse.ArgumentParser()
    parser.add_argument('db_file')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_file)
    try:
        with conn:
            logging.info('creating the schema tables')
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS article_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doi VARCHAR,
                    pmid VARCHAR,
                    pmcid VARCHAR,
                    license VARCHAR,
                    pubtype VARCHAR,
                    journal VARCHAR,
                    author_fulltext VARCHAR DEFAULT '' NOT NULL
                )'''
            )
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS article_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER, content VARCHAR,
                    content_source VARCHAR,
                    content_type VARCHAR,
                    FOREIGN KEY(article_id) REFERENCES article_metadata(id)
                )'''
            )
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS acronyms (
                    article_id INTEGER,
                    acronym VARCHAR NOT NULL,
                    definition VARCHAR NOT NULL,
                    FOREIGN KEY(article_id) REFERENCES article_metadata(id)
                )'''
            )
            conn.execute(
                '''\
                CREATE TABLE IF NOT EXISTS text_matches (
                    query_text VARCHAR NOT NULL,
                    extracted_text VARCHAR,
                    start_pos INTEGER,
                    end_pos INTEGER,
                    content_id INTEGER NOT NULL,
                    score REAL,
                    table_header VARCHAR,
                    error_message VARCHAR,
                    section VARCHAR,
                    match_source VARCHAR,
                    FOREIGN KEY(content_id) REFERENCES article_content(id)
                )'''
            )
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS partition_assignment (
                    partition VARCHAR NOT NULL,
                    pmid VARCHAR PRIMARY KEY
                )'''
            )
            conn.execute(
                '''\
                CREATE TABLE IF NOT EXISTS text_hashing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text VARCHAR NOT NULL UNIQUE,
                    meta VARCHAR,
                    last_updated datetime default current_timestamp
                )'''
            )
            conn.execute(
                '''\
                CREATE TABLE IF NOT EXISTS pubtator_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text VARCHAR NOT NULL UNIQUE,
                    session_id VARCHAR NOT NULL,
                    created_at datetime DEFAULT current_timestamp,
                    response VARCHAR,
                    updated_at datetime DEFAULT current_timestamp
                )'''
            )
            # conn.execute(
            #     'CREATE TABLE pubtator_biocxml (article_id INTEGER, content VARCHAR NOT NULL, content_source VARCHAR, FOREIGN KEY(article_id) REFERENCES article_metadata(id))'
            # )
            # conn.execute(
            #     'CREATE TABLE pubtator_entities (article_id INTEGER, entity_type VARCHAR NOT NULL, entity VARCHAR NOT NULL, frequency INTEGER, FOREIGN KEY(article_id) REFERENCES article_metadata(id)) '
            # )

        # df = fetch_id_mapping().merge(fetch_license_data(), how='left').fillna('')
        # data = []
        # for i, row in df.iterrows():
        #     data.append([row.doi, row.pmid, row.pmcid, row.license])

        # with conn:
        #     logging.info(f'importing the metadata ({len(data)}')
        #     # load the initial metadata

        #     conn.executemany(
        #         'INSERT INTO article_metadata(doi, pmid, pmcid, license) VALUES (?, ?, ?, ?)', data
        #     )

        with conn:
            for table, field in [
                ('article_content', 'article_id'),
                ('acronyms', 'article_id'),
                ('text_matches', 'content_id'),
                ('article_metadata', 'pmcid'),
                ('article_metadata', 'doi'),
                ('acronyms', 'acronym'),
                ('pubtator_annotations', 'text'),
            ]:
                conn.execute(f'CREATE INDEX idx_{table}_{field} ON {table}({field})')

            for table, fields in [
                ('article_metadata', ['pmid']),
                ('text_matches', ['content_id', 'query_text']),
            ]:
                conn.execute(
                    f'CREATE UNIQUE INDEX idx_uq_{table}_{"_".join(fields)} ON {table}({", ".join(fields)})'
                )
    finally:
        conn.close()

    with open(args.db_file + '.INIT', 'w') as fh:
        fh.write('')
