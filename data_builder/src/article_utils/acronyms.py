"""
Parse acronyms from full-text context of articles based on pattern matching
"""

import argparse
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

import bioc
import pandas as pd
import requests_cache
import tqdm

from civic.util import load_dump

from .db import backfill_bioc, connect, get_all_fulltext
from .util import dump_jsonl

# often used interchangably in articles for the same use of an acronym
SUBS = {
    'carcinoma': 'cancer',
    'carcinomas': 'cancer',
    'florescence': 'fluorescent',  # mistake in article
    'florescence': 'fluorescent',  # mistake in article
    'florescent': 'fluorescent',
    'stranded': 'strand',
    'indices': 'index',
    'polimerase': 'polymerase',  # mistake in article
    'remission': 'response',
    'microarray': 'micro array',
    'gammapathies': 'gammopathies',
    'gammopaties': 'gammopathies',  # mistake in article
    'gammapathy': 'gammopathy',  # mistake in article
    'progress': 'progression',
    'linda': 'lindau',
}
STOP_WORDS = set(['and', 'or', 'a', 'the', 'to', 'of', 'for', 'not', 'nor'])
WORD_SPLIT_PATTERN = r'(\s+|,|;|\.|\(|\)|{|}|:|-)'
MAX_ACRONYM_LENGTH = 10


def count_nones(list_like):
    return sum(1 if item is None else 0 for item in list_like)


def non_none_apply(list_like, func=max):
    values = [value for value in list_like if value is not None]
    if values:
        return func(values)
    return None


@dataclass
class WordMetadata:
    text: str
    acronym_like: bool
    word_like: bool
    sentence_number: int
    suffix: bool


def bracket_match(text: str):
    """
    Check if some text contains brackets that are not matched
    """
    bracket_stack = []
    bracket_closes = {e: s for s, e in zip('[{(', ']})')}
    for char in text:
        if char in {'(', '[', '{'}:
            bracket_stack.append(char)
        elif char in bracket_closes:
            if not bracket_stack:
                return False
            if bracket_stack[-1] != bracket_closes[char]:
                return False
            bracket_stack.pop()

    if bracket_stack:
        return False
    return True


def looks_like_acronym(word):
    if len(word) > MAX_ACRONYM_LENGTH:
        return False
    return bool(re.match(r'^[a-z]?[a-z]?[A-Z][a-zA-Z]*[-/]?[a-zA-Z]*[A-Z][a-z]?s?$', word))


def merge_complex_acronyms(words: list[WordMetadata]) -> list[WordMetadata]:
    """
    Normally we want to split words on hyphens/slashes but some acronyms include hyphens so we need to rejoin ones that look like they are an acronym
    """
    merged_words = []
    word_index = 0
    while word_index < len(words):
        if len(words) - word_index < 5:
            merged_words.append(words[word_index])
            word_index += 1
            continue
        open_br, word1, hyphen, word2, close_br = words[word_index : word_index + 5]
        acronym = word1.text + hyphen.text + word2.text

        if (
            open_br.text == '('
            and close_br.text == ')'
            and hyphen.text in {'-', '/'}
            and looks_like_acronym(acronym)
        ):
            merged_words.extend(
                [open_br, WordMetadata(acronym, True, True, word1.sentence_number, False), close_br]
            )
            word_index += 5
        else:
            merged_words.append(words[word_index])
            word_index += 1

    return merged_words


def split_words(text) -> list[WordMetadata]:
    """
    Split text into words with their associated metadata/classification
    """
    words = re.split(WORD_SPLIT_PATTERN, text)  # split words
    word_meta: list[WordMetadata] = []  # (acronym, word_like, sentence number)
    sentence_count = 0

    # collect metadata about the words being processed
    for pos, word in enumerate(words):
        word_like = not re.match(WORD_SPLIT_PATTERN, word)
        # if word in stop_words:
        #     word_like = False
        acronym_like = False

        if pos > 0 and pos < len(words) - 1:
            if all(
                [
                    words[pos - 1] == '(',
                    words[pos + 1] == ')',
                    looks_like_acronym(word),
                ]
            ):
                acronym_like = True

        is_suffix = pos > 0 and words[pos - 1] == '-'
        word_meta.append(
            WordMetadata(word, acronym_like, word_like and not is_suffix, sentence_count, is_suffix)
        )

        if word == '.':
            sentence_count += 1

    return word_meta


def mark_acronyms(text, max_intra_word_letters=2) -> dict[str, set[str]]:
    """
    Regex-based method to detect obvious acronym definitions
    """
    words = split_words(text)
    # merge any words inside brackets split only by hyphens
    words = merge_complex_acronyms(words)

    acronyms = {}

    # now match the putative acronyms to the words preceding them
    for acronym_pos, acronym in enumerate(words):
        if not acronym.acronym_like:
            continue
        # simplest case, number of tokens
        letter_index_matches: list[set[Optional[int]]] = [set() for _ in acronym.text]

        for words_pos in range(acronym_pos - 2, -1, -1):
            if words[words_pos].sentence_number != words[acronym_pos].sentence_number:
                # only use words from the same sentence
                break
            words_covered = 0
            for span_pos in range(words_pos, acronym_pos):
                if words[span_pos].word_like and words[span_pos].text not in STOP_WORDS:
                    words_covered += 1
            if words_covered > len(acronym.text) + max_intra_word_letters:
                # more than the max number of words allowed to be skipped in an acronym. do not need to look past here
                break

            for letter_pos, letter in enumerate(acronym.text):
                if words[words_pos].text and words[words_pos].text[0].lower() == letter.lower():
                    letter_index_matches[letter_pos].add(words_pos)

        if not letter_index_matches[0]:
            continue

        # from the word matches create all possible paths
        paths = [[c] for c in letter_index_matches[0] if words[c].word_like]
        for choices in letter_index_matches[1:]:
            new_paths = []
            for path in paths:
                curr_max = non_none_apply(path, max)
                for choice in choices | {None}:
                    if curr_max is None or choice is None or choice > curr_max:
                        new_paths.append(path[:] + [choice])
            paths = new_paths

        filtered_paths = []
        # filter out improbable paths
        for path in paths:
            missed_letters = count_nones(path)
            if missed_letters > max_intra_word_letters:
                continue
            path_start = non_none_apply(path, min)
            path_end = non_none_apply(path, max)
            if not bracket_match(''.join([w.text for w in words[path_start : path_end + 1]])):
                continue
            total_words = sum(m.word_like for m in words[path_start : path_end + 1])

            if total_words <= (len(path) - missed_letters + max_intra_word_letters):
                filtered_paths.append(path)

        if filtered_paths:
            best_path = min(
                filtered_paths,
                key=lambda p: (
                    count_nones(p),
                    1 - non_none_apply(p, max),  # path closer to acronym
                    non_none_apply(p, max) - non_none_apply(p, min),  # shorted path
                ),
            )

            # check if the missing letters are in any of the surrounding words
            for i, (match, letter) in list(enumerate(zip(best_path, acronym.text)))[1:]:
                if match is not None:
                    continue
                prev_choice = best_path[i - 1]

                if prev_choice is None:
                    # cannot interpolate from previous match
                    break

                if words[prev_choice].text.lower() in STOP_WORDS:
                    # never look inside a stop word
                    continue
                # already used the first letter
                if (
                    len(words[prev_choice].text) > 1
                    and letter.lower() in words[prev_choice].text[1:]
                ):
                    best_path[i] = prev_choice

            if count_nones(best_path) == 0:
                # join everything up to just before the acronym itself
                defn = ''.join([w.text for w in words[min(best_path) : acronym_pos - 1]])
                defn = defn.strip()
                acronyms.setdefault(acronym.text, set()).add(defn)
    return acronyms


def convert_to_singular(word):
    subs = [
        (r'\'s', ''),
        ('exes', 'ex'),
        (r'ses', 's'),
        (r'ies', 'y'),
        (r'([^ios])s', r'\1'),
        (r'ated', 'ating'),
        (r'iency', 'ient'),
        (r'ced', 'cing'),
        ('ally', 'al'),
        ('tation', 'tating'),
        ('ence', 'ent'),
        ('ios', 'io'),
    ]

    for patt, rep in subs:
        word = re.sub(patt + r'$', rep, word)
    return word


def normalize_defn(defn):
    # defn = re.sub(r's\b', '', defn.lower().strip())
    defn = defn.lower().strip()
    defn = re.sub(r'(-|=|/)', ' ', defn)
    defn = re.sub(r'[,]', ' ', defn)
    words = defn.split()
    words = [SUBS.get(w, w) for w in words]
    words = [convert_to_singular(w) for w in words]
    words = [w for w in words if w not in STOP_WORDS]
    defn = ''.join(words)
    return defn


def grab_acronyms(doc) -> list[tuple[str, str]]:
    # join text with period to avoid interpreting acronyms across passages
    full_text = '. '.join(passage.text for passage in doc.passages)
    acronyms: dict[str, list[str]] = {a: list(defn) for a, defn in mark_acronyms(full_text).items()}
    result = []
    for acronym in acronyms:
        for defn in acronyms[acronym]:
            # don't include definitions with delimiter characters in them
            if not re.search(r'(\t)', defn):
                result.append((acronym, defn))
    return result


def main():
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'info').upper())
    parser = argparse.ArgumentParser()
    parser.add_argument('--article_db_file', help='path to the sqlite article db file')
    parser.add_argument(
        '--civic_dump',
        required=True,
        help='path to the civic dump to determine the articles to be parsed',
    )
    parser.add_argument('output', help='path to the JSONL file containing the acronym summary')
    parser.add_argument(
        'errors', help='path to the JSONL file containing the acronym finding errors'
    )
    parser.add_argument('--genes', required=True, help='path to the ncbi gene info dump')
    args = parser.parse_args()

    db_conn = connect(args.article_db_file)
    pmids = set(load_dump(args.civic_dump).pmid.dropna().unique().tolist())

    logging.info(f'reading: {args.genes}')
    genes_df = pd.read_csv(args.genes, usecols=['Symbol'], sep='\t')
    genes_df['is_gene'] = True
    genes_df = genes_df.rename(columns={'Symbol': 'acronym'})

    dfs = []
    metadata = {}
    try:
        backfill_bioc(db_conn, pmids, strict=False)
        bioc_articles = get_all_fulltext(db_conn, 'bioc')
        for article in tqdm.tqdm(bioc_articles):
            if article['content_source'] == 'pubmed' and article['author_fulltext']:
                logging.warning(
                    f'abstract-only content despite listed author-uploaded fulltext ({article["author_fulltext"]}) available for pubmed article {article["pmid"]}'
                )
            metadata[article['id']] = article
            collection = bioc.loads(article['content'])
            if len(collection.documents) < 1:
                continue

            acronyms = grab_acronyms(collection.documents[0])
            df = pd.DataFrame(acronyms, columns=['acronym', 'defn'])
            df['pmid'] = article['pmid']
            dfs.append(df)
            with db_conn:
                db_conn.executemany(
                    'INSERT INTO acronyms(acronym, definition, article_id) VALUES (?, ?, ?)',
                    [(a, d, article['id']) for a, d in acronyms],
                )

    finally:
        db_conn.close()

    df = pd.concat(dfs)

    # filter out any acronyms that are gene symbols (don't need to expand them as they should direct match the claim or be an alias of it)
    df = df.merge(genes_df, how='left')
    df = df[df.is_gene != True]

    def predict_acronym(defn):
        words = re.split(r'[\s\-=]+', defn.strip())
        return ''.join([w[0] for w in words if w and w not in STOP_WORDS]).upper()

    df['normalized_defn'] = df.defn.apply(normalize_defn)
    df['predicted_acronym'] = df.defn.apply(predict_acronym)
    df['acronym_matches'] = df.acronym == df.predicted_acronym

    counts_df = (
        df.groupby(['pmid', 'acronym'])
        .agg({'normalized_defn': 'nunique', 'acronym_matches': 'sum'})
        .reset_index()
        .rename(columns={'normalized_defn': 'total_defns', 'acronym_matches': 'total_matches'})
    )
    df = df.merge(counts_df[['pmid', 'total_defns', 'acronym', 'total_matches']])

    counts_df = (
        df.groupby(['pmid', 'acronym', 'normalized_defn'])
        .agg({'defn': 'nunique'})
        .reset_index()
        .rename(columns={'defn': 'normal_score'})
    )
    df = df.merge(counts_df)

    # remove defns where their are conflicts but only one of them is a direct match
    keep_df = pd.concat(
        [
            df[df.total_defns == 1],
            df[(df.total_defns > 1) & (df.total_matches == 1)],
            df[(df.total_defns > 1) & (df.total_matches != 1) & (df.normal_score > 1)],
        ]
    )
    keep_df = keep_df[['pmid', 'acronym', 'defn']]
    keep_df['keep'] = True
    df = df.merge(keep_df, how='left')
    df['keep'] = df['keep'].fillna(False)

    errors_df = df.groupby(['acronym', 'pmid']).agg({'keep': 'sum'}).reset_index()
    errors_df = errors_df[errors_df.keep == 0]
    errors_df = errors_df[['acronym', 'pmid']].merge(df)
    dump_jsonl(errors_df, args.errors)

    df = df[df.keep].drop_duplicates(['pmid', 'acronym'], keep='first')

    dump_jsonl(df[['pmid', 'acronym', 'defn']], args.output)


def load_acronyms_by_pmid(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    cursor = conn.cursor()
    try:
        records = cursor.execute(
            'SELECT * FROM acronyms JOIN article_metadata ON (acronyms.article_id = article_metadata.id)'
        ).fetchall()
    finally:
        cursor.close()

    result: dict[str, dict[str, str]] = {}
    for record in records:
        result.setdefault(record['pmid'], dict())[record['acronym']] = record['definition']
    return result


def has_acronym(text, acronym):
    return bool(re.search(r'(\b' + acronym + r's?\b)', text))


def sub_acronym(text, acronym, defn):
    if defn in text:
        return text
    text = re.sub(r'(\b' + acronym + r's?\b)', f'\\1 ({defn})', text, 1)
    return text


if __name__ == '__main__':
    requests_cache.install_cache('.requests_cache')
    main()
