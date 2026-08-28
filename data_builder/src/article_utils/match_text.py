import argparse
import json
import logging
import os
import re
from functools import lru_cache
from typing import TypedDict

import bioc
import pandas as pd
from rapidfuzz import fuzz

from hypothesis.util import is_junk

from .align import find_in_text, needleman_wunsch_local_match, range_overlap, word_align
from .db import TextMatch, add_matches, connect
from .util import dump_records


class ParsedAnnotation(TypedDict):
    pmid: str
    pmcid: str
    comment: str
    claim_id: str
    example: str
    id: str
    text: str
    text_start: float
    updated: str
    uri: str
    username: str
    label: str


def span_overlap(span1: tuple[int, int], span2: tuple[int, int]) -> bool:
    if span1[1] < span2[0] or span1[0] > span2[1]:
        return False
    return True


def fetch_table_header(text, text_start, document: bioc.BioCDocument, keep_lines=2) -> str:
    for passage in document.passages:
        if not passage.infons.get('xml_path', '').endswith('table'):
            continue

        passage_loc = (passage.offset, passage.offset + len(passage.text))
        if span_overlap((text_start, text_start + len(text)), passage_loc):
            text = ''.join(re.split(r'(\n)', passage.text)[0 : keep_lines * 2 - 1])
            return json.dumps(
                {
                    'start': passage_loc[0],
                    'text': text,
                    'end': passage_loc[0] + len(text),
                    'keep_lines': keep_lines,
                }
            )
    return ''


def get_span_sections(document: bioc.BioCDocument, start_pos: int, end_pos: int) -> list[str]:
    section_names = set()

    for passage in document.passages:
        passage_start = passage.offset
        passage_end = passage_start + len(passage.text)

        if not range_overlap((start_pos, end_pos), (passage_start, passage_end)):
            continue
        if 'section' in passage.infons:
            if passage.infons.get('subsection'):
                section_names.add(passage.infons['section'] + '//' + passage.infons['subsection'])
            else:
                section_names.add(passage.infons['section'])

    return sorted(section_names)


def normalize_whitespace(text):
    return re.sub(r'(\s|\n)', ' ', text)


@lru_cache(maxsize=4096)
def _compile_substring_pattern(substring: str, add_word_boundaries: bool, case_sensitive: bool):
    flags = re.MULTILINE
    if not case_sensitive:
        flags = flags | re.IGNORECASE
    patt = re.escape(substring).replace('\\ ', '\\s+')
    if add_word_boundaries:
        if not re.match(r'\W', substring[0]):
            patt = r'\b' + patt
        if not re.match(r'\W', substring[-1]):
            patt = patt + r'\b'

    return re.compile(patt, flags=flags)


def regex_match_substring(
    text: str,
    substring: str,
    max_boundary_score: int = 1,
    add_word_boundaries=False,
    case_sensitive=False,
) -> list[dict]:
    """
    escape a substring to use regex to find all instances of it in some text input
    """
    patt = _compile_substring_pattern(
        substring, add_word_boundaries=add_word_boundaries, case_sensitive=case_sensitive
    )
    result = []

    for match_obj in patt.finditer(text):
        boundary_score = 0
        if match_obj.start() > 0 and not re.match(r'(\W|_)', text[match_obj.start() - 1]):
            boundary_score += 1
        if match_obj.end() < len(text) and not re.match(r'(\W|_)', text[match_obj.end()]):
            boundary_score += 1
        if boundary_score > max_boundary_score:
            continue
        result.append(
            {
                'start': match_obj.start(),
                'end': match_obj.end(),
                'text': match_obj.group(0),
            }
        )
    return result


def string_match(reference_text: str, query_text: str):
    start_index, end_index = find_in_text(reference_text, query_text)
    extracted_text = reference_text[start_index:end_index]
    return TextMatch(
        start_pos=start_index,
        end_pos=end_index,
        score=None,
        extracted_text=extracted_text,
        table_header='',
        section='',
        query_text=query_text,
        error_message='',
    )


def match_text_to_text(
    reference_text: str, query_text: str, min_aln_score: float, align_by_word=True, **kwargs
) -> TextMatch:
    score = None
    extracted_text = None
    kwargs.setdefault('gap_exception', is_junk)
    # fallback because it is significantly slower
    if align_by_word:
        start_index, end_index, score = word_align(reference_text, query_text, **kwargs)
    else:
        start_index, end_index, score = needleman_wunsch_local_match(
            reference_text, query_text, **kwargs
        )

    if score >= min_aln_score:
        extracted_text = reference_text[start_index:end_index]
    elif score == 0:
        raise ValueError('failed to extract text')
    else:
        error_msg = f'failed to extract text, max alignment score: {score} less than required threshold {min_aln_score}.'
        if score > 0:
            error_msg += f' Closest Match was: {repr(reference_text[start_index:end_index])}'
        raise ValueError(error_msg)

    if not extracted_text:
        raise ValueError('The Extracted text is empty')

    return TextMatch(
        start_pos=start_index,
        end_pos=end_index,
        score=score,
        extracted_text=extracted_text,
        table_header='',
        section='',
        query_text=query_text,
        error_message='',
    )


def rapid_match(ref_text, query_text, min_aln_score: float, **kwargs) -> TextMatch:

    aln = fuzz.partial_ratio_alignment(
        ref_text,
        query_text,
        score_cutoff=50,  # permissive since we realign anyway
    )
    if not aln:
        raise ValueError('failed to match using the rapid method')
    # add some window to either side since this matching method is not super accurate
    ref_start = max([aln.src_start - len(query_text) // 2, 0])
    ref_end = min([aln.src_end + len(query_text) // 2, len(ref_text)])

    # now realign with the slower method on the much smaller portion of text
    subsection = ref_text[ref_start:ref_end]
    short_match = match_text_to_text(subsection, query_text, min_aln_score=min_aln_score, **kwargs)
    short_match['start_pos'] += ref_start
    short_match['end_pos'] += ref_start
    # rescore the alignment
    return short_match


def exhaustive_match(
    reference_text: str, query_text: str, min_aln_score: float, **kwargs
) -> TextMatch:
    """
    Try all methods of text matching, use the fastest ones first to improve efficiency
    """
    errors = []
    match = None

    norm_query_text = normalize_whitespace(query_text)
    norm_reference_text = normalize_whitespace(reference_text)

    for func_i, func in enumerate(
        [
            lambda: string_match(norm_reference_text, norm_query_text),
            lambda: rapid_match(
                norm_reference_text, norm_query_text, min_aln_score, align_by_word=True, **kwargs
            ),
            lambda: rapid_match(
                norm_reference_text, norm_query_text, min_aln_score, align_by_word=False, **kwargs
            ),
        ]
    ):
        try:
            match = func()
        except Exception as err:
            errors.append(str(err))

        if match:
            # fix extracted text and query text to match non-normalized text
            match['extracted_text'] = reference_text[match['start_pos'] : match['end_pos']]
            match['query_text'] = query_text
            return match
    raise ValueError(';'.join(set(errors)))


def match_text_to_document(
    query_text: str, document: bioc.BioCDocument, min_aln_score: float, header_lines: int, **kwargs
) -> TextMatch:
    full_text = ''.join([p.text for p in document.passages])

    match = exhaustive_match(full_text, query_text, min_aln_score, **kwargs)

    # grab some context data at the same time
    # table header if this annotation is in a table
    if match['start_pos'] is not None and match['extracted_text']:
        match['table_header'] = fetch_table_header(
            match['extracted_text'], match['start_pos'], document, keep_lines=header_lines
        )

        match['section'] = ';'.join(
            get_span_sections(document, match['start_pos'], match['end_pos'])
        )

    return match


def load_match_files():
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'info').upper())
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_file', required=True)
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()

    dfs = []

    for input_file in args.input:
        logging.info(f'reading: {input_file}')
        dfs.append(pd.read_json(input_file, lines=True, dtype=False))

    df = pd.concat(dfs).drop_duplicates()

    conn = connect(args.db_file)
    try:
        # load previous matches
        matches = conn.execute('''SELECT content_id, query_text FROM text_matches''').fetchall()
        logging.info(f'fetched {len(matches)} previous matches')
        matches_df = pd.DataFrame.from_records([dict(r) for r in matches])

        if matches_df.shape[0] > 0:
            matches_df['_matched'] = True
            df = df.merge(matches_df, how='left')
            df = df[df._matched != True]  # noqa
            df = df.drop(columns=['_matched'])

        add_matches(conn, dump_records(df))
    finally:
        conn.close()


if __name__ == '__main__':
    load_match_files()

if __name__ == '__main__':
    load_match_files()
