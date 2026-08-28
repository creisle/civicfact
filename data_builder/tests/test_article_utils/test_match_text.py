import os
from io import StringIO

import bioc
import pytest
import requests_cache
from bioconverters import pmcxml2bioc

from article_utils.match_text import (
    exhaustive_match,
    get_span_sections,
    match_text_to_document,
    regex_match_substring,
    string_match,
)
from hypothesis.util import is_junk

# globally cache requests
requests_cache.install_cache(
    '.requests_cache',
)


def read_test_pmc_file(pmcid):
    filepath = os.path.join(os.path.dirname(__file__), f'../data/{pmcid}.xml')

    with open(filepath, 'r') as fh:
        docs = list(pmcxml2bioc(fh))
    return docs[0]


@pytest.fixture
def PMC4938529():
    return read_test_pmc_file('PMC4938529')


@pytest.mark.parametrize(
    'span_start,span_end,expected_section', [[31478, 31737, 'article//results']]
)
def test_get_span_sections(PMC4938529, span_start, span_end, expected_section):
    sections = get_span_sections(PMC4938529, span_start, span_end)
    assert expected_section in sections


@pytest.mark.parametrize(
    'article_id, target_text, matched_text, matched_table_header',
    [
        [
            'PMC5216719',
            '35L747_P755delinsSKG (9.4%)negative (740)E545K (8.3%)d',
            '35	L747_P755delinsSKG (9.4%)\tnegative (740)\tE545K (8.3%) d',
            'Case a\tEGFR TKI-sensitizing\tEGFR T790M b\tPIK3CA c, d, e\tOther mutations c, d, e\tmut no. f',
        ],
        [
            'PMC4936809',
            'Aug 25, 2012 – Jun 24, 2013Everolimus and fulvestrant10 monthsSD, PFS =10 months',
            'Aug 25, 2012 - Jun 24, 2013		Everolimus and fulvestrant	10 months	SD, PFS =10 months',
            'Time\tEmerging sites of metastatic disease\tRescued treatment regimens\tPeriod a\tResponse (month)',
        ],
    ],
)
def test_exhaustive_match_tables(article_id, target_text, matched_text, matched_table_header):
    doc = read_test_pmc_file(article_id)

    match = match_text_to_document(target_text, doc, min_aln_score=0.8)

    assert match['extracted_text'] == matched_text
    assert match['table_header'] == matched_table_header, 'header failed to match'


@pytest.mark.parametrize(
    'pmcid,target_text,expected_match',
    [
        (
            'PMC4057395',
            ' of the MRE11 inhibitor mirin and treated them with BMN673 (Fig. 3C). In both cases mirin had a negative impact ',
            'of the MRE11 inhibitor mirin and treated them with BMN673 (Fig. 3C). In both cases mirin had a negative impact',
        )
    ],
)
def test_exhaustive_match(pmcid, target_text, expected_match):
    doc = read_test_pmc_file(pmcid)
    match = exhaustive_match(''.join(p.text for p in doc.passages), target_text, 0.8)
    assert match['extracted_text'].strip() == expected_match.strip()


@pytest.mark.parametrize(
    'target_text,ref_text,expected_match',
    [
        (
            'tumors',
            '(47%). The mean tumor size was 72.7 mm for mutated tumors compared to 65.3 mm for wildtype tumors (p = 0.06).',
            'tumors',
        ),
        (
            'TNM',
            'classified according to the 1987 revision of the UICC-TNM classification, using information from the cancer',
            'TNM',
        ),
        (
            'VHL',
            'in order for pVHL to lose its function, biallelic VHL defects have to occur. We did not measure LOH, since',
            'VHL',
        ),
    ],
)
def test_string_match(target_text, ref_text, expected_match):
    # string_match(norm_reference_text, norm_query_text)
    match = string_match(
        ref_text,
        target_text,
        # min_aln_score=0.75,
        # ref_gap_penalty=-0.5,
        # gap_exception=lambda char: is_junk(char) or '-' == char,
    )
    assert match['extracted_text'].strip() == expected_match.strip()


@pytest.mark.parametrize(
    'query_text, prefix, matched_text, suffix',
    [
        (
            'VHL',
            'in order for pVHL to lose its function, biallelic ',
            'VHL',
            ' defects have to occur. We did not measure LOH, since',
        ),
        (
            'tumor',
            'mutations were truncating mutations (47%). The mean ',
            'tumor',
            ' size was 72.7 mm for mutated tumors compared to 65.3',
        ),
        (
            'MM',
            'extensively investigated in biological samples derived from ',
            'MM',
            ' patients. However, data concerning their potential',
        ),
        (
            'L',
            'lung cancers developing in both LSL-Kras-G12D Tsc1L/',
            'L',
            ' and LSL-Kras-G12D Tsc1L/+ mice in comparison to tumors',
        ),
        (
            'PA',
            ' tumors of childhood, and pilocytic astrocytomas (',
            'PA',
            ') are the most common central nervous system tumor ',
        ),
    ],
)
def test_string_match_pos(query_text, prefix, matched_text, suffix):
    # string_match(norm_reference_text, norm_query_text)
    ref_text = prefix + matched_text + suffix
    match = string_match(
        ref_text,
        query_text,
    )
    assert match['start_pos'] == len(prefix)


@pytest.mark.parametrize(
    'query_string, prefix, matched_text, suffix',
    [
        (
            'von Hippel-Lindau',
            'Prevalence of ',
            'von Hippel-Lindau',
            ' gene mutations in sporadic renal cell carcinoma:',
        ),
        (
            'PA',
            ' tumors of childhood, and pilocytic astrocytomas (',
            'PA',
            ') are the most common central nervous system tumor ',
        ),
    ],
)
def test_regex_match_substring(query_string, prefix, matched_text, suffix):
    text = prefix + matched_text + suffix
    matches = regex_match_substring(
        text,
        query_string,
        add_word_boundaries=bool(len(query_string) < 3),
        case_sensitive=bool(len(query_string) < 3),
    )
    assert any([m['text'] == matched_text and m['start'] == len(prefix) for m in matches])
