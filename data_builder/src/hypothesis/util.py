import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Optional

from bioconverters.utils import cleanup_text

NLI_STATUS = Literal['SUPPORTS', 'REFUTES', 'NEI']


@dataclass
class CivicTag:
    id: int
    civic_eid: str
    is_statement_context: bool = False
    status: NLI_STATUS = 'SUPPORTS'
    example_number: Optional[str] = None

    def to_json(self):
        return {
            'id': self.id,
            'civic_eid': self.civic_eid,
            'is_statement_context': self.is_statement_context,
            'example_number': self.example_number,
            'status': self.status,
        }


def parse_tag(id: int, tag: str) -> CivicTag:
    parts = re.split(r'-+', tag)
    is_statement_context = False
    status = 'SUPPORTS'
    example_number = '1'  # assume only 1 example unless otherwise specified
    civic_eid = None

    for part in parts:
        part = part.strip().lower()
        m = re.match(r'((e[io]?d)|(edi))[:;]?(\d+)$', part)
        if m:
            civic_eid = m.group(4)
        elif part.startswith('status:'):
            status = part[len('status:') :].upper()
            if status == 'SUP':
                status = 'SUPPORTS'
            elif status == 'REF':
                status = 'REFUTES'
        elif part.startswith('ex:'):
            example_number = part[len('ex:') :]
        elif part == 'statement':
            is_statement_context = True

    if civic_eid is None:
        raise ValueError(f'Failed to parse tag ({tag})')
    return CivicTag(
        id,
        civic_eid,
        is_statement_context=is_statement_context,
        example_number=example_number,
        status=status,
    )


@dataclass
class CivicAnnotation:
    id: str
    uri: str
    text: str
    prefix: str
    suffix: str
    tags: list[CivicTag]
    pmc_id: Optional[int]
    pubmed_id: Optional[int]
    author: str
    updated: str
    in_table: bool = False


def get_pmcid(text) -> int:
    if re.match(r'^(PMC)?\d+$', str(text)):
        return int(text.replace('PMC', ''))  # type: ignore
    raise KeyError(f'Unable to parse PMCID from {text}')


def get_pmcid_from_url(url) -> Optional[int]:
    for part in url.split('/'):
        if part and re.match(r'^PMC\d+$', part):
            return get_pmcid(part)
    raise KeyError(f'Unable to parse PMCID from {url}')


def get_pubmed_from_url(url) -> Optional[int]:
    m = re.match(r'https?://pubmed.ncbi.nlm.nih.gov/(\d+)/?$', url)
    if m:
        return int(m.group(1))
    raise KeyError(f'Unable to parse pubmed ID from {url}')


def is_junk_char(char):
    if unicodedata.category(char)[0] in {'C', 'Z'}:
        return True
    return False


def is_junk(input_string):
    if all([is_junk_char(char) for char in input_string] + [True]):
        return True
    return False


def cleanup_hyp_errors(text):
    """
    Clean up the hypothesis selected text and remove any known bugs (ex. insertions based on mark-up)
    """
    text = cleanup_text(text)
    pubmed_marker_pattern = (
        r'{\s*"type":\s*"(clinical-trial|entrez-nucleotide)",\s*"attrs":\s*{[^}]*}\s*}'
    )
    text = re.sub(pubmed_marker_pattern, '', text)
    text = re.sub(r'\s\s+', ' ', text)

    # weird figure name doubling
    text = re.sub(r'(figure)\s*\(figure', r'\1', text, flags=re.IGNORECASE)
    return text


def parse_hypothesis_annotation(annotation: dict) -> CivicAnnotation:
    """
    Convert the raw JSON annotation fetched from Hypothes.is into an annotation object
    """
    assert len(annotation['target']) == 1
    target = annotation['target'][0]
    text = None
    html = None
    for selector in target['selector']:
        if selector['type'] == 'TextPositionSelector':
            pass  # ignore since the positions do not reflect the actual position in the text
        elif selector['type'] == 'TextQuoteSelector':
            if text is not None:
                raise ValueError('Ambiguous text has multiple TextQuoteSelector values')
            text = selector
        elif selector['type'] == 'RangeSelector':
            if html is not None:
                raise ValueError('Ambiguous text has multiple RangeSelector values')
            html = selector

    if text is None:
        raise ValueError('missing required selector, TextQuoteSelector')

    in_table = False
    if not target['source'].endswith('.pdf'):
        if html is None:  # pdfs don't have html
            raise ValueError('missing required selector RangeSelector')
        html_start = re.sub(r'\[\d+\]', '', html['startContainer']).split('/')[-1]
        html_end = re.sub(r'\[\d+\]', '', html['endContainer']).split('/')[-1]
        in_table = bool(html_start == html_end and html_start == 'td')
    tags = []
    for i, tag in enumerate(annotation['tags']):
        try:
            tags.append(parse_tag(i, tag))
        except ValueError:
            pass

    pmc_id = None
    try:
        pmc_id = get_pmcid_from_url(annotation['uri'])
    except Exception:
        pass

    pubmed_id = None
    try:
        pubmed_id = get_pubmed_from_url(annotation['uri'])
    except Exception:
        pass

    return CivicAnnotation(
        author=annotation['user'],
        updated=annotation['updated'],
        id=annotation['id'],
        uri=annotation['uri'],
        text=cleanup_hyp_errors(text['exact']),
        suffix=text['suffix'],
        prefix=text['prefix'],
        in_table=in_table,
        tags=tags,
        pmc_id=pmc_id,
        pubmed_id=pubmed_id,
    )
