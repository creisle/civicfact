import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Union

import pandas as pd

LABEL_COL = 'gold_label_name'
ID_COL = 'id'
NEUTRAL_LABEL = 'NEI'
PARTITION_COL = 'partition'
SELECTION_DELIMITER = ' . . . '
TABLE_DELIM = '\t'
NEUTRAL_WORKFLOW_NAME = 'neutral-bot'

CIVIC_CLAIM_COL_ORDER = [
    'molecularProfile',
    'variantOrigin',
    'evidenceType',
    'evidenceDirection',
    'significance',
    'therapyInteractionType',
    'therapies',
    'disease',
    'phenotypes',
]

VALID_EVIDENCE_DIRECTIONS = {'DOES_NOT_SUPPORT', 'SUPPORTS', 'N/A'}


def null_or_empty(element):
    if isinstance(element, dict) or isinstance(element, list):
        return len(element) == 0
    return pd.isnull(element)


def get_col_from_table(table_text, target_col, row_delim='\n', col_delim='\t'):
    table_cells = [row.split(col_delim) for row in table_text.split(row_delim)]
    return table_cells[1][table_cells[0].index(target_col)]


def convert_col_to_words(col_name) -> str:
    words = re.sub(
        r'([a-z])([A-Z])', lambda pat: pat.group(1) + ' ' + pat.group(2).lower(), col_name
    )
    return words


def convert_enum_to_words(enum):
    if re.match(r'^([A-Z]|/|_|[0-9])+$', enum) and '_' in enum:
        result = re.sub(r'_', ' ', enum)
        result = result.lower()
        return result
    return enum


def get_table_header(data: list[list[str]]):
    columns = set()
    for col, _ in data:
        columns.add(col)
    min_repeat_size = None
    if len(data) != len(columns):

        # min repeat size is length of unique columns
        for repeat_size in range(len(columns), len(data)):
            valid_repeat = True
            for pos_i in range(repeat_size, len(data)):
                if data[pos_i][0] != data[pos_i % repeat_size][0]:
                    valid_repeat = False
                    break
            if valid_repeat:
                min_repeat_size = repeat_size
                break
    if min_repeat_size is None:
        return [c for c, _ in data]
    return [c for c, _ in data[:min_repeat_size]]


def preprocess_table_content(table):
    new_table = []
    for i in range(len(table)):

        if table[i][0] == 'evidenceType':
            new_table.append([table[i][0], convert_enum_to_words(table[i][1])])
        else:
            new_table.append(table[i])
    return new_table


def linearize_table(data: list[list[str]], col_delim='\t', row_delim='\n', add_header=True):
    table_header = [convert_col_to_words(c) for c in get_table_header(data)]
    rows = [col_delim.join(table_header)] if add_header else []
    if not data:
        return ''
    for row_start in range(0, len(data), len(table_header)):
        row = [v for _, v in data[row_start : row_start + len(table_header)]]
        if 'evidence type' in table_header and re.match(
            r'^[A-Z]+$', row[table_header.index('evidence type')]
        ):
            row[table_header.index('evidence type')] = row[
                table_header.index('evidence type')
            ].lower()
        rows.append(col_delim.join(row))
    return row_delim.join(rows)


def linearize_table_as_tuples(content, relation_delim=' is ', column_delim='; '):
    result = []
    for col, val in content:
        if val:
            result.append(f'{convert_col_to_words(col)}{relation_delim}{val}')
    return column_delim.join(result)


@dataclass
class Text:
    content: Union[str, list[tuple[str, str]], dict[str, str]]
    type: Literal['table', 'unstructured', 'claim'] = 'unstructured'
    meta: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(json.dumps(self))

    def __getitem__(self, key):
        if key in ['type', 'content', 'meta']:
            return getattr(self, key)
        raise TypeError(f'Text object cannot be indexed by: {key}')

    def to_json(self):
        return {'type': self.type, 'content': self.content, 'meta': self.meta}

    def concat(self, **kwargs):
        if self.type == 'claim':
            return json.dumps(self.content, sort_keys=True, indent='  ')
        if self.type == 'table' or isinstance(self.content, list):
            data = preprocess_table_content(self.content)
            return linearize_table(data, **kwargs)

        return self.content

    @staticmethod
    def parse(flat, type_='claim'):
        if type_ != 'claim':
            raise NotImplementedError(f'unrecognized type: {type_}')
        content = json.loads(flat)

        return Text(type=type, content=content)


def listify(items, delim=',', conjunction='and'):
    items = list(items)
    conjunction = f' {conjunction.strip()} '
    delim = f'{delim.strip()} '
    if len(items) <= 2:
        return conjunction.join(items)
    items = items[:-1] + [conjunction + items[-1]]
    return delim.join(items)


def standardize_mp_name(mp):
    matches = list(re.finditer(r'\s+(AND|OR)\s+', mp, re.IGNORECASE))
    if not matches or len(matches) < 2:
        return mp
    if '(' in mp:
        logging.warning(
            f'skipping MP normalization. we have yet to make someting to standarize MPs so complicated they require parenthesis: {mp}'
        )
        return mp

    split_mp = re.split(r'\s+OR\s+', mp, re.IGNORECASE)
    result = []
    for partial_mp in split_mp:
        partal_split_mp = sorted(re.split(r'\s+AND\s+', partial_mp, re.IGNORECASE))
        partal_split_mp = ' AND '.join(partal_split_mp)
        result.append(partal_split_mp)
    result = ' OR '.join(sorted(result))
    return result


def build_civic_claim(row, prefix='civic.') -> Text:
    drug_names = [d['name'] for d in (row[f'{prefix}therapies'] or [])]
    evidence_direction = row[f'{prefix}evidenceDirection']
    assert (
        evidence_direction in VALID_EVIDENCE_DIRECTIONS
    ), f'{evidence_direction} is not a valid evidenceDirection. Expetced one of: {VALID_EVIDENCE_DIRECTIONS} '
    data = {
        'molecularProfile': standardize_mp_name(row[f'{prefix}molecularProfile.name']),
        'evidenceType': row[f'{prefix}evidenceType'],
        'significance': row[f'{prefix}significance'],
        'therapies': [d['name'] for d in (row[f'{prefix}therapies'] or [])],
        'disease': (
            row[f'{prefix}disease.name']
            if row[f'{prefix}disease.name'] and not pd.isnull(row[f'{prefix}disease.name'])
            else row[f'{prefix}disease.name']
        ),
        'phenotypes': [p['name'] for p in (row.get(f'{prefix}phenotypes') or []) if p['name']],
        'evidenceDirection': evidence_direction,
        'variantOrigin': row[f'{prefix}variantOrigin'],
        'therapyInteractionType': row[f'{prefix}therapyInteractionType'],
    }
    synonyms = []

    if row[f'{prefix}disease.name'] and row.get(f'{prefix}disease.diseaseAliases'):
        doid = row.get('doid', '')
        synonyms.append(
            {
                'name': row[f'{prefix}disease.name'],
                'type': 'disease',
                'synonyms': row[f'{prefix}disease.diseaseAliases'],
                'xrefs': f'DOID:{doid}' if doid else '',
            }
        )

    if drug_names:

        for drug in row[f'{prefix}therapies']:
            ncitId = drug.get('ncitId', '')
            synonyms.append(
                {
                    'type': 'therapy',
                    'name': drug['name'],
                    'synonyms': drug.get('therapyAliases', []),
                    'xrefs': f'NCI:{ncitId}' if ncitId else '',
                }
            )

    for variant in row[f'{prefix}molecularProfile.variants']:
        entrezId = variant['gene'].get('entrezId', '')
        synonyms.append(
            {
                'type': 'gene',
                'name': variant['gene']['name'],
                'synonyms': variant['gene'].get('geneAliases', []),
                'xrefs': f'entrezId:{entrezId}' if entrezId else '',
            }
        )
        variant_name = variant['name']
        if variant['gene']['name'] not in variant_name:
            variant_name = variant['gene']['name'] + ' ' + variant_name
        synonyms.append(
            {
                'type': 'variant',
                'name': variant_name,
                'synonyms': variant.get('variantAliases', []),
            }
        )

    result = {}
    for col in CIVIC_CLAIM_COL_ORDER:
        value = data[col]
        if not null_or_empty(value):
            result[col] = value
    if 'therapyInteractionType' in result and len(result.get('therapies', [])) < 2:
        del result['therapyInteractionType']
    return Text(result, type='claim', meta=dict(synonyms=[s for s in synonyms if s['synonyms']]))


def concat_text(text_list: list[Text] | Text, delimiter=SELECTION_DELIMITER, **kwargs):
    if not isinstance(text_list, list):
        text_list = [text_list]
    text_list = [Text(**t) if not isinstance(t, Text) else t for t in text_list]
    elements = [t.concat(**kwargs) for t in text_list]
    return delimiter.join([e.strip('\n') for e in elements])


def encrypt_id(id: str) -> str:
    encoded = base64.b64encode(id.encode('ascii', errors='ignore'))
    return encoded.decode('utf-8')


def partial_format(template, **kwargs):
    new_template = template.replace('{', '{{').replace('}', '}}')
    for name in kwargs:
        expected_key = f'{{{{{name}}}}}'
        if expected_key not in new_template:
            raise KeyError(name)
        new_template.replace(expected_key, f'{{{name}}}')
    return new_template.format(**kwargs)


def choose_template(claim_text: Text):
    claim = {k: v for k, v in claim_text['content']}
    evidence_type = claim['evidenceType'].lower()

    irr_verb = 'is'

    claim['molecularProfile'] = (
        claim['molecularProfile']
        .replace(' OR ', ' or ')
        .replace(' AND ', ' and ')
        .replace(' NOT ', ' not ')
    )
    template = ''

    if ' or ' in claim['molecularProfile']:
        irr_verb = 'are'
    elif ' and ' in claim['molecularProfile']:
        template = 'a molecular profile consisting of '

    if evidence_type == 'predictive':
        if ' or ' in claim['therapies']:
            template += f'{{molecularProfile}} {irr_verb} predictive of {{significance}} to'
            if ',' in claim['therapies']:
                template += ' any of'
            template += ' {therapies} in {diseases}'
        elif ' and ' in claim['therapies']:
            template += f'{{molecularProfile}} {irr_verb} predictive of {{significance}} to a combination of {{therapies}} in {{diseases}}'
        else:
            template += f'{{molecularProfile}} {irr_verb} predictive of {{significance}} to {{therapies}} in {{diseases}}'
    elif evidence_type == 'functional':
        if claim['significance'].endswith('ic'):
            template += f'{{molecularProfile}} results in {{significance}} functionality'
        elif claim['significance'].startswith('u'):
            template += f'{{molecularProfile}} results in {{significance}}'
        else:
            template += f'{{molecularProfile}} results in a {{significance}}'
    elif claim['significance'] == 'na':
        template += f'{{molecularProfile}} {irr_verb} a {evidence_type} indicator in {{diseases}}'
    elif evidence_type == 'diagnostic':
        template += f'{{molecularProfile}} {irr_verb} a {{significance}} {evidence_type} indicator in {{diseases}}'
    elif evidence_type == 'prognostic':
        template += f'{{molecularProfile}} {irr_verb} associated with a prognosis of {{significance}} in {{diseases}}'
    elif claim['significance'] == 'predisposition':
        template += (
            f'{{molecularProfile}} {irr_verb} associated with {{significance}} to {{diseases}}'
        )
    elif evidence_type == 'predisposing':
        if claim['significance'].startswith('uncertain'):
            irr_verb = 'has'
        template += f'{{molecularProfile}} {irr_verb} {{significance}} in {{diseases}}'
    else:
        template += (
            f'{{molecularProfile}} {irr_verb} associated with {{significance}} in {{diseases}}'
        )

    if 'phenotypes' in claim:
        if ' and ' in claim['phenotypes']:
            template += ' with the following phenotypes: {phenotypes}'
        else:
            template += ' with the {phenotypes} phenotype'
    return template


def convert_with_template(claim_text: Text):
    claim = {k: v for k, v in claim_text['content']}
    template = choose_template(claim_text)

    return template.format(**claim)
