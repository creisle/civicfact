import pytest

from hypothesis.util import parse_hypothesis_annotation, parse_tag


@pytest.mark.parametrize(
    'tag,eid,is_statement_context,status,example_number',
    [
        ['eid:686', '686', False, 'SUPPORTS', '1'],
        ['eid:686-status:NEI', '686', False, 'NEI', '1'],
        ['eid:686-status:REFUTES', '686', False, 'REFUTES', '1'],
        ['eid:686-statement', '686', True, 'SUPPORTS', '1'],
        ['variant-eid:686', '686', False, 'SUPPORTS', '1'],
        ['variant-eid:686-statement', '686', True, 'SUPPORTS', '1'],
        ['eid:9', '9', False, 'SUPPORTS', '1'],
        ['eid:9-ex:3', '9', False, 'SUPPORTS', '3'],
        ['eid:9-ex:3-statement', '9', True, 'SUPPORTS', '3'],
        ['eid:9-ex:3-status:NEI', '9', False, 'NEI', '3'],
        ['eid:9-ex:3-status:NEI-statement', '9', True, 'NEI', '3'],
    ],
)
def test_parse_tag_eid(tag, eid, is_statement_context, status, example_number):
    parsed = parse_tag(1, tag)
    print(parsed)
    assert parsed.civic_eid == eid
    assert parsed.is_statement_context == is_statement_context
    assert parsed.status == status
    assert parsed.example_number == example_number


def test_parse_hypothesis_annotation():
    test_ann = {
        'id': 'HP_NgN22Eeu4D5PUOr_ELA',
        'created': '2021-07-05T17:26:20.419289+00:00',
        'updated': '2021-07-05T17:27:40.345795+00:00',
        'user': 'acct:camgrisdale@hypothes.is',
        'uri': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2873663/',
        'text': 'TP53 C135F variant results in loss of function and is found only in somatic setting',
        'tags': ['EID:9440'],
        'group': 'JRwd19vg',
        'permissions': {
            'read': ['group:JRwd19vg'],
            'admin': ['acct:camgrisdale@hypothes.is'],
            'update': ['acct:camgrisdale@hypothes.is'],
            'delete': ['acct:camgrisdale@hypothes.is'],
        },
        'target': [
            {
                'source': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2873663/',
                'selector': [
                    {
                        'type': 'RangeSelector',
                        'endOffset': 1,
                        'startOffset': 0,
                        'endContainer': '/div[1]/div[1]/div[2]/div[1]/div[2]/div[2]/div[1]/div[3]/div[1]/div[1]/div[7]/div[2]/div[1]/div[2]/table[1]/tbody[1]/tr[3]/td[5]',
                        'startContainer': '/div[1]/div[1]/div[2]/div[1]/div[2]/div[2]/div[1]/div[3]/div[1]/div[1]/div[7]/div[2]/div[1]/div[2]/table[1]/tbody[1]/tr[3]/td[1]',
                    },
                    {'end': 30775, 'type': 'TextPositionSelector', 'start': 30762},
                    {
                        'type': 'TextQuoteSelector',
                        'exact': 'C135FLOSS4930',
                        'prefix': '5RALTERED210L130VALTERED2130Neo.',
                        'suffix': 'C135YLOSS70110Neo.A138VFUNCTIONA',
                    },
                ],
            }
        ],
        'document': {
            'title': [
                'Altered-function p53 missense mutations identified in breast cancers can have subtle effects on transactivation'
            ]
        },
        'links': {
            'html': 'https://hypothes.is/a/HP_NgN22Eeu4D5PUOr_ELA',
            'incontext': 'https://hyp.is/HP_NgN22Eeu4D5PUOr_ELA/www.ncbi.nlm.nih.gov/pmc/articles/PMC2873663/',
            'json': 'https://hypothes.is/api/annotations/HP_NgN22Eeu4D5PUOr_ELA',
        },
        'user_info': {'display_name': None},
        'flagged': False,
        'moderation': {'flagCount': 0},
        'hidden': False,
    }
    parsed = parse_hypothesis_annotation(test_ann)
    assert parsed.tags
