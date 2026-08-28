import pytest

from nli.util import convert_with_template, get_table_header, linearize_table


@pytest.mark.parametrize(
    'table,header',
    [
        (['a1', 'b2'], ['a', 'b']),
        (['a1', 'b2', 'a3'], ['a', 'b']),
        (['a1', 'b2', 'a3', 'c2'], ['a', 'b', 'a', 'c']),
    ],
)
def test_get_table_header(table, header):
    assert header == get_table_header(table)


@pytest.mark.parametrize(
    'table,text',
    [
        (['a1', 'b2'], "a\tb\n1\t2"),
        (['a1', 'b2', 'a3'], "a\tb\n1\t2\n3"),
        (['a1', 'b2', 'a3', 'c2'], "a\tb\ta\tc\n1\t2\t3\t2"),
    ],
)
def test_linearize_table(table, text):
    assert text == linearize_table(table, col_delim='\t', row_delim='\n')
    assert '\n'.join(text.split('\n')[1:]) == linearize_table(
        table, col_delim='\t', row_delim='\n', add_header=False
    )


@pytest.mark.parametrize(
    'claim,text',
    [
        (
            ['MUT', 'predictive', 'sensitivity', 'cancer', 'drug'],
            'MUT is predictive of sensitivity to drug in cancer',
        ),
        (
            ['MUT', 'predictive', 'sensitivity', 'cancer', 'drug A and drug B'],
            'MUT is predictive of sensitivity to a combination of drug A and drug B in cancer',
        ),
        (
            ['MUT', 'predictive', 'sensitivity', 'cancer', 'drug A, drug C, or drug B'],
            'MUT is predictive of sensitivity to any of drug A, drug C, or drug B in cancer',
        ),
        (
            ['MUT', 'prognostic', 'poor outcome', 'cancer'],
            'MUT is associated with a prognosis of poor outcome in cancer',
        ),
        (
            ['MUT or MUT', 'prognostic', 'poor outcome', 'cancer'],
            'MUT or MUT are associated with a prognosis of poor outcome in cancer',
        ),
        (
            ['MUT and MUT', 'prognostic', 'poor outcome', 'cancer'],
            'a molecular profile consisting of MUT and MUT is associated with a prognosis of poor outcome in cancer',
        ),
        (
            ['MUT', 'diagnostic', 'positive', 'cancer'],
            'MUT is a positive diagnostic indicator in cancer',
        ),
        (
            ['MUT', 'predisposing', 'predisposition', 'cancer'],
            'MUT is associated with predisposition to cancer',
        ),
        (
            ['MUT', 'functional', 'unaltered function'],
            'MUT results in unaltered function',
        ),
        (
            ['MUT', 'functional', 'neomorphic'],
            'MUT results in neomorphic functionality',
        ),
        (
            ['MUT', 'functional', 'dominant negative'],
            'MUT results in a dominant negative',
        ),
        (['MUT', 'predisposing', 'pathogenic', 'disease'], 'MUT is pathogenic in disease'),
        (
            ['MUT', 'predisposing', 'uncertain significance', 'disease'],
            'MUT has uncertain significance in disease',
        ),
        (
            ['MUT', 'predictive', 'sensitivity', 'cancer', 'drug1 or drug2'],
            'MUT is predictive of sensitivity to drug1 or drug2 in cancer',
        ),
    ],
)
def test_convert_with_template(claim, text):
    header = [
        'molecularProfile',
        'evidenceType',
        'significance',
        'diseases',
        'therapies',
        'phenotypes',
    ]

    assert text == convert_with_template({'content': list(zip(header, claim))})
