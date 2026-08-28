import pytest

from civic.generate_claims import expand_variant_aliases


@pytest.mark.parametrize(
    'alias, exp_result', [('Y98FS (c.291delC)', ['Y98FS', 'c.291delC', '291delC', 'del291C'])]
)
def test_expand_variant_alises(alias, exp_result):
    result = expand_variant_aliases('', alias)
    assert sorted(result) == sorted(set(exp_result) | set([alias]))
