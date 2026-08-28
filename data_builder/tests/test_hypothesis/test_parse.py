import pytest

from hypothesis.parse import parse_in_table, parse_status_label
from nli.util import NEUTRAL_LABEL


@pytest.mark.parametrize(
    'tag,label', [("eid:8639-status:NEI", NEUTRAL_LABEL), ('status:NEI', NEUTRAL_LABEL)]
)
def test_parse_status_label(tag, label):
    assert parse_status_label(tag) == label


@pytest.mark.parametrize(
    'start_container,end_container,status',
    [
        (
            "/main[1]/article[1]/section[3]/div[1]/div[1]/div[7]/div[2]/div[2]/div[2]/table[1]/tbody[1]/tr[27]/td[9]/strong[1]",
            "/main[1]/article[1]/section[3]/div[1]/div[1]/div[7]/div[2]/div[2]/div[2]/table[1]/tbody[1]/tr[15]/td[1]/strong[1]/em[1]",
            True,
        )
    ],
)
def test_parse_in_table(start_container: str, end_container: str, status: bool):
    mock = {
        'selector': [
            {
                'type': "RangeSelector",
                'startContainer': start_container,
                'endContainer': end_container,
            }
        ]
    }
    assert parse_in_table(mock) == status
