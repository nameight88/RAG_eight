import pytest
from date_normalizer import normalize_date


@pytest.mark.parametrize("raw,expected", [
    ("2016.06.16", "2016-06-16"),
    ("2016. 1. 22", "2016-01-22"),
    ("2014. 1. 7", "2014-01-07"),
    ("2016. 11. 22", "2016-11-22"),
    ("2016. 11. 7", "2016-11-07"),
    ("2014.1.15", "2014-01-15"),
    ("2023년 01월 05일", "2023-01-05"),
    ("", None),
    (None, None),
    ("invalid", None),
    ("20230105", "2023-01-05"),
])
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected
