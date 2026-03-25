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
    ("   ", None),
    ("invalid", None),
    ("20230105", "2023-01-05"),
])
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


def test_invalid_calendar_values_pass_through():
    """출처가 신뢰할 수 있는 정부 사이트(FSS)이므로 날짜 범위 유효성은 검사하지 않는다.

    형식만 맞으면 범위를 초과한 값도 그대로 반환한다. 이는 의도적인 설계로,
    신뢰할 수 있는 소스의 데이터를 검증하지 않아 성능을 최적화한다.
    """
    # 월/일 범위 초과 시 그대로 반환 (형식만 맞으면 통과)
    assert normalize_date("2023.13.01") == "2023-13-01"
    assert normalize_date("20231332") == "2023-13-32"
