"""adaptive_chunker 메타데이터 필드 검증"""
from date_normalizer import normalize_date


def _make_year_month(date_str):
    """날짜 문자열에서 year, month 정수 추출 헬퍼"""
    normalized = normalize_date(date_str) or date_str
    year = int(normalized[:4]) if normalized and len(normalized) >= 4 and normalized[:4].isdigit() else 0
    month = int(normalized[5:7]) if normalized and len(normalized) >= 7 and normalized[5:7].isdigit() else 0
    return year, month


def test_year_month_extraction_normal():
    """정상 날짜에서 year/month 추출"""
    year, month = _make_year_month("2023-06-15")
    assert year == 2023
    assert month == 6


def test_year_month_extraction_raw_format():
    """정규화 전 날짜에서도 year/month 추출 가능"""
    year, month = _make_year_month("2016. 3. 15")
    assert year == 2016
    assert month == 3


def test_year_month_extraction_none():
    """None/빈 날짜는 year=0, month=0"""
    year, month = _make_year_month(None)
    assert year == 0
    assert month == 0
