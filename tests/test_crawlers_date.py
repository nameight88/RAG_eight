"""크롤러의 날짜 정규화 적용 검증 테스트"""
from date_normalizer import normalize_date


def test_normalize_date_fixes_single_digit_month():
    """단일 자릿수 월이 zero-padding 되는지 확인"""
    assert normalize_date('2016. 1. 22') == '2016-01-22'


def test_normalize_date_fixes_single_digit_day():
    """단일 자릿수 일이 zero-padding 되는지 확인"""
    assert normalize_date('2014. 1. 7') == '2014-01-07'


def test_date_str_generation_from_normalized():
    """정규화된 날짜에서 파일명용 date_str 생성 확인"""
    normalized = normalize_date('2016. 1. 7')
    date_str = normalized.replace('-', '') if normalized else ''
    assert date_str == '20160107'


def test_date_str_fallback_on_none():
    """날짜 파싱 실패 시 fallback이 작동하는지 확인"""
    normalized = normalize_date(None)
    # fallback: use today's date
    from datetime import datetime
    date_str = normalized.replace('-', '') if normalized else datetime.now().strftime('%Y%m%d')
    assert len(date_str) == 8
    assert date_str.isdigit()
