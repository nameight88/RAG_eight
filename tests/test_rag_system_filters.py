"""rag_system 명시적 필터 파라미터 검증"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_explicit_date_from_overrides_auto():
    """명시적 date_from이 자동 추출 날짜 필터를 대체해야 한다"""
    from rag_filters import _apply_explicit_filters

    auto_filters = {'date_filter': 'date', 'date_value': '2022'}
    result = _apply_explicit_filters(
        auto_filters,
        date_from='2020-01-01',
        date_to=None,
        institution=None,
        doc_type=None
    )
    assert result.get('date_from') == '2020-01-01'
    assert 'date_value' not in result
    assert 'date_filter' not in result


def test_explicit_institution_filter():
    """명시적 기관명 필터가 적용되어야 한다"""
    from rag_filters import _apply_explicit_filters

    auto_filters = {}
    result = _apply_explicit_filters(
        auto_filters,
        date_from=None,
        date_to=None,
        institution='국민은행',
        doc_type=None
    )
    assert result.get('institution') == '국민은행'


def test_no_explicit_filters_preserves_auto():
    """명시적 필터 없으면 자동 추출 결과가 유지된다"""
    from rag_filters import _apply_explicit_filters

    auto_filters = {'date_filter': 'date', 'date_value': '2022', 'doc_type': 'management'}
    result = _apply_explicit_filters(
        auto_filters,
        date_from=None,
        date_to=None,
        institution=None,
        doc_type=None
    )
    assert result.get('date_value') == '2022'
    assert result.get('doc_type') == 'management'


def test_all_explicit_filters():
    """모든 명시적 필터가 동시에 적용된다"""
    from rag_filters import _apply_explicit_filters

    auto_filters = {'date_filter': 'date', 'date_value': '2020'}
    result = _apply_explicit_filters(
        auto_filters,
        date_from='2023-01-01',
        date_to='2023-12-31',
        institution='신한은행',
        doc_type='sanctions'
    )
    assert result.get('date_from') == '2023-01-01'
    assert result.get('date_to') == '2023-12-31'
    assert result.get('institution') == '신한은행'
    assert result.get('doc_type') == 'sanctions'
    assert 'date_value' not in result
