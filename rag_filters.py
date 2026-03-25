"""
RAG 검색 필터 유틸리티.

rag_system.py에서 분리된 이유: 무거운 ML 임포트 없이 테스트 가능하도록.
"""
from typing import Any, Dict, Optional


def _apply_explicit_filters(
    auto_filters: Dict[str, Any],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    institution: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    명시적 필터 파라미터를 자동 추출 필터에 병합한다.
    명시적 파라미터가 있으면 해당 자동 추출값을 대체한다.
    """
    filters = auto_filters.copy()

    if date_from is not None:
        filters.pop('date_filter', None)
        filters.pop('date_value', None)
        filters['date_from'] = date_from

    if date_to is not None:
        filters['date_to'] = date_to

    if institution is not None:
        filters['institution'] = institution

    if doc_type is not None:
        filters['doc_type'] = doc_type

    return filters
