"""
날짜 문자열을 YYYY-MM-DD 형식으로 정규화하는 모듈.

지원 입력 형식:
- "2016.06.16", "2014.1.15"         (점 구분, 공백 없음)
- "2016. 1. 22", "2016. 11. 7"      (점+공백 구분)
- "2023년 01월 05일"                 (한글 형식)
- "20230105"                         (순수 숫자 8자리)
"""

import re
from typing import Optional


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """
    날짜 문자열을 YYYY-MM-DD 형식으로 정규화한다.

    Args:
        raw: 임의 형식의 날짜 문자열 (None 허용)

    Returns:
        "YYYY-MM-DD" 형식 문자열, 파싱 실패 시 None
    """
    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip()
    if not raw:
        return None

    # 한글 형식: "2023년 01월 05일"
    kor_match = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', raw)
    if kor_match:
        y, m, d = kor_match.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"

    # 순수 숫자 8자리: "20230105"
    if re.fullmatch(r'\d{8}', raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    # 점/공백 구분: "2016.06.16", "2016. 1. 7", "2014.1.15"
    parts_match = re.match(r'(\d{4})\s*[.]\s*(\d{1,2})\s*[.]\s*(\d{1,2})', raw)
    if parts_match:
        y, m, d = parts_match.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"

    return None
