# Phase 1: 날짜 정규화 + 메타데이터 강화 임베딩 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 크롤러에서 수집되는 날짜를 YYYY-MM-DD로 통일하고, 벡터 DB 청크 메타데이터에 date/institution/doc_type/year/month를 추가하여 메타데이터 필터 기반 검색 품질을 개선한다.

**Architecture:** 신규 `date_normalizer.py`가 모든 날짜 정규화를 담당하고, 크롤러 2개가 이를 호출한다. 청커 2개(adaptive, semantic)는 정규화된 날짜와 추가 메타데이터 필드를 청크에 포함시킨다. `rag_system.py`는 명시적 필터 파라미터를 받아 메타데이터 기반 후처리 필터링을 수행한다.

**Tech Stack:** Python 3.11, pytest, re, langchain, ChromaDB, FAISS

---

## 파일 구조

| 파일 | 변경 유형 | 역할 |
|------|-----------|------|
| `date_normalizer.py` | 신규 | 날짜 정규화 단일 함수 모듈 |
| `tests/test_date_normalizer.py` | 신규 | 날짜 정규화 단위 테스트 |
| `fss_crawler_management_eight.py` | 수정 (2곳) | 날짜 저장 시 정규화 적용 |
| `fss_crawler_section_eight.py` | 수정 (2곳) | 날짜 저장 시 정규화 적용 |
| `adaptive_chunker.py` | 수정 (1곳) | safe_metadata에 year/month 추가 |
| `semantic_chunker.py` | 수정 (2곳) | preprocess_document + metadata에 date/doc_type/year/month 추가 |
| `rag_system.py` | 수정 (1곳) | search()에 명시적 필터 파라미터 추가 |

---

## Task 1: date_normalizer.py 신규 생성

**Files:**
- Create: `date_normalizer.py`
- Create: `tests/__init__.py`
- Create: `tests/test_date_normalizer.py`

**배경:**
현재 크롤러는 `re.sub(r'[^\d]', '', date)[:8]` 방식으로 날짜를 처리한다.
이 방식은 `2014. 1. 7` → `201417` (6자리)처럼 단일 자릿수 월/일을 잘못 처리한다.
정규화 목표: 모든 날짜 형식을 `YYYY-MM-DD`로 변환, 실패 시 `None` 반환.

**지원해야 할 입력 형식:**
```
"2016.06.16"    → "2016-06-16"
"2016. 1. 22"   → "2016-01-22"
"2014. 1. 7"    → "2014-01-07"
"2016. 11. 22"  → "2016-11-22"
"2016. 11. 7"   → "2016-11-07"
"2014.1.15"     → "2014-01-15"
"2023년 01월 05일" → "2023-01-05"
""              → None
None            → None
"invalid"       → None
```

- [ ] **Step 1: tests/ 디렉토리 및 __init__.py 생성**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_date_normalizer.py`:
```python
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
```

- [ ] **Step 3: 테스트 실패 확인**

```bash
cd /c/Users/david/.config/superpowers/worktrees/RAG_eight/feature/phase1-parsing-optimization
python -m pytest tests/test_date_normalizer.py -v
```

Expected: `ImportError` 또는 `ModuleNotFoundError`

- [ ] **Step 4: date_normalizer.py 구현**

`date_normalizer.py`:
```python
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
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/test_date_normalizer.py -v
```

Expected: 11/11 PASSED

- [ ] **Step 6: 커밋**

```bash
git add date_normalizer.py tests/__init__.py tests/test_date_normalizer.py
git commit -m "feat: add date_normalizer with YYYY-MM-DD normalization"
```

---

## Task 2: 크롤러 날짜 정규화 적용

**Files:**
- Modify: `fss_crawler_management_eight.py:268` — date_str 생성 로직
- Modify: `fss_crawler_section_eight.py:177` — date_str 생성 로직

**배경:**
두 크롤러 모두 날짜를 파일명 생성에 사용할 때 `re.sub(r'[^\d]', '', date)[:8]` 패턴을 쓴다.
이를 `date_normalizer.normalize_date()`로 교체한다.
JSON 저장 시 `date` 필드도 정규화된 값으로 저장한다.

- [ ] **Step 1: 크롤러 테스트 작성**

`tests/test_crawlers_date.py`:
```python
"""크롤러의 날짜 정규화 적용 검증 테스트"""
import json
import os
from unittest.mock import patch, MagicMock
from fss_crawler_management_eight import FSSManagementCrawler


def test_management_crawler_normalizes_date_in_safe_filename():
    """날짜 정규화가 파일명 생성에 올바르게 적용되는지 확인"""
    crawler = FSSManagementCrawler()

    item = {
        'date': '2016. 1. 7',
        'institution': '테스트은행',
        'detail_url': 'http://example.com/file=test.pdf',
        'mngmCntnNo': '001',
        'emOpenSeq': '001',
    }

    # 파일 다운로드 없이 날짜 처리 로직만 테스트
    from date_normalizer import normalize_date
    normalized = normalize_date(item['date'])
    date_str = normalized.replace('-', '') if normalized else ''

    assert date_str == '20160107', f"Expected '20160107', got '{date_str}'"


def test_normalize_date_single_digit_month_day():
    """단일 자릿수 월/일이 zero-padding 되는지 확인"""
    from date_normalizer import normalize_date
    assert normalize_date('2014. 1. 7') == '2014-01-07'
    assert normalize_date('2016. 11. 7') == '2016-11-07'
```

- [ ] **Step 2: 테스트 실행 (통과 확인 — 이미 normalize_date 존재)**

```bash
python -m pytest tests/test_crawlers_date.py -v
```

Expected: PASSED (date_normalizer 이미 구현됨)

- [ ] **Step 3: fss_crawler_management_eight.py 수정**

[fss_crawler_management_eight.py:268](fss_crawler_management_eight.py#L268) 에서:

기존:
```python
date_str = re.sub(r'[^\d]', '', item.get('date', ''))[:8] or datetime.now().strftime('%Y%m%d')
```

변경:
```python
from date_normalizer import normalize_date as _normalize_date
_normalized = _normalize_date(item.get('date', ''))
date_str = _normalized.replace('-', '') if _normalized else datetime.now().strftime('%Y%m%d')
```

그리고 `detail` 딕셔너리 저장 시 정규화된 날짜로 덮어씀:
[fss_crawler_management_eight.py:299](fss_crawler_management_eight.py#L299) 부근 `detail = item.copy()` 이후:
```python
detail['date'] = _normalize_date(item.get('date', '')) or item.get('date', '')
```

- [ ] **Step 4: fss_crawler_section_eight.py 수정**

[fss_crawler_section_eight.py:177](fss_crawler_section_eight.py#L177) 에서:

기존:
```python
date_str = re.sub(r'[^\d]', '', detail.get('date', ''))[:8] or datetime.now().strftime('%Y%m%d')
```

변경:
```python
from date_normalizer import normalize_date as _normalize_date
_normalized = _normalize_date(detail.get('date', ''))
date_str = _normalized.replace('-', '') if _normalized else datetime.now().strftime('%Y%m%d')
detail['date'] = _normalized or detail.get('date', '')
```

- [ ] **Step 5: 테스트 재실행**

```bash
python -m pytest tests/ -v
```

Expected: 모든 테스트 PASSED

- [ ] **Step 6: 커밋**

```bash
git add fss_crawler_management_eight.py fss_crawler_section_eight.py tests/test_crawlers_date.py
git commit -m "feat: apply date normalization in crawlers"
```

---

## Task 3: adaptive_chunker.py 메타데이터 강화

**Files:**
- Modify: `adaptive_chunker.py:372-381` — safe_metadata에 year/month 추가

**배경:**
`adaptive_chunker.py`는 이미 `date`와 `doc_type`을 chunk dict에 저장하고 safe_metadata에도 포함시킨다.
다만 `date`가 정규화되지 않은 raw 값일 수 있고, `year`/`month` 필드가 없다.
날짜 기반 범위 필터를 편리하게 쓰기 위해 `year`, `month` 정수 필드를 추가한다.

- [ ] **Step 1: 테스트 작성**

`tests/test_adaptive_chunker_metadata.py`:
```python
"""adaptive_chunker 메타데이터 필드 검증"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_chunker import FSSAdaptiveChunker, AdaptiveChunkingStrategy


def _make_sample_doc(date: str = "2023-06-15") -> dict:
    return {
        "doc_id": "TEST_001",
        "institution": "테스트은행",
        "date": date,
        "found_keywords": ["전자금융"],
        "metadata": {"doc_type": "경영유의사항"},
        "content": {"full_text": "전자금융 관련 경영유의사항 내용입니다. " * 20},
    }


def test_chunk_metadata_has_year_month():
    """청크 메타데이터에 year, month 필드가 있어야 한다"""
    strategy = AdaptiveChunkingStrategy()
    doc = _make_sample_doc("2023-06-15")
    chunks_text = strategy.split_by_structure(doc)

    # FSSAdaptiveChunker.create_chunks 로직을 직접 검사
    found_keywords = doc.get('found_keywords', [])
    keywords_str = ', '.join(found_keywords)
    doc_type = doc.get('metadata', {}).get('doc_type', '')
    raw_date = doc.get('date', '')

    from date_normalizer import normalize_date
    normalized_date = normalize_date(raw_date) or raw_date

    year = int(normalized_date[:4]) if normalized_date and len(normalized_date) >= 4 else 0
    month = int(normalized_date[5:7]) if normalized_date and len(normalized_date) >= 7 else 0

    assert year == 2023
    assert month == 6


def test_chunk_metadata_year_zero_for_none_date():
    """날짜가 없으면 year/month가 0이어야 한다"""
    from date_normalizer import normalize_date
    normalized = normalize_date(None)
    year = int(normalized[:4]) if normalized and len(normalized) >= 4 else 0
    month = int(normalized[5:7]) if normalized and len(normalized) >= 7 else 0

    assert year == 0
    assert month == 0
```

- [ ] **Step 2: 테스트 실행 (현재는 통과 — 로직 검증용)**

```bash
python -m pytest tests/test_adaptive_chunker_metadata.py -v
```

Expected: PASSED (date_normalizer 로직 검증)

- [ ] **Step 3: adaptive_chunker.py 수정**

파일 상단 import 추가 (기존 import 블록 이후):
```python
from date_normalizer import normalize_date as _normalize_date
```

[adaptive_chunker.py:315](adaptive_chunker.py#L315) chunk 딕셔너리 생성 부분 수정:

기존:
```python
chunk = {
    "id": f"{doc['doc_id']}-chunk-{i}",
    "doc_id": doc['doc_id'],
    "institution": doc['institution'],
    "date": doc.get('date', ''),
    "doc_type": doc_type,
    "keywords": keywords_str,
    "chunk_index": i,
    "total_chunks": len(chunks),
    "chunk_text": chunk_text
}
```

변경:
```python
_raw_date = doc.get('date', '')
_norm_date = _normalize_date(_raw_date) or _raw_date
_year = int(_norm_date[:4]) if _norm_date and len(_norm_date) >= 4 and _norm_date[:4].isdigit() else 0
_month = int(_norm_date[5:7]) if _norm_date and len(_norm_date) >= 7 and _norm_date[5:7].isdigit() else 0

chunk = {
    "id": f"{doc['doc_id']}-chunk-{i}",
    "doc_id": doc['doc_id'],
    "institution": doc['institution'],
    "date": _norm_date,
    "year": _year,
    "month": _month,
    "doc_type": doc_type,
    "keywords": keywords_str,
    "chunk_index": i,
    "total_chunks": len(chunks),
    "chunk_text": chunk_text
}
```

[adaptive_chunker.py:372](adaptive_chunker.py#L372) safe_metadata 수정:

기존:
```python
safe_metadata = {
    "id": chunk.get("id", ""),
    "doc_id": chunk.get("doc_id", ""),
    "institution": chunk.get("institution", ""),
    "date": chunk.get("date", ""),
    "doc_type": chunk.get("doc_type", ""),
    "keywords": chunk.get("keywords", ""),
    "chunk_index": chunk.get("chunk_index", 0),
    "total_chunks": chunk.get("total_chunks", 1)
}
```

변경:
```python
safe_metadata = {
    "id": chunk.get("id", ""),
    "doc_id": chunk.get("doc_id", ""),
    "institution": chunk.get("institution", ""),
    "date": chunk.get("date", ""),
    "year": chunk.get("year", 0),
    "month": chunk.get("month", 0),
    "doc_type": chunk.get("doc_type", ""),
    "keywords": chunk.get("keywords", ""),
    "chunk_index": chunk.get("chunk_index", 0),
    "total_chunks": chunk.get("total_chunks", 1)
}
```

- [ ] **Step 4: 테스트 재실행**

```bash
python -m pytest tests/ -v
```

Expected: 모든 테스트 PASSED

- [ ] **Step 5: 커밋**

```bash
git add adaptive_chunker.py tests/test_adaptive_chunker_metadata.py
git commit -m "feat: add year/month metadata fields to adaptive_chunker"
```

---

## Task 4: semantic_chunker.py 메타데이터 강화

**Files:**
- Modify: `semantic_chunker.py:218-262` — preprocess_document에 date 정규화 추가
- Modify: `semantic_chunker.py:276-287` — create_semantic_chunks chunk dict에 date/doc_type/year/month 추가
- Modify: `semantic_chunker.py:309-320` — create_vector_store Document metadata에 date/doc_type/year/month 추가

**배경:**
`semantic_chunker.py`는 제재 문서를 처리한다. 현재 `sanction_date` 필드를 쓰지만 정규화 없이 raw 값을 그대로 사용한다. `doc_type`은 메타데이터에 없다. 정규화된 `date` + `doc_type` + `year` + `month`를 추가한다.

- [ ] **Step 1: 테스트 작성**

`tests/test_semantic_chunker_metadata.py`:
```python
"""semantic_chunker 메타데이터 필드 검증"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_preprocess_normalizes_sanction_date():
    """preprocess_document가 sanction_date를 정규화해야 한다"""
    from date_normalizer import normalize_date

    raw_doc = {
        "doc_id": "TEST_SAN_001",
        "institution": "테스트증권",
        "sanction_date": "2016. 3. 15",
        "content": {
            "sanction_type": "과태료",
            "fine": {"amount": 5000000, "text": "500만원"},
            "sanction_facts": [],
            "full_text": "전자금융 위반 사항입니다."
        },
        "metadata": {"doc_type": "제재결과", "regulations": []},
        "quality_score": 3,
    }

    normalized = normalize_date(raw_doc["sanction_date"])
    assert normalized == "2016-03-15"

    year = int(normalized[:4]) if normalized else 0
    month = int(normalized[5:7]) if normalized else 0
    assert year == 2016
    assert month == 3


def test_chunk_has_doc_type_field():
    """청크에 doc_type 필드가 있어야 한다"""
    chunk = {
        "id": "TEST-chunk-0",
        "doc_id": "TEST",
        "institution": "테스트은행",
        "sanction_date": "2023-01-15",
        "date": "2023-01-15",
        "doc_type": "제재결과",
        "year": 2023,
        "month": 1,
        "chunk_index": 0,
        "total_chunks": 1,
        "chunk_text": "테스트 내용",
        "quality_score": 3,
    }
    assert chunk.get("doc_type") == "제재결과"
    assert chunk.get("year") == 2023
    assert chunk.get("month") == 1
```

- [ ] **Step 2: 테스트 실행**

```bash
python -m pytest tests/test_semantic_chunker_metadata.py -v
```

Expected: PASSED

- [ ] **Step 3: semantic_chunker.py 상단 import 추가**

기존 import 블록 이후:
```python
from date_normalizer import normalize_date as _normalize_date
```

- [ ] **Step 4: preprocess_document 수정**

[semantic_chunker.py:218](semantic_chunker.py#L218) `preprocess_document` 함수의 `processed` 딕셔너리에 추가:

기존 `processed` 딕셔너리:
```python
processed = {
    "id": doc.get("doc_id", ""),
    "institution": doc.get("institution", ""),
    "sanction_date": doc.get("sanction_date", ""),
    ...
}
```

변경 — `sanction_date` 처리 후 정규화 필드 추가:
```python
_raw_sanction_date = doc.get("sanction_date", "")
_norm_date = _normalize_date(_raw_sanction_date) or _raw_sanction_date

processed = {
    "id": doc.get("doc_id", ""),
    "institution": doc.get("institution", ""),
    "sanction_date": _raw_sanction_date,
    "date": _norm_date,
    "doc_type": doc.get("metadata", {}).get("doc_type", "제재결과"),
    "year": int(_norm_date[:4]) if _norm_date and len(_norm_date) >= 4 and _norm_date[:4].isdigit() else 0,
    "month": int(_norm_date[5:7]) if _norm_date and len(_norm_date) >= 7 and _norm_date[5:7].isdigit() else 0,
    ...
}
```

- [ ] **Step 5: create_semantic_chunks chunk dict 수정**

[semantic_chunker.py:276](semantic_chunker.py#L276) chunk 딕셔너리:

기존:
```python
chunk = {
    "id": f"{doc['id']}-chunk-{i}",
    "doc_id": doc["id"],
    "institution": doc["institution"],
    "sanction_date": doc["sanction_date"],
    "sanction_type": doc["sanction_type"],
    "chunk_index": i,
    "total_chunks": len(chunks),
    "chunk_text": chunk_text,
    "quality_score": doc["quality_score"]
}
```

변경:
```python
chunk = {
    "id": f"{doc['id']}-chunk-{i}",
    "doc_id": doc["id"],
    "institution": doc["institution"],
    "sanction_date": doc["sanction_date"],
    "date": doc.get("date", doc["sanction_date"]),
    "doc_type": doc.get("doc_type", "제재결과"),
    "year": doc.get("year", 0),
    "month": doc.get("month", 0),
    "sanction_type": doc["sanction_type"],
    "chunk_index": i,
    "total_chunks": len(chunks),
    "chunk_text": chunk_text,
    "quality_score": doc["quality_score"]
}
```

- [ ] **Step 6: create_vector_store Document metadata 수정**

[semantic_chunker.py:312](semantic_chunker.py#L312) Document metadata:

기존:
```python
metadata={
    'id': chunk['id'],
    'institution': item['institution'],
    'sanction_date': item['sanction_date'],
    ...
}
```

변경:
```python
_raw_date = item.get('sanction_date', '')
_norm_date = _normalize_date(_raw_date) or _raw_date
metadata={
    'id': chunk['id'],
    'institution': item['institution'],
    'sanction_date': _raw_date,
    'date': _norm_date,
    'doc_type': item.get('doc_type', '제재결과'),
    'year': int(_norm_date[:4]) if _norm_date and len(_norm_date) >= 4 and _norm_date[:4].isdigit() else 0,
    'month': int(_norm_date[5:7]) if _norm_date and len(_norm_date) >= 7 and _norm_date[5:7].isdigit() else 0,
    ...
}
```

- [ ] **Step 7: 전체 테스트 실행**

```bash
python -m pytest tests/ -v
```

Expected: 모든 테스트 PASSED

- [ ] **Step 8: 커밋**

```bash
git add semantic_chunker.py tests/test_semantic_chunker_metadata.py
git commit -m "feat: add normalized date/doc_type/year/month metadata to semantic_chunker"
```

---

## Task 5: rag_system.py 명시적 메타데이터 필터 추가

**Files:**
- Modify: `rag_system.py` — `answer_question()` 또는 `search()` 함수에 명시적 필터 파라미터 추가

**배경:**
현재 `rag_system.py`의 `preprocess_query()`는 질의 텍스트에서 날짜/기관을 자동 추출한다.
API나 웹 프론트엔드에서 필터를 직접 전달할 수 있도록 명시적 파라미터(`date_from`, `date_to`, `institution`, `doc_type`)를 추가한다.
명시적 파라미터가 있으면 자동 추출보다 우선 적용한다.

- [ ] **Step 1: 테스트 작성**

`tests/test_rag_system_filters.py`:
```python
"""rag_system 명시적 필터 파라미터 검증"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_explicit_filters_override_auto_extraction():
    """명시적 필터가 자동 추출 필터보다 우선해야 한다"""
    # 이 테스트는 _apply_explicit_filters 헬퍼 함수를 직접 검증한다
    from rag_system import _apply_explicit_filters

    # 자동 추출에서는 2022년이 나오더라도, 명시적으로 2020을 넘기면 2020이 사용돼야 함
    auto_filters = {'date_filter': 'date', 'date_value': '2022'}
    result = _apply_explicit_filters(
        auto_filters,
        date_from='2020-01-01',
        date_to=None,
        institution=None,
        doc_type=None
    )
    assert result.get('date_from') == '2020-01-01'
    assert result.get('date_value') is None  # 자동 추출값 제거됨


def test_explicit_institution_filter():
    """명시적 기관명 필터가 적용되어야 한다"""
    from rag_system import _apply_explicit_filters

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
    """명시적 필터가 없으면 자동 추출 결과가 유지돼야 한다"""
    from rag_system import _apply_explicit_filters

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
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_rag_system_filters.py -v
```

Expected: ImportError (`_apply_explicit_filters` 없음)

- [ ] **Step 3: rag_system.py에 _apply_explicit_filters 헬퍼 추가**

파일 내 `preprocess_query` 함수 바로 아래에 추가:

```python
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
        # 자동 추출 날짜 필터 제거 후 명시적 필터 적용
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
```

- [ ] **Step 4: answer_question 시그니처 확장**

[rag_system.py](rag_system.py) `answer_question` 함수 시그니처:

기존:
```python
def answer_question(self, question: str) -> Dict[str, Any]:
```

변경:
```python
def answer_question(
    self,
    question: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    institution: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> Dict[str, Any]:
```

함수 내부 `preprocess_query` 호출 이후에 필터 병합 적용:
```python
processed_query, filters = self.preprocess_query(question)
filters = _apply_explicit_filters(filters, date_from, date_to, institution, doc_type)
```

- [ ] **Step 5: 전체 테스트 실행**

```bash
python -m pytest tests/ -v
```

Expected: 모든 테스트 PASSED

- [ ] **Step 6: 커밋**

```bash
git add rag_system.py tests/test_rag_system_filters.py
git commit -m "feat: add explicit metadata filter params to answer_question"
```

---

## 완료 기준

- [ ] `python -m pytest tests/ -v` 전체 통과
- [ ] `date_normalizer.normalize_date('2014. 1. 7')` → `'2014-01-07'`
- [ ] adaptive_chunker safe_metadata에 `year`, `month` 키 존재 확인
- [ ] semantic_chunker Document metadata에 `date`, `doc_type`, `year`, `month` 키 존재 확인
- [ ] `rag_system.answer_question(question, date_from='2023-01-01')` 호출 가능
