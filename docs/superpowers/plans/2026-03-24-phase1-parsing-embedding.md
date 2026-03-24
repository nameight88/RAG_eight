# Phase 1: 날짜 정규화 + 메타데이터 강화 임베딩 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 크롤러에서 수집된 날짜를 `YYYY-MM-DD`로 통일하고, 벡터 청크 메타데이터에 `date`, `institution`, `doc_type`, `year`, `month` 필드를 추가해 날짜·기관 기반 필터 검색을 가능하게 한다.

**Architecture:** `date_normalizer.py`를 신규 생성해 날짜 정규화 책임을 격리하고, 두 크롤러와 두 청커에서 이를 import해 사용한다. `rag_system.py`는 명시적 필터 파라미터를 받아 기존 `preprocess_query` 자동 추출보다 우선 적용한다.

**Tech Stack:** Python 3.12, pytest 7.4, 기존 LangChain/FAISS/ChromaDB 스택 유지

---

## 파일 맵

| 파일 | 유형 | 역할 |
|------|------|------|
| `date_normalizer.py` | 신규 | 날짜 정규화 단일 책임 모듈 |
| `tests/test_date_normalizer.py` | 신규 | 날짜 정규화 단위 테스트 |
| `fss_crawler_management_eight.py` | 수정 | 날짜 저장 시 `normalize_date()` 적용 |
| `fss_crawler_section_eight.py` | 수정 | 날짜 저장 시 `normalize_date()` 적용 |
| `adaptive_chunker.py` | 수정 | 메타데이터에 `year`, `month` 필드 추가 |
| `semantic_chunker.py` | 수정 | 메타데이터에 `date`, `year`, `month`, `doc_type` 표준화 |
| `rag_system.py` | 수정 | 명시적 필터 파라미터 추가, 우선순위 규칙 적용 |

---

## Task 1: 날짜 정규화 모듈 생성 (TDD)

**Files:**
- Create: `date_normalizer.py`
- Create: `tests/__init__.py`
- Create: `tests/test_date_normalizer.py`

- [ ] **Step 1: tests 디렉토리 및 테스트 파일 생성**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_date_normalizer.py`:
```python
import pytest
from date_normalizer import normalize_date


class TestNormalizeDate:
    """날짜 정규화 함수 단위 테스트"""

    def test_standard_format(self):
        assert normalize_date("2016.06.16") == "2016-06-16"

    def test_space_separated_single_digit_month_and_day(self):
        assert normalize_date("2014. 1. 7") == "2014-01-07"

    def test_space_separated_single_digit_day(self):
        assert normalize_date("2016. 11. 22") == "2016-11-22"

    def test_space_separated_single_digit_month(self):
        assert normalize_date("2016. 1. 22") == "2016-01-22"

    def test_no_space_single_digit_month(self):
        assert normalize_date("2014.1.15") == "2014-01-15"

    def test_no_space_single_digit_month_and_day(self):
        assert normalize_date("2014.1.7") == "2014-01-07"

    def test_already_normalized(self):
        assert normalize_date("2023-05-12") == "2023-05-12"

    def test_korean_format(self):
        assert normalize_date("2023년 5월 12일") == "2023-05-12"

    def test_empty_string_returns_none(self):
        assert normalize_date("") is None

    def test_none_returns_none(self):
        assert normalize_date(None) is None

    def test_unparseable_returns_none(self):
        assert normalize_date("알 수 없음") is None
```

- [ ] **Step 3: 테스트 실행해 실패 확인**

```bash
pytest tests/test_date_normalizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'date_normalizer'`

- [ ] **Step 4: `date_normalizer.py` 구현**

```python
"""
날짜 정규화 유틸리티
- 금융감독원 크롤러에서 수집되는 다양한 날짜 형식을 YYYY-MM-DD로 통일
"""
import re
from typing import Optional


def normalize_date(raw_date: Optional[str]) -> Optional[str]:
    """
    임의 형식의 날짜 문자열을 YYYY-MM-DD로 정규화한다.

    지원 형식:
        - YYYY.MM.DD / YYYY.M.D / YYYY.M.DD / YYYY.MM.D
        - YYYY. MM. DD / YYYY. M. D 등 공백 포함 변형
        - YYYY-MM-DD (이미 정규화된 경우)
        - YYYY년 MM월 DD일

    반환:
        정규화된 "YYYY-MM-DD" 문자열, 파싱 불가 시 None
    """
    if not raw_date or not isinstance(raw_date, str):
        return None

    raw = raw_date.strip()
    if not raw:
        return None

    # 이미 YYYY-MM-DD 형식
    if re.match(r'^\d{4}-\d{2}-\d{2}$', raw):
        return raw

    # 한글 형식: 2023년 5월 12일
    korean = re.match(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', raw)
    if korean:
        y, m, d = korean.group(1), korean.group(2), korean.group(3)
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    # 점/공백 구분 형식: YYYY.MM.DD, YYYY. M. D 등
    dot = re.match(r'(\d{4})\s*[.]\s*(\d{1,2})\s*[.]\s*(\d{1,2})', raw)
    if dot:
        y, m, d = dot.group(1), dot.group(2), dot.group(3)
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    return None
```

- [ ] **Step 5: 테스트 실행해 통과 확인**

```bash
pytest tests/test_date_normalizer.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 6: 커밋**

```bash
git add date_normalizer.py tests/__init__.py tests/test_date_normalizer.py
git commit -m "feat: add date_normalizer with full test coverage"
```

---

## Task 2: 경영유의사항 크롤러 날짜 정규화 적용

**Files:**
- Modify: `fss_crawler_management_eight.py:268`

현재 코드 (268행):
```python
date_str = re.sub(r'[^\d]', '', item.get('date', ''))[:8] or datetime.now().strftime('%Y%m%d')
```

- [ ] **Step 1: import 추가 및 날짜 저장 로직 수정**

`fss_crawler_management_eight.py` 상단 import 블록에 추가:
```python
from date_normalizer import normalize_date
```

268행의 `date_str` 생성 코드 교체:
```python
# 날짜 정규화 (YYYY-MM-DD → YYYYMMDD 변환)
normalized = normalize_date(item.get('date', ''))
if normalized:
    date_str = normalized.replace('-', '')
else:
    date_str = datetime.now().strftime('%Y%m%d')
    print(f"⚠️  날짜 파싱 실패: {item.get('date', '')} → 오늘 날짜 사용")
```

- [ ] **Step 2: 저장되는 JSON의 date 필드도 정규화된 값으로 저장**

`get_management_detail_and_download` 함수 내 `detail` dict 생성 부분에서
`detail['date']` 값을 정규화된 값으로 덮어씌운다:

`get_management_detail_and_download` 함수 내 `detail = item.copy()` 다음에 추가:
```python
detail['date'] = normalize_date(item.get('date', '')) or item.get('date', '')
```

- [ ] **Step 3: 기존 동작 보존 확인 (수동)**

```bash
python -c "
from fss_crawler_management_eight import FSSManagementCrawler
c = FSSManagementCrawler()
# 날짜가 있는 샘플 item으로 date_str 생성 확인
from date_normalizer import normalize_date
test_dates = ['2016.06.16', '2014. 1. 7', '2016. 11. 22', '']
for d in test_dates:
    norm = normalize_date(d)
    print(f'{d!r} -> {norm}')
"
```

Expected: 모든 날짜가 `YYYY-MM-DD` 또는 `None` 반환

- [ ] **Step 4: 커밋**

```bash
git add fss_crawler_management_eight.py
git commit -m "feat: apply date normalization in management crawler"
```

---

## Task 3: 제재결과 크롤러 날짜 정규화 적용

**Files:**
- Modify: `fss_crawler_section_eight.py:177`

현재 코드 (177행):
```python
date_str = re.sub(r'[^\d]', '', detail.get('date', ''))[:8] or datetime.now().strftime('%Y%m%d')
```

- [ ] **Step 1: import 추가 및 날짜 저장 로직 수정**

`fss_crawler_section_eight.py` 상단 import 블록에 추가:
```python
from date_normalizer import normalize_date
```

177행 교체:
```python
normalized = normalize_date(detail.get('date', ''))
if normalized:
    date_str = normalized.replace('-', '')
    detail['date'] = normalized
else:
    date_str = datetime.now().strftime('%Y%m%d')
    print(f"⚠️  날짜 파싱 실패: {detail.get('date', '')} → 오늘 날짜 사용")
```

- [ ] **Step 2: 동작 확인 (수동)**

```bash
python -c "
from date_normalizer import normalize_date
# 제재결과 JSON에서 실제 날짜 샘플 테스트
import json
with open('data/fss_sanctions_parsed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
docs = data.get('documents', data) if isinstance(data, dict) else data
none_count = 0
for d in docs:
    result = normalize_date(d.get('date', ''))
    if result is None:
        none_count += 1
        print(f'파싱 실패: {d.get(\"date\", \"\")}')
print(f'전체: {len(docs)}, 파싱 실패: {none_count}')
"
```

- [ ] **Step 3: 커밋**

```bash
git add fss_crawler_section_eight.py
git commit -m "feat: apply date normalization in sanctions crawler"
```

---

## Task 4: adaptive_chunker 메타데이터 강화

**Files:**
- Modify: `adaptive_chunker.py:372-381` (safe_metadata 블록)

현재 `safe_metadata`:
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

- [ ] **Step 1: import 추가**

`adaptive_chunker.py` 상단에:
```python
from date_normalizer import normalize_date
```

- [ ] **Step 2: safe_metadata 블록 교체**

```python
raw_date = chunk.get("date", "")
norm_date = normalize_date(raw_date) or raw_date
year = int(norm_date[:4]) if norm_date and len(norm_date) >= 4 and norm_date[:4].isdigit() else 0
month = int(norm_date[5:7]) if norm_date and len(norm_date) >= 7 and norm_date[5:7].isdigit() else 0

safe_metadata = {
    "id": chunk.get("id", ""),
    "doc_id": chunk.get("doc_id", ""),
    "institution": chunk.get("institution", ""),
    "date": norm_date,
    "year": year,
    "month": month,
    "doc_type": chunk.get("doc_type", "경영유의사항"),
    "keywords": chunk.get("keywords", ""),
    "chunk_index": chunk.get("chunk_index", 0),
    "total_chunks": chunk.get("total_chunks", 1)
}
```

FAISS 저장 경로의 `metadata` dict도 동일하게 수정 (`adaptive_chunker.py:492` 근처):
```python
raw_date = chunk.get('date', '')
norm_date = normalize_date(raw_date) or raw_date
year = int(norm_date[:4]) if norm_date and len(norm_date) >= 4 and norm_date[:4].isdigit() else 0
month = int(norm_date[5:7]) if norm_date and len(norm_date) >= 7 and norm_date[5:7].isdigit() else 0

metadata={
    'id': chunk.get('id', ''),
    'doc_id': chunk.get('doc_id', ''),
    'institution': chunk.get('institution', ''),
    'date': norm_date,
    'year': year,
    'month': month,
    'doc_type': chunk.get('doc_type', '경영유의사항'),
    'keywords': chunk.get('keywords', ''),
    'chunk_index': chunk.get('chunk_index', 0),
    'total_chunks': chunk.get('total_chunks', 1),
}
```

- [ ] **Step 3: 동작 확인 (수동)**

```bash
python -c "
from adaptive_chunker import FSSAdaptiveChunker
# 로드만 확인 (에러 없으면 OK)
print('adaptive_chunker import 성공')
from date_normalizer import normalize_date
print('normalize_date import 성공')
"
```

- [ ] **Step 4: 커밋**

```bash
git add adaptive_chunker.py
git commit -m "feat: add year/month metadata fields in adaptive_chunker"
```

---

## Task 5: semantic_chunker 메타데이터 강화

**Files:**
- Modify: `semantic_chunker.py` (processed dict 및 메타데이터 저장 부분)

현재 `semantic_chunker.py:221-229`에서 `sanction_date` 필드를 사용하고 있음. 이를 정규화된 `date`로 표준화한다.

- [ ] **Step 1: import 추가**

`semantic_chunker.py` 상단에:
```python
from date_normalizer import normalize_date
```

- [ ] **Step 2: processed dict에 정규화 적용 (221행 근처)**

```python
raw_date = doc.get("sanction_date", "") or doc.get("date", "")
norm_date = normalize_date(raw_date) or raw_date
year = int(norm_date[:4]) if norm_date and len(norm_date) >= 4 and norm_date[:4].isdigit() else 0
month = int(norm_date[5:7]) if norm_date and len(norm_date) >= 7 and norm_date[5:7].isdigit() else 0

processed = {
    "id": doc.get("doc_id", ""),
    "institution": doc.get("institution", ""),
    "date": norm_date,
    "year": year,
    "month": month,
    "doc_type": "제재결과",
    "sanction_type": doc.get("content", {}).get("sanction_type", ""),
    "fine_amount": doc.get("content", {}).get("fine", {}).get("amount", 0),
    "sanction_facts": doc.get("content", {}).get("sanction_facts", []),
    "full_text": doc.get("content", {}).get("full_text", ""),
}
```

- [ ] **Step 3: 메타데이터 저장 부분에도 `year`, `month`, `doc_type` 추가**

`semantic_chunker.py`의 벡터 저장 시 metadata dict (312~316행 근처, `chunk` dict 생성 블록):

```python
# 기존: chunk dict에 date, institution 등 포함
# 수정: year, month, doc_type 추가
chunk = {
    "id": f"{processed['id']}-chunk-{i}",
    "doc_id": processed["id"],
    "institution": processed["institution"],
    "date": processed["date"],
    "year": processed["year"],
    "month": processed["month"],
    "doc_type": processed["doc_type"],
    "chunk_text": chunk_text,
    "chunk_index": i,
    "total_chunks": len(chunks),
    "keywords": keywords_str,
}
```

ChromaDB `metadatas` 저장 시에도 `year`, `month`, `doc_type` 포함 확인 (chunk dict에서 자동으로 포함됨).

- [ ] **Step 4: 동작 확인 (수동)**

```bash
python -c "
from semantic_chunker import FSSSemanticChunker
print('semantic_chunker import 성공')
"
```

- [ ] **Step 5: 커밋**

```bash
git add semantic_chunker.py
git commit -m "feat: standardize metadata fields in semantic_chunker"
```

---

## Task 6: RAG 시스템 명시적 필터 파라미터 추가

**Files:**
- Modify: `rag_system.py:660-719` (preprocess_query), `895-940` (_apply_filters)

- [ ] **Step 1: 명시적 필터를 받는 `search_with_filters` 함수 추가**

`rag_system.py`의 `_apply_filters` 함수 아래에 추가:

```python
def search_with_filters(
    self,
    query: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    institution: Optional[str] = None,
    doc_type: Optional[str] = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    명시적 필터 파라미터를 받아 RAG 검색 수행.
    명시적 파라미터가 None이면 preprocess_query 자동 추출 결과 사용.
    """
    # 자동 추출 필터
    processed_query, auto_filters = self.preprocess_query(query)

    # 명시적 파라미터로 자동 추출 결과 덮어쓰기
    if date_from is not None:
        auto_filters['date_from'] = date_from
        auto_filters.pop('date_filter', None)
        auto_filters.pop('date_value', None)
    if date_to is not None:
        auto_filters['date_to'] = date_to
    if institution is not None:
        auto_filters['institution'] = institution
        auto_filters.pop('institution_types', None)
    if doc_type is not None:
        auto_filters['doc_type'] = doc_type

    # answer_question 내부적으로 preprocess_query를 다시 호출하지 않도록
    # 필터를 직접 적용하는 방식으로 검색 수행
    return self.answer_question(processed_query, _override_filters=auto_filters)
```

단, `answer_question` 함수 시그니처에 `_override_filters` 파라미터를 추가해야 한다:

`rag_system.py:721` `answer_question` 함수 시그니처 변경:
```python
def answer_question(self, question: str, _override_filters: dict | None = None) -> Dict[str, Any]:
```

함수 내부에서 `preprocess_query` 호출 후:
```python
processed_query, filters = self.preprocess_query(question)
# 명시적 override 필터가 있으면 덮어쓰기
if _override_filters:
    filters.update(_override_filters)
```

- [ ] **Step 2: `_apply_filters`에 `date_from`/`date_to` 범위 필터 지원 추가**

기존 `_apply_filters` 내 날짜 필터 블록(`if "date_filter" in filters`) **아래에** 다음 블록 추가
(기존 자동 추출 필터와 충돌 없음 — `date_from`/`date_to` 키가 기존 로직에 사용되지 않으므로):


```python
# date_from / date_to 범위 필터 (명시적 파라미터)
if "date_from" in filters or "date_to" in filters:
    doc_date = metadata.get("date", "")
    if doc_date:
        from date_normalizer import normalize_date
        norm = normalize_date(doc_date) or doc_date
        if "date_from" in filters and norm < filters["date_from"]:
            include_doc = False
        if "date_to" in filters and norm > filters["date_to"]:
            include_doc = False

# 명시적 institution 필터
if "institution" in filters and include_doc:
    inst = metadata.get("institution", "")
    if filters["institution"].lower() not in inst.lower():
        include_doc = False
```

- [ ] **Step 3: 동작 확인 (수동)**

```bash
python -c "
import sys
sys.path.insert(0, '.')
from rag_system import RAGSystem
print('RAGSystem import 성공')
# search_with_filters 메서드 존재 확인
r = RAGSystem.__dict__
print('search_with_filters 존재:', 'search_with_filters' in r)
"
```

- [ ] **Step 4: 커밋**

```bash
git add rag_system.py
git commit -m "feat: add explicit filter params to RAG search, override preprocess_query"
```

---

## Task 7: 기존 JSON 데이터 날짜 정규화 현황 측정

- [ ] **Step 1: 기존 파싱된 JSON에서 날짜 현황 리포트 실행**

```bash
python -c "
import json
from date_normalizer import normalize_date

for fname, label in [
    ('data/fss_management_parsed.json', '경영유의사항'),
    ('data/fss_sanctions_parsed.json', '제재결과'),
]:
    with open(fname, 'r', encoding='utf-8') as f:
        data = json.load(f)
    docs = data.get('documents', data) if isinstance(data, dict) else data
    total = len(docs)
    ok = sum(1 for d in docs if normalize_date(d.get('date','')) is not None)
    fail = total - ok
    print(f'[{label}] 전체: {total}, 정규화 성공: {ok}, 실패: {fail}')
    if fail > 0:
        for d in docs:
            if normalize_date(d.get('date','')) is None:
                print(f'  실패 날짜: {repr(d.get(\"date\",\"\"))}')
"
```

- [ ] **Step 2: 결과 기록 후 커밋**

```bash
git add -A
git commit -m "chore: measure date normalization coverage on existing data"
```

---

## Task 8: 전체 테스트 실행 및 최종 확인

- [ ] **Step 1: 전체 테스트 실행**

```bash
pytest tests/ -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 2: import 체인 최종 확인**

```bash
python -c "
from date_normalizer import normalize_date
from adaptive_chunker import FSSAdaptiveChunker
from semantic_chunker import FSSSemanticChunker
print('모든 모듈 import 성공')
"
```

- [ ] **Step 3: 최종 커밋**

```bash
git add -A
git commit -m "chore: phase1 complete - date normalization and metadata enrichment"
```
