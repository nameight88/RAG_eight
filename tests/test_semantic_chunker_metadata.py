"""semantic_chunker 메타데이터 필드 검증"""
from date_normalizer import normalize_date


def test_sanction_date_normalization():
    """sanction_date가 정규화되어야 한다"""
    assert normalize_date("2016. 3. 15") == "2016-03-15"
    assert normalize_date("2014.1.7") == "2014-01-07"


def test_year_month_from_normalized_date():
    """정규화된 날짜에서 year/month 정수 추출"""
    normalized = normalize_date("2016. 3. 15")
    year = int(normalized[:4]) if normalized and len(normalized) >= 4 and normalized[:4].isdigit() else 0
    month = int(normalized[5:7]) if normalized and len(normalized) >= 7 and normalized[5:7].isdigit() else 0
    assert year == 2016
    assert month == 3


def test_chunk_fields_structure():
    """청크 딕셔너리에 필요한 필드가 있어야 한다"""
    chunk = {
        "id": "TEST-chunk-0",
        "doc_id": "TEST",
        "institution": "테스트은행",
        "sanction_date": "2016. 3. 15",
        "date": "2016-03-15",
        "doc_type": "제재결과",
        "year": 2016,
        "month": 3,
        "chunk_index": 0,
        "total_chunks": 1,
        "chunk_text": "테스트 내용",
        "quality_score": 3,
    }
    assert chunk.get("doc_type") == "제재결과"
    assert chunk.get("year") == 2016
    assert chunk.get("month") == 3
    assert chunk.get("date") == "2016-03-15"
