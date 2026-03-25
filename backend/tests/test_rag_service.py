"""RAG 서비스 래퍼 테스트"""
from unittest.mock import MagicMock


def test_rag_service_returns_answer_and_sources():
    """RAG 서비스가 answer와 sources를 반환해야 한다"""
    from backend.rag_service import RagService

    mock_result = {
        "answer": "전자금융 관련 제재 결과입니다.",
        "sources": [
            {
                "content": "전자금융 위반 사항",
                "metadata": {
                    "institution": "국민은행",
                    "date": "2023-01-15",
                    "doc_type": "제재결과",
                },
                "score": 0.89,
            }
        ],
    }

    service = RagService.__new__(RagService)
    service._rag_system = MagicMock()
    service._rag_system.answer_question.return_value = mock_result

    result = service.query("전자금융 제재 알려줘")

    assert result["answer"] == "전자금융 관련 제재 결과입니다."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["metadata"]["institution"] == "국민은행"


def test_rag_service_query_passes_filters():
    """필터 파라미터가 answer_question에 전달돼야 한다"""
    from backend.rag_service import RagService

    service = RagService.__new__(RagService)
    service._rag_system = MagicMock()
    service._rag_system.answer_question.return_value = {"answer": "결과", "sources": []}

    service.query(
        "질문",
        date_from="2023-01-01",
        institution="신한은행",
        doc_type="sanctions",
    )

    call_kwargs = service._rag_system.answer_question.call_args
    assert call_kwargs.kwargs.get("date_from") == "2023-01-01"
    assert call_kwargs.kwargs.get("institution") == "신한은행"
