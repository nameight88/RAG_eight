"""FSSRagSystem 싱글턴 래퍼."""
import sys
import os
from typing import Any, Dict, Optional

# 프로젝트 루트를 sys.path에 추가 (rag_system.py 접근)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


class RagService:
    """FSSRagSystem을 싱글턴으로 래핑한 서비스."""

    def __init__(self, vector_db_path: str, use_faiss: bool = True):
        from rag_system import FSSRagSystem
        self._rag_system = FSSRagSystem(
            vector_db_path=vector_db_path,
            use_openai_embeddings=True,
            use_faiss=use_faiss,
        )

    def query(
        self,
        question: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        institution: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """RAG 질의 수행 후 answer와 sources 반환."""
        return self._rag_system.answer_question(
            question=question,
            date_from=date_from,
            date_to=date_to,
            institution=institution,
            doc_type=doc_type,
        )


# 싱글턴 인스턴스 (lifespan에서 초기화)
_rag_service: Optional[RagService] = None


def get_rag_service() -> RagService:
    if _rag_service is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="RAG 서비스가 초기화되지 않았습니다. OPENAI_API_KEY 환경변수를 설정하세요.",
        )
    return _rag_service


def init_rag_service(vector_db_path: str, use_faiss: bool = True) -> RagService:
    global _rag_service
    _rag_service = RagService(vector_db_path, use_faiss)
    return _rag_service
