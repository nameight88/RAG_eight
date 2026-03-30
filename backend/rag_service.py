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
        # 상대경로는 프로젝트 루트 기준 절대경로로 변환
        if not os.path.isabs(vector_db_path):
            vector_db_path = os.path.join(_root, vector_db_path)
        self._rag_system = FSSRagSystem(
            vector_db_path=vector_db_path,
            use_openai_embeddings=True,
            use_faiss=use_faiss,
            use_openai_llm=True,
        )
        self._rag_system.initialize_llm()

    @property
    def db_type(self) -> str:
        return self._rag_system.db_type

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


# doc_type → RagService 매핑 (시작 시 채워짐)
_rag_services: Dict[str, RagService] = {}


def get_rag_service(doc_type: Optional[str] = None) -> RagService:
    """doc_type에 맞는 서비스를 반환한다. 없으면 기본(sanctions) 서비스를 반환."""
    if not _rag_services:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="RAG 서비스가 초기화되지 않았습니다. OPENAI_API_KEY 환경변수를 설정하세요.",
        )
    if doc_type and doc_type in _rag_services:
        return _rag_services[doc_type]
    return _rag_services.get('sanctions') or next(iter(_rag_services.values()))


def init_rag_service(vector_db_path: str, use_faiss: bool = True) -> RagService:
    """RagService를 초기화하고 db_type 키로 등록한다."""
    svc = RagService(vector_db_path, use_faiss)
    _rag_services[svc.db_type] = svc
    return svc
