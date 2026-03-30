"""
금융감독원 제재/경영유의사항 코퍼스를 활용한 RAG 질의응답 시스템
- 벡터 저장소에서 관련 문서 검색
- LLM을 이용한 질의응답
"""

import os
import json
import re
import torch
import pickle  # pickle 모듈 추가
import numpy as np  # numpy 모듈 추가
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from rag_filters import _apply_explicit_filters

# BM25 Hybrid Search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# BGE Reranker
try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

# FAISS 관련 임포트
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️ FAISS를 임포트할 수 없습니다. pip install faiss-cpu를 실행하여 설치해주세요.")
    FAISS_AVAILABLE = False

# Pydantic 호환성을 위한 커스텀 Unpickler 클래스 추가
class PydanticCompatibleUnpickler(pickle.Unpickler):
    """Pydantic v1/v2 호환성을 위한 커스텀 Unpickler"""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            # Pydantic 관련 클래스 처리
            if module == "pydantic.main" and name == "BaseModel":
                import pydantic
                return pydantic.BaseModel
            elif module == "langchain.schema" and name == "Document":
                from langchain_core.documents import Document
                return Document
            elif module == "langchain.docstore.document" and name == "Document":
                from langchain_core.documents import Document
                return Document
            elif module == "langchain.docstore.in_memory" and name == "InMemoryDocstore":
                from langchain_community.docstore.in_memory import InMemoryDocstore
                return InMemoryDocstore
            else:
                # 기타 클래스는 동적으로 처리
                try:
                    import importlib
                    mod = importlib.import_module(module)
                    return getattr(mod, name)
                except:
                    # 최후의 수단: 빈 클래스 반환
                    class DummyClass:
                        def __init__(self, *args, **kwargs):
                            pass
                        def __getstate__(self):
                            return {}
                        def __setstate__(self, state):
                            pass
                    return DummyClass

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("unsupported persistent object")

# .env 파일에서 환경 변수 로드
load_dotenv()


class FSSRagSystem:
    """금융감독원 제재/경영유의사항 RAG 시스템"""
    
    # 벡터 저장소 캐시 - 경로별로 저장
    _vector_store_cache = {}
    # 임베딩 모델 캐시
    _embeddings_cache = {}
    
    def __init__(
        self,
        vector_db_path: str = "./data/vector_db/fss_sanctions",
        embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 5,
        use_anthropic: bool = False,
        use_openai_embeddings: bool = True,
        use_faiss: bool = True,
        use_openai_llm: bool = True,
        create_from_json: str = None,  # JSON 파일 경로 추가
    ) -> None:
        """FSS RAG 시스템 초기화"""
        
        # 설정 저장
        self.vector_db_path = os.path.abspath(vector_db_path) if vector_db_path else None
        self.embed_model_name = embed_model_name
        self.top_k = top_k
        self.use_anthropic = use_anthropic  # Anthropic Claude 사용 여부
        self.anthropic_api_key = os.getenv("ANTHROPIC_APIKEY")
        self.use_openai_llm = use_openai_llm  # OpenAI LLM 사용 여부
        self.use_openai_embeddings = use_openai_embeddings  # OpenAI 임베딩 사용 여부
        self.use_faiss = use_faiss  # FAISS 사용 여부 (False면 Chroma 사용)
        self.create_from_json = create_from_json  # JSON 파일에서 생성
        
        # 제재 데이터인지 경영유의인지 판단
        if create_from_json:
            if "sanctions" in create_from_json:
                self.db_type = "sanctions"
            elif "management" in create_from_json:
                self.db_type = "management"
            else:
                self.db_type = "unknown"
        elif vector_db_path:
            if "sanctions" in vector_db_path:
                self.db_type = "sanctions"
            elif "management" in vector_db_path:
                self.db_type = "management"
            else:
                self.db_type = "unknown"
        else:
            self.db_type = "unknown"
            
        print(f"🔄 DB 타입: {self.db_type}")
        
        # OpenAI 설정
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_model_name = "gpt-4o"  # 기본 모델
        
        # 초기화
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        # BM25 Hybrid Search
        self.bm25_index = None        # BM25Okapi 인덱스
        self.bm25_corpus = []         # 토큰화된 말뭉치 (검색용)
        self.bm25_docs = []           # 원본 Document 객체 목록
        self._reranker = None         # BGE Cross-encoder Reranker (lazy load)
        
        # JSON 파일에서 생성하는 경우
        if self.create_from_json:
            self.create_vector_store_from_json()
        # 기존 벡터 저장소 로드
        elif self.vector_db_path:
            self.load_vector_store()
    
    def get_embeddings(self):
        """임베딩 모델 가져오기 (캐시 활용)"""
        cache_key = f"openai_{self.openai_api_key}" if self.use_openai_embeddings else self.embed_model_name
        
        # 캐시에 이미 있는지 확인
        if cache_key in FSSRagSystem._embeddings_cache:
            print(f"📚 캐시된 임베딩 모델 사용: {cache_key}")
            return FSSRagSystem._embeddings_cache[cache_key]
        
        try:
            # OpenAI API 사용
            if self.use_openai_embeddings and self.openai_api_key:
                print(f"🧠 OpenAI 임베딩 API 초기화 중...")
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small", 
                    openai_api_key=self.openai_api_key
                )
                print(f"✅ OpenAI 임베딩 초기화 완료")
            # 로컬/HuggingFace 모델 사용
            else:
                print(f"🧠 HuggingFace 임베딩 모델 초기화 중: {self.embed_model_name}")
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.embed_model_name,
                    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                print(f"✅ 임베딩 모델 초기화 완료")
                
            # 캐시에 저장
            FSSRagSystem._embeddings_cache[cache_key] = embeddings
            
            return embeddings
            
        except Exception as e:
            print(f"❌ 임베딩 모델 초기화 실패: {e}")
            return None
    
    def load_faiss_from_local(self, local_path: str) -> Any:
        """로컬 저장소에서 FAISS 로드"""
        try:
            print(f"✅ 기존 FAISS 벡터 저장소를 로드합니다: {local_path}")
            
            # 보안 옵션 추가: allow_dangerous_deserialization=True
            faiss_vectorstore = FAISS.load_local(
                local_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # 안전하지 않은 역직렬화 허용 (직접 생성한 안전한 파일임)
            )
            return faiss_vectorstore
            
        except Exception as e:
            print(f"❌ FAISS 벡터 저장소 로드 실패: {e}")
            return None
    
    def load_vector_store(self):
        """벡터 저장소 로드 (메타데이터 기반)"""
        try:
            print(f"📚 벡터 저장소 로드 중: {self.vector_db_path}")

            # FAISS 사용 가능 여부 확인
            if self.use_faiss and not FAISS_AVAILABLE:
                print("⚠️ FAISS를 사용할 수 없어 Chroma로 전환합니다.")
                self.use_faiss = False

            # 벡터 저장소 정보 파일 경로
            info_path = os.path.join(self.vector_db_path, 'vector_store_info.json')
            if not os.path.exists(info_path):
                print(f"❌ 'vector_store_info.json' 파일을 찾을 수 없습니다: {info_path}")
                print("오류: 벡터 저장소의 메타데이터가 없어 임베딩 모델을 확인할 수 없습니다.")
                print("데이터 생성 파이프라인을 다시 실행하여 벡터 저장소를 재생성해주세요.")
                return False

            with open(info_path, 'r', encoding='utf-8') as f:
                vs_info = json.load(f)
            
            use_openai = vs_info.get('use_openai', False)
            embed_model = vs_info.get('embed_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            if isinstance(embed_model, str):
                embed_model = embed_model.replace("openai/", "")

            # OpenAI API 키 확인 및 설정
            if use_openai:
                print(f"🧠 OpenAI 임베딩 API 초기화 중 ({embed_model})...")
                if not self.openai_api_key:
                    self.openai_api_key = os.getenv("OPENAI_API_KEY")
                    if not self.openai_api_key:
                        print("❌ OpenAI API 키가 설정되지 않았습니다.")
                        return False
                
                try:
                    from langchain_openai import OpenAIEmbeddings
                    self.embeddings = OpenAIEmbeddings(
                        model=embed_model,
                        openai_api_key=self.openai_api_key,
                        show_progress_bar=True,
                        request_timeout=60
                    )
                    # 임베딩 테스트
                    test_text = "테스트"
                    try:
                        test_embedding = self.embeddings.embed_query(test_text)
                        print(f"✅ OpenAI 임베딩 테스트 성공 (벡터 크기: {len(test_embedding)})")
                    except Exception as e:
                        print(f"❌ OpenAI 임베딩 테스트 실패: {str(e)}")
                        return False
                except Exception as e:
                    print(f"❌ OpenAI 임베딩 초기화 실패: {str(e)}")
                    return False
            else:
                print(f"🧠 HuggingFace 임베딩 초기화 중: {embed_model}")
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embed_model,
                    model_kwargs={'device': 'cpu'},  # CUDA 오류 방지
                    encode_kwargs={'normalize_embeddings': True}
                )

            # 벡터 저장소 로드 (FAISS 또는 Chroma)
            vector_store_type = vs_info.get('vector_store_type', 'FAISS' if self.use_faiss else 'Chroma').upper()

            if vector_store_type == 'FAISS' and FAISS_AVAILABLE:
                faiss_path = os.path.join(self.vector_db_path, "faiss")
                index_path = os.path.join(faiss_path, "index.faiss")
                docstore_path = os.path.join(faiss_path, "index.pkl")
                
                if not os.path.exists(index_path) or not os.path.exists(docstore_path):
                    print(f"❌ FAISS 인덱스 파일을 찾을 수 없습니다: {faiss_path}")
                    return False
                
                try:
                    print(f"✅ 기존 FAISS 벡터 저장소를 로드합니다: {faiss_path}")
                    
                    # 방법 1: 표준 load_local 시도
                    try:
                        from langchain_community.vectorstores import FAISS
                        self.vector_store = FAISS.load_local(
                            faiss_path,
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        print("✅ FAISS 벡터 저장소 로드 완료 (표준 방법)")
                    except (KeyError, AttributeError) as e:
                        if '__fields_set__' in str(e) or 'pydantic' in str(e).lower():
                            print("⚠️ Pydantic 호환성 문제 감지, 커스텀 로더 사용...")
                            
                            # 방법 2: 커스텀 로더 사용
                            import faiss
                            from langchain_community.docstore.in_memory import InMemoryDocstore
                            from langchain_community.vectorstores import FAISS
                            
                            # FAISS 인덱스 로드
                            index = faiss.read_index(index_path)
                            
                            # JSON 파일에서 문서 로드
                            json_filename = "fss_sanctions_parsed.json" if "sanctions" in self.vector_db_path else "fss_management_parsed.json"
                            json_path = os.path.join(self.vector_db_path, json_filename)
                            
                            if os.path.exists(json_path):
                                print(f"📄 JSON 파일에서 문서 로드 중: {json_path}")
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                
                                # 문서 생성
                                from langchain_core.documents import Document
                                documents = []
                                
                                # 데이터 구조 확인
                                if isinstance(data, dict) and 'documents' in data:
                                    docs_list = data['documents']
                                elif isinstance(data, list):
                                    docs_list = data
                                else:
                                    print("❌ 알 수 없는 JSON 데이터 구조")
                                    return False
                                
                                for doc in docs_list:
                                    if not isinstance(doc, dict):
                                        continue
                                        
                                    # 텍스트 추출
                                    content = doc.get('content', {})
                                    if isinstance(content, dict):
                                        # 제재 정보의 경우
                                        full_text = content.get('full_text', '')
                                        if not full_text:
                                            # 상세 내용 구성
                                            sanction_facts = content.get('sanction_facts', [])
                                            facts_text = ""
                                            for fact in sanction_facts:
                                                if isinstance(fact, dict):
                                                    facts_text += f"\n- {fact.get('title', '')}: {fact.get('content', '')}"
                                            
                                            fine_info = content.get('fine', {})
                                            if isinstance(fine_info, dict):
                                                fine_text = fine_info.get('text', '')
                                            else:
                                                fine_text = str(fine_info)
                                            
                                            full_text = f"제재사실:\n{facts_text}\n\n제재내용: {content.get('sanction_type', '')}\n{fine_text}\n{content.get('executive_sanction', '')}"
                                        text = full_text
                                    else:
                                        text = str(content)
                                    
                                    # 메타데이터 구성
                                    metadata = {
                                        'institution': doc.get('institution', ''),
                                        'doc_id': doc.get('doc_id', ''),
                                    }
                                    
                                    # 문서 타입 설정
                                    if "sanctions" in self.vector_db_path:
                                        metadata['doc_type'] = '제재정보'
                                        if isinstance(content, dict):
                                            metadata['sanction_type'] = content.get('sanction_type', '')
                                    else:
                                        metadata['doc_type'] = '경영유의사항'
                                        if isinstance(content, dict):
                                            metadata['management_type'] = content.get('management_type', '')
                                    
                                    # 날짜 필드 추가 (date_normalizer로 정규화)
                                    from date_normalizer import normalize_date as _norm_date
                                    raw_date = (doc.get('date') or doc.get('sanction_date') or doc.get('disclosure_date') or '')
                                    norm_date = _norm_date(raw_date) or raw_date
                                    metadata['date'] = norm_date
                                    if norm_date and len(norm_date) >= 7:
                                        try:
                                            metadata['year'] = int(norm_date[:4])
                                            metadata['month'] = int(norm_date[5:7])
                                        except ValueError:
                                            metadata['year'] = 0
                                            metadata['month'] = 0
                                    else:
                                        metadata['year'] = 0
                                        metadata['month'] = 0
                                    
                                    # 추가 메타데이터
                                    doc_metadata = doc.get('metadata', {})
                                    if isinstance(doc_metadata, dict):
                                        # 규정 정보 추가
                                        if 'regulations' in doc_metadata:
                                            metadata['regulations'] = doc_metadata['regulations']
                                        
                                        # 기타 메타데이터 복사
                                        for key, value in doc_metadata.items():
                                            if key not in metadata and value:
                                                metadata[key] = value
                                    
                                    if text.strip():  # 빈 텍스트는 제외
                                        print(f"📄 문서 로드: {metadata['institution']} ({metadata['date']})")
                                        documents.append(Document(page_content=text, metadata=metadata))
                                
                                print(f"📄 {len(documents)}개의 문서를 로드했습니다.")
                                
                                # 문서 임베딩 생성
                                print("🔄 문서 임베딩 생성 중...")
                                texts = [doc.page_content for doc in documents]
                                metadatas = [doc.metadata for doc in documents]
                                
                                # FAISS 벡터 저장소 생성
                                self.vector_store = FAISS.from_texts(
                                    texts,
                                    self.embeddings,
                                    metadatas=metadatas
                                )
                                print("✅ 벡터 저장소 재구성 완료")
                            else:
                                print(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
                                return False
                        else:
                            raise e
                    
                    # 벡터 저장소 테스트
                    try:
                        test_query = "테스트"
                        test_results = self.vector_store.similarity_search(test_query, k=1)
                        if test_results:
                            print(f"✅ 벡터 저장소 검색 테스트 성공 (결과 수: {len(test_results)})")
                            # 첫 번째 결과의 메타데이터 출력
                            print(f"📄 테스트 문서 메타데이터: {test_results[0].metadata}")
                        else:
                            print("⚠️ 벡터 저장소 검색 결과가 없습니다.")
                    except Exception as test_error:
                        print(f"⚠️ 벡터 저장소 테스트 중 오류: {test_error}")
                        import traceback
                        traceback.print_exc()
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ FAISS 로드 실패: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False
                    
            elif vector_store_type == 'CHROMA' or (vector_store_type == 'FAISS' and not FAISS_AVAILABLE):
                chroma_path = self.vector_db_path
                print(f"✅ 기존 Chroma 벡터 저장소를 로드합니다: {chroma_path}")
                try:
                    from langchain_community.vectorstores import Chroma
                    self.vector_store = Chroma(
                        persist_directory=chroma_path,
                        embedding_function=self.embeddings
                    )
                except Exception as e:
                    print(f"❌ Chroma 벡터 저장소 로드 실패: {e}")
                    return False
            else:
                print(f"❌ 알 수 없는 벡터 저장소 타입입니다: {vector_store_type}")
                return False

            print(f"✅ 벡터 저장소 로드 완료")
            self.check_vector_store()
            return True

        except Exception as e:
            print(f"❌ 벡터 저장소 로드 중 치명적 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def check_vector_store(self):
        """벡터 저장소 상태 및 기능 확인"""
        try:
            print("🔍 벡터 저장소 상태 확인 중...")
            
            # 메서드 확인
            methods = [
                method for method in dir(self.vector_store)
                if callable(getattr(self.vector_store, method)) and not method.startswith("_")
            ]
            print(f"✅ 사용 가능한 메서드: {', '.join(methods[:5])}... (총 {len(methods)}개)")
            
            # 간단한 검색 테스트
            try:
                print("🔍 간단한 검색 테스트 중...")
                test_query = "금융"
                if hasattr(self.vector_store, "similarity_search") and callable(getattr(self.vector_store, "similarity_search")):
                    results = self.vector_store.similarity_search(test_query, k=1)
                    if results:
                        print(f"✅ 테스트 검색 성공: {len(results)}개 결과")
                        # 첫 번째 결과 메타데이터 확인
                        if results[0].metadata:
                            print(f"📄 메타데이터 키: {', '.join(list(results[0].metadata.keys()))}")
                    else:
                        print("⚠️ 테스트 검색 결과 없음")
                else:
                    print("⚠️ similarity_search 메서드 없음")
            except Exception as e:
                print(f"❌ 테스트 검색 실패: {str(e)}")

            # BM25 인덱스 로드 (없으면 FAISS docstore에서 빌드)
            if BM25_AVAILABLE:
                faiss_dir = os.path.join(self.vector_db_path, "faiss")
                if not self._load_bm25(faiss_dir):
                    print("🔄 BM25 인덱스 없음 → FAISS docstore에서 빌드 중...")
                    try:
                        all_docs = list(self.vector_store.docstore._dict.values())
                        self._build_bm25(all_docs)
                        self._save_bm25(faiss_dir)
                    except Exception as e:
                        print(f"⚠️ BM25 빌드 실패: {e}")

        except Exception as e:
            print(f"❌ 벡터 저장소 확인 중 오류: {str(e)}")

    # ── BM25 Hybrid Search ──────────────────────────────────────────────

    @staticmethod
    def _tokenize_ko(text: str) -> List[str]:
        """한국어 토크나이징: 공백 분리 + 2~4자 문자 n-gram 추가 (konlpy 없이 동작)"""
        tokens = text.lower().split()
        # 2~4자 n-gram 추가 (조항번호·기관명 부분 매칭용)
        ngrams = []
        for token in tokens:
            for n in range(2, min(5, len(token) + 1)):
                for i in range(len(token) - n + 1):
                    ngrams.append(token[i:i + n])
        return tokens + ngrams

    def _build_bm25(self, docs) -> None:
        """FAISS docstore의 Document 목록으로 BM25 인덱스 빌드"""
        if not BM25_AVAILABLE:
            return
        self.bm25_docs = list(docs)
        self.bm25_corpus = [self._tokenize_ko(d.page_content) for d in self.bm25_docs]
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        print(f"✅ BM25 인덱스 빌드 완료 ({len(self.bm25_docs)}개 청크)")

    def _save_bm25(self, save_dir: str) -> None:
        """BM25 인덱스를 파일로 저장"""
        if not self.bm25_index:
            return
        path = os.path.join(save_dir, "bm25_index.pkl")
        with open(path, "wb") as f:
            pickle.dump({"index": self.bm25_index, "corpus": self.bm25_corpus, "docs": self.bm25_docs}, f)
        print(f"✅ BM25 인덱스 저장: {path}")

    def _load_bm25(self, save_dir: str) -> bool:
        """저장된 BM25 인덱스 로드"""
        path = os.path.join(save_dir, "bm25_index.pkl")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.bm25_index = data["index"]
            self.bm25_corpus = data["corpus"]
            self.bm25_docs = data["docs"]
            print(f"✅ BM25 인덱스 로드 완료 ({len(self.bm25_docs)}개 청크)")
            return True
        except Exception as e:
            print(f"⚠️ BM25 인덱스 로드 실패: {e}")
            return False

    def _hybrid_search(self, query: str, k: int = 50) -> List[Dict[str, Any]]:
        """BM25 + Dense 검색 결과를 RRF로 합산"""
        RRF_K = 60  # RRF 상수 (일반적으로 60 사용)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}

        # ── Dense 검색 (FAISS) ──
        try:
            dense_results = self.vector_store.similarity_search_with_score(query, k=k)
            for rank, (doc, dist) in enumerate(dense_results):
                doc_id = doc.metadata.get("doc_id", "") + doc.page_content[:30]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)
                doc_map[doc_id] = {"doc": doc, "dense_score": float(max(0.0, 1 - (dist ** 2) / 2))}
        except Exception as e:
            print(f"⚠️ Dense 검색 실패: {e}")

        # ── BM25 검색 ──
        if self.bm25_index and self.bm25_docs:
            try:
                tokens = self._tokenize_ko(query)
                bm25_scores = self.bm25_index.get_scores(tokens)
                # 상위 k개 인덱스
                top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
                for rank, idx in enumerate(top_indices):
                    if bm25_scores[idx] <= 0:
                        break
                    doc = self.bm25_docs[idx]
                    doc_id = doc.metadata.get("doc_id", "") + doc.page_content[:30]
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)
                    if doc_id not in doc_map:
                        doc_map[doc_id] = {"doc": doc, "dense_score": 0.0}
                    doc_map[doc_id]["bm25_score"] = float(bm25_scores[idx])
            except Exception as e:
                print(f"⚠️ BM25 검색 실패: {e}")

        # ── RRF 점수로 정렬 ──
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, rrf in ranked:
            entry = doc_map[doc_id]
            doc = entry["doc"]
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": rrf,
                "dense_score": entry.get("dense_score", 0.0),
                "bm25_score": entry.get("bm25_score", 0.0),
            })
        return results

    def _hyde_query(self, query: str) -> str:
        """HyDE: LLM으로 가상 문서 생성 → 임베딩 검색에 활용"""
        if not self.llm:
            return query
        try:
            hyde_prompt = (
                "당신은 금융감독원 제재 전문가입니다. "
                "아래 질문에 대한 전형적인 금융감독원 제재 문서를 한국어로 2~3문장 작성하세요. "
                "실제 제재 결과물처럼 기관명, 날짜, 위반 내용, 제재 조치를 포함하세요.\n\n"
                f"질문: {query}\n\n가상 문서:"
            )
            result = self.llm.invoke(hyde_prompt)
            hypothetical = result.content if hasattr(result, "content") else str(result)
            print(f"💡 HyDE 가상 문서: {hypothetical[:100]}...")
            return hypothetical.strip()
        except Exception as e:
            print(f"⚠️ HyDE 생성 실패 (원본 쿼리 사용): {e}")
            return query

    def _rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """BGE Cross-encoder로 결과 재정렬"""
        if not RERANKER_AVAILABLE or not results:
            return results[:top_k]
        try:
            if not hasattr(self, "_reranker") or self._reranker is None:
                print("🔄 BGE Reranker 로드 중 (BAAI/bge-reranker-v2-m3)...")
                self._reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
                print("✅ BGE Reranker 로드 완료")
            pairs = [[query, r["content"]] for r in results]
            scores = self._reranker.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [scores]
            for item, s in zip(results, scores):
                item["rerank_score"] = float(s)
            reranked = sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            print(f"✅ Rerank 완료: top 점수 {reranked[0]['rerank_score']:.3f}")
            return reranked[:top_k]
        except Exception as e:
            print(f"⚠️ Rerank 실패 (원본 순서 사용): {e}")
            return results[:top_k]

    # ────────────────────────────────────────────────────────────────────

    def initialize_llm(self) -> None:
        """LLM 초기화"""
        try:
            # Anthropic Claude API 사용
            if self.use_anthropic:
                try:
                    # Anthropic API 키 확인
                    anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_APIKEY")
                    if not anthropic_api_key:
                        print("❌ Anthropic API 키가 설정되지 않았습니다.")
                        return
                    
                    print("🧠 Anthropic Claude API 초기화 중...")
                    try:
                        # 신규 버전 import 시도
                        try:
                            from langchain_anthropic import ChatAnthropic
                            
                            # LLM 초기화
                            self.llm = ChatAnthropic(
                                model="claude-3-opus-20240229",  # 최신 Claude 모델
                                anthropic_api_key=anthropic_api_key
                            )
                        except ImportError:
                            # 기존 버전 fallback
                            print("⚠️ langchain_anthropic 임포트 실패, 기존 방식 시도...")
                            from langchain.chat_models import ChatAnthropic
                            self.llm = ChatAnthropic(
                                model_name="claude-3-opus-20240229",  # 최신 Claude 모델
                                anthropic_api_key=anthropic_api_key
                            )
                        
                        print("✅ Anthropic Claude API 초기화 완료")
                    except Exception as anthro_error:
                        print(f"❌ Anthropic 초기화 실패: {anthro_error}")
                        return
                    
                except Exception as e:
                    print(f"❌ Anthropic API 초기화 실패: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return
            # OpenAI API 사용
            else:
                try:
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    if not openai_api_key:
                        print("❌ OpenAI API 키가 설정되지 않았습니다.")
                        return
                    
                    print(f"🧠 OpenAI API 초기화 중: {self.llm_model_name}...")
                    
                    # 모델 이름 호환성 확인
                    model_name = self.llm_model_name
                    
                    # 임포트 시도
                    try:
                        # 신규 버전 import
                        from langchain_openai import ChatOpenAI
                        
                        # LLM 초기화
                        self.llm = ChatOpenAI(
                            model=model_name,
                            temperature=0.3,
                            openai_api_key=openai_api_key
                        )
                    except ImportError:
                        # 기존 버전 fallback
                        print("⚠️ langchain_openai 임포트 실패, 기존 방식 시도...")
                        from langchain.chat_models import ChatOpenAI
                        self.llm = ChatOpenAI(
                            model_name=model_name,
                            temperature=0.3,
                            openai_api_key=openai_api_key
                        )
                    
                    print("✅ OpenAI API 초기화 완료")
                    
                except Exception as e:
                    print(f"❌ OpenAI API 초기화 실패: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return
            
            # QA 체인 설정
            self.setup_qa_chain()
            
        except Exception as e:
            print(f"❌ LLM 초기화 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()

    def setup_qa_chain(self) -> None:
        """QA 체인 설정"""
        try:
            # 벡터 저장소와 LLM이 모두 초기화되었는지 확인
            if not self.vector_store:
                print("❌ 벡터 저장소가 초기화되지 않았습니다.")
                return
            
            if not self.llm:
                print("❌ LLM이 초기화되지 않았습니다.")
                return
            
            # QA 체인을 직접 구성하지 않고 검색 과정을 별도로 관리
            print("✅ QA 체인 생성 완료")
            self.qa_chain = True  # 더미 값, QA 체인이 준비되었다는 표시용
            
        except Exception as e:
            print(f"❌ QA 체인 설정 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            self.qa_chain = None
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """메타데이터가 필터 조건에 맞는지 확인"""
        if not filters:
            return True
        
        # 기관 유형 필터링
        if 'institution_types' in filters and filters['institution_types']:
            institution = metadata.get('institution', '').lower()
            found_match = False
            for inst_type in filters['institution_types']:
                if inst_type.lower() in institution:
                    found_match = True
                    break
            if not found_match:
                return False
        
        # 날짜 필터링
        if 'date_filter' in filters and 'date_value' in filters:
            date_str = metadata.get('date', '')
            if not date_str:
                # 다른 날짜 관련 필드 확인
                date_str = metadata.get('sanction_date', '')
                if not date_str:
                    date_str = metadata.get('disclosure_date', '')
                
                # 여전히 날짜 정보가 없는 경우
                if not date_str:
                    print(f"⚠️ 날짜 정보 없음: {metadata}")
                    return False
                
            # 날짜 형식 정규화 (YYYY.MM.DD 또는 YYYY-MM-DD)
            date_str = date_str.replace('-', '.').strip()
            
            # 연도만 추출
            year_match = re.search(r'(20\d{2})', date_str)
            if not year_match:
                print(f"⚠️ 날짜 형식 인식 불가: {date_str}")
                return False
                
            document_year = year_match.group(1)
            filter_year = filters['date_value']
            
            # 최근 1년 필터링 (예: 2023년 이상)
            if len(filter_year) == 4 and filter_year.isdigit():
                if int(document_year) < int(filter_year):
                    return False
            
            print(f"✅ 날짜 매칭: 문서={document_year}, 필터={filter_year}")
        
        return True

    def preprocess_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """질문 전처리 및 필터 추출"""
        processed_query = query
        
        # 필터 초기화
        filters = {}
        
        # 은행/보험사/증권사 필터링
        institution_types = []
        if '은행' in query:
            institution_types.append('은행')
        if '보험' in query:
            institution_types.append('보험')
        if '증권' in query:
            institution_types.append('증권')
        if '카드' in query:
            institution_types.append('카드')
        if '금융' in query:
            institution_types.append('금융')
        
        if institution_types:
            filters['institution_types'] = institution_types
        
        # 날짜 필터링 (최근 1년, 올해, 2023년 등)
        date_filter = None
        if '최근 1년' in query or '지난 1년' in query:
            date_filter = 'date'
            # 현재 연도를 사용
            current_year = datetime.now().year
            date_value = str(current_year - 1)  # 1년 전부터
            filters['date_filter'] = date_filter
            filters['date_value'] = date_value
            print(f"📅 날짜 필터링: {date_value}년부터")
        elif '올해' in query:
            date_filter = 'date'
            date_value = str(datetime.now().year)
            filters['date_filter'] = date_filter
            filters['date_value'] = date_value
            print(f"📅 날짜 필터링: {date_value}년")
        else:
            # 연도 추출 (YYYY년)
            year_match = re.search(r'(20\d{2})년', query)
            if year_match:
                date_filter = 'date'
                date_value = year_match.group(1)
                filters['date_filter'] = date_filter
                filters['date_value'] = date_value
                print(f"📅 날짜 필터링: {date_value}년")
        
        # 문서 유형 필터링
        doc_type_filter = None
        if '경영유의' in query or '경영 유의' in query:
            doc_type_filter = 'management'
        elif '제재' in query or '징계' in query or '과태료' in query or '과징금' in query:
            doc_type_filter = 'sanctions'

        if doc_type_filter:
            filters['doc_type'] = doc_type_filter
        
        return processed_query, filters
    
    def answer_question(
        self,
        question: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        institution: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """질문에 답변"""
        try:
            # 벡터 저장소 체크
            if not self.vector_store:
                return {
                    "answer": "벡터 저장소가 로드되지 않았습니다. 먼저 벡터 저장소를 로드해주세요.",
                    "sources": []
                }

            # LLM 체크
            if not self.llm:
                return {
                    "answer": "LLM이 초기화되지 않았습니다. 사이드바에서 'LLM 초기화' 버튼을 클릭해주세요.",
                    "sources": []
                }

            # 질문 전처리
            print(f"❓ 질문 처리: '{question}'")
            processed_query, filters = self.preprocess_query(question)
            filters = _apply_explicit_filters(filters, date_from, date_to, institution, doc_type)
            
            # 문서 유형 필터 확인
            if 'doc_type' in filters and filters['doc_type'] != self.db_type:
                if filters['doc_type'] == 'management':
                    return {
                        "answer": "현재 제재 DB가 선택되어 있습니다. 경영유의사항에 대해 질문하시려면 DB를 변경해주세요.",
                        "sources": []
                    }
                else:
                    return {
                        "answer": "현재 경영유의 DB가 선택되어 있습니다. 제재 정보에 대해 질문하시려면 DB를 변경해주세요.",
                        "sources": []
                    }
            
            if filters:
                print(f"🔍 추출된 필터: {filters}")
            
            # 문서 검색 수행 (날짜/기관 필터가 있으면 더 많이 검색)
            search_k = 10 if filters else 5
            search_results = self.search_documents(processed_query, k=search_k)
            
            # 검색 결과가 없는 경우
            if not search_results:
                print("⚠️ 검색 결과 없음")
                return {
                    "answer": "질문과 관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                    "sources": []
                }
            
            # 검색된 문서들로 컨텍스트 구성
            context = ""
            sources = []
            
            for idx, doc in enumerate(search_results):
                try:
                    metadata = doc.get("metadata", {})
                    content = doc.get("content", "")
                    
                    # 유효한 메타데이터 확인
                    institution = metadata.get("institution", "")
                    if not institution:
                        institution = "미상"
                        
                    # 날짜 필드 확인
                    date = metadata.get("date", "")
                    if not date:
                        date = metadata.get("sanction_date", "")
                    if not date:
                        date = metadata.get("disclosure_date", "")
                    if not date:
                        date = "날짜 미상"
                    
                    # 문서 타입 확인
                    doc_type = metadata.get("doc_type", "")
                    if not doc_type and self.db_type == "sanctions":
                        doc_type = "제재정보"
                    elif not doc_type:
                        doc_type = "경영유의사항"
                    
                    context += f"[문서 {idx+1}]\n"
                    context += f"기관: {institution}\n"
                    context += f"날짜: {date}\n"
                    context += f"유형: {doc_type}\n"
                    context += f"내용:\n{content}\n\n"
                    
                    sources.append({
                        "content": content,
                        "metadata": metadata,
                        "score": doc.get("score", 1.0)
                    })
                except Exception as doc_error:
                    print(f"⚠️ 문서 처리 오류 (무시됨): {doc_error}")
                    continue
            
            # LLM으로 답변 생성
            try:
                # 컨텍스트가 너무 길면 자르기
                if len(context) > 12000:
                    print(f"⚠️ 컨텍스트가 너무 깁니다: {len(context)}자 → 12000자로 자릅니다")
                    context = context[:12000] + "..."
                    
                prompt = f"""당신은 금융감독원 제재 및 경영유의 정보 전문 분석가입니다.
아래는 검색된 금융감독원 제재/경영유의 자료입니다:

{context}

질문: {question}

답변 지침:
1. 위 자료에 있는 내용을 최대한 활용하여 구체적으로 답변하세요.
2. 여러 문서가 있을 경우 공통 패턴, 주요 제재 유형, 대표 사례를 정리해 주세요.
3. 기관명, 날짜, 제재 유형(과태료/업무정지/주의 등), 금액 등 구체적 정보를 포함하세요.
4. 자료가 부족해도 있는 정보를 토대로 최선을 다해 답변하세요.
5. 한국어로 답변하세요.
"""

                print("🧠 LLM에 답변 요청 중...")
                answer = ""
                
                # 다양한 LLM 호출 방식 시도
                try:
                    # 방식 1: invoke 메서드 (최신 LangChain)
                    if hasattr(self.llm, "invoke"):
                        result = self.llm.invoke(prompt)
                        if hasattr(result, "content"):
                            answer = result.content
                        else:
                            answer = str(result)
                    # 방식 2: __call__ 메서드 (구 LangChain)
                    else:
                        answer = str(self.llm(prompt))
                        
                    print("✅ LLM 응답 수신 완료")
                    
                except Exception as llm_error:
                    print(f"❌ LLM 호출 실패: {llm_error}")
                    # 기본 응답 생성
                    answer = "죄송합니다. LLM 처리 중 오류가 발생했습니다. 검색된 관련 문서는 다음과 같습니다:\n\n"
                    for idx, doc in enumerate(search_results[:3]):
                        metadata = doc.get("metadata", {})
                        institution = metadata.get('institution', 'N/A')
                        date = metadata.get('date', metadata.get('sanction_date', metadata.get('disclosure_date', 'N/A')))
                        answer += f"{idx+1}. {institution} ({date})\n"
                
                return {
                    "answer": answer,
                    "sources": sources
                }
                
            except Exception as e:
                print(f"❌ 답변 생성 중 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # 오류 발생 시 검색 결과만 반환
                answer = f"답변 생성 중 오류가 발생했습니다. 관련 문서 검색 결과:\n\n"
                for i, doc in enumerate(search_results):
                    metadata = doc.get("metadata", {})
                    institution = metadata.get('institution', 'N/A')
                    date = metadata.get('date', metadata.get('sanction_date', metadata.get('disclosure_date', 'N/A')))
                    answer += f"{i+1}. {institution} ({date})\n"
                
                return {
                    "answer": answer,
                    "sources": sources
                }
                
        except Exception as e:
            print(f"❌ 질문 처리 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "answer": "질문을 처리하는 중 오류가 발생했습니다. 다시 시도해주세요.",
                "sources": []
            }
    
    def _apply_filters(self, docs, filters):
        """추출된 필터를 기반으로 문서 필터링"""
        if not filters:
            return docs
        
        filtered_docs = []
        
        for doc in docs:
            metadata = doc.metadata
            include_doc = True
            
            # 날짜 필터 적용
            if "date_filter" in filters and "date_value" in filters:
                date_field = filters["date_filter"]
                min_year = filters["date_value"]
                
                if date_field in metadata:
                    doc_date = metadata[date_field]
                    try:
                        # 날짜 형식 다양성 처리 (YYYY.MM.DD 또는 YYYY-MM-DD)
                        doc_year = re.search(r"(\d{4})", doc_date).group(1)
                        if doc_year < min_year:
                            include_doc = False
                    except:
                        pass
            
            # 기관 유형 필터 적용
            if "institution_types" in filters and include_doc:
                institution = metadata.get("institution", "").lower()
                
                institution_match = False
                for inst_type in filters["institution_types"]:
                    if inst_type.lower() in institution:
                        institution_match = True
                        break
                
                if not institution_match:
                    include_doc = False
            
            # 제재 유형 필터 적용
            if "sanction_types" in filters and include_doc:
                sanction_type = metadata.get("sanction_type", "").lower()
                management_type = metadata.get("management_type", "").lower()
                
                type_field = sanction_type if sanction_type else management_type
                
                sanction_match = False
                for sanc_type in filters["sanction_types"]:
                    if sanc_type.lower() in type_field or sanc_type.lower() in doc.page_content.lower():
                        sanction_match = True
                        break
                
                if not sanction_match:
                    include_doc = False
            
            # 법규 필터 적용
            if "regulations" in filters and include_doc:
                # 메타데이터에 regulations 필드가 있으면 사용
                regulations = []
                if "regulations" in metadata and isinstance(metadata["regulations"], list):
                    regulations = metadata["regulations"]
                
                # 본문 검색
                content_lower = doc.page_content.lower()
                
                reg_match = False
                for reg in filters["regulations"]:
                    # 메타데이터 검색
                    for doc_reg in regulations:
                        if reg.lower() in doc_reg.lower():
                            reg_match = True
                            break
                    
                    # 본문 검색
                    if reg.lower() in content_lower:
                        reg_match = True
                        break
                
                if not reg_match:
                    include_doc = False
            
            # 내부통제 필터 적용
            if "internal_control" in filters and filters["internal_control"] and include_doc:
                content_lower = doc.page_content.lower()
                
                internal_control_keywords = ["내부통제", "내부 통제", "통제", "관리체계", "관리 체계"]
                internal_control_match = any(keyword in content_lower for keyword in internal_control_keywords)
                
                if not internal_control_match:
                    include_doc = False
            
            # 필터를 모두 통과한 문서만 추가
            if include_doc:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """벡터 저장소에서 문서 검색"""
        try:
            if not self.vector_store:
                print("❌ 벡터 저장소가 로드되지 않았습니다.")
                return []
            
            # 전처리된 쿼리 생성
            processed_query, filters = self.preprocess_query(query)
            
            print(f"🔍 검색어: '{processed_query}', 필터: {filters}")
            
            try:
                # 날짜 필터가 있으면 더 많이 검색 (필터링 후 k개 확보)
                fetch_k = max(k * 10, 50) if filters else max(k * 3, 30)

                # ── ② HyDE: 가상 문서로 검색 쿼리 보강 ──
                hyde_query = self._hyde_query(processed_query)

                # ── Hybrid Search (BM25 + Dense, RRF 합산) ──
                if self.bm25_index:
                    print("📚 BM25 + Dense Hybrid 검색 중...")
                    hybrid_results = self._hybrid_search(hyde_query, k=fetch_k)
                    print(f"✅ Hybrid 검색 완료: {len(hybrid_results)}개")
                else:
                    print("📚 Dense 검색 중 (BM25 없음)...")
                    docs_with_scores = self.vector_store.similarity_search_with_score(hyde_query, k=fetch_k)
                    hybrid_results = [
                        {"content": d.page_content, "metadata": d.metadata, "score": float(max(0.0, 1 - (s ** 2) / 2))}
                        for d, s in docs_with_scores
                    ]
                    print(f"✅ Dense 검색 완료: {len(hybrid_results)}개")

                if not hybrid_results:
                    print("❌ 검색 결과가 없습니다.")
                    return []

                # ── 필터링 적용 ──
                filtered_results = []
                for item in hybrid_results:
                    if filters and not self._match_filters(item["metadata"], filters):
                        continue
                    filtered_results.append(item)

                print(f"✅ 필터링 후: {len(filtered_results)}개 문서 남음")

                # 결과가 없을 경우 필터 없이 상위 결과 반환
                if not filtered_results and filters and hybrid_results:
                    print("⚠️ 필터링 결과 없음 → 필터 제거 후 반환")
                    filtered_results = hybrid_results

                # ── ③ BGE Reranker: 원본 질문 기준으로 재정렬 ──
                rerank_input = filtered_results[:min(30, len(filtered_results))]
                final_results = self._rerank(processed_query, rerank_input, top_k=k)

                return final_results
                
            except Exception as e:
                print(f"❌ 검색 중 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                return []
        
        except Exception as e:
            print(f"❌ 검색 실행 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _doc_passes_filters(self, doc, filters):
        """문서가 필터 조건을 만족하는지 확인"""
        if not filters:
            return True
        
        metadata = doc.metadata
        
        # 날짜 필터 적용
        if "date_filter" in filters and "date_value" in filters:
            date_field = filters["date_filter"]
            min_year = filters["date_value"]
            
            if date_field in metadata:
                doc_date = metadata[date_field]
                try:
                    # 날짜 형식 다양성 처리 (YYYY.MM.DD 또는 YYYY-MM-DD)
                    doc_year = re.search(r"(\d{4})", doc_date).group(1)
                    if doc_year < min_year:
                        return False
                except:
                    pass
        
        # 기관 유형 필터 적용
        if "institution_types" in filters:
            institution = metadata.get("institution", "").lower()
            
            institution_match = False
            for inst_type in filters["institution_types"]:
                if inst_type.lower() in institution:
                    institution_match = True
                    break
            
            if not institution_match:
                return False
        
        # 제재 유형 필터 적용
        if "sanction_types" in filters:
            sanction_type = metadata.get("sanction_type", "").lower()
            management_type = metadata.get("management_type", "").lower()
            
            type_field = sanction_type if sanction_type else management_type
            
            sanction_match = False
            for sanc_type in filters["sanction_types"]:
                if sanc_type.lower() in type_field or sanc_type.lower() in doc.page_content.lower():
                    sanction_match = True
                    break
            
            if not sanction_match:
                return False
        
        # 법규 필터 적용
        if "regulations" in filters:
            # 메타데이터에 regulations 필드가 있으면 사용
            regulations = []
            if "regulations" in metadata and isinstance(metadata["regulations"], list):
                regulations = metadata["regulations"]
            
            # 본문 검색
            content_lower = doc.page_content.lower()
            
            reg_match = False
            for reg in filters["regulations"]:
                # 메타데이터 검색
                for doc_reg in regulations:
                    if reg.lower() in doc_reg.lower():
                        reg_match = True
                        break
                
                # 본문 검색
                if reg.lower() in content_lower:
                    reg_match = True
                    break
            
            if not reg_match:
                return False
        
        # 내부통제 필터 적용
        if "internal_control" in filters and filters["internal_control"]:
            content_lower = doc.page_content.lower()
            
            internal_control_keywords = ["내부통제", "내부 통제", "통제", "관리체계", "관리 체계"]
            internal_control_match = any(keyword in content_lower for keyword in internal_control_keywords)
            
            if not internal_control_match:
                return False
        
        return True
    
    def interactive_mode(self) -> None:
        """대화형 모드"""
        print("\n🤖 금융 제재/경영유의사항 RAG 시스템을 시작합니다. 종료하려면 'exit' 또는 'quit'을 입력하세요.")
        print("💡 'search:'로 시작하면 검색 모드, 그 외에는 질의응답 모드로 동작합니다.")
        
        while True:
            user_input = input("\n❓ 입력: ")
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("👋 RAG 시스템을 종료합니다.")
                break
            
            # 검색 모드
            if user_input.lower().startswith("search:"):
                query = user_input[7:].strip()
                if not query:
                    print("❌ 검색어를 입력해주세요.")
                    continue
                
                print(f"🔍 검색: '{query}'")
                results = self.search_documents(query)
                
                if not results:
                    print("검색 결과가 없습니다.")
                    continue
                
                print("\n📚 검색 결과:")
                for i, result in enumerate(results):
                    print(f"\n결과 #{i+1} (점수: {result['score']:.4f})")
                    
                    # DB 타입에 따라 다른 필드 출력
                    if self.db_type == "sanctions":
                        print(f"기관: {result['metadata'].get('institution', 'N/A')}")
                        print(f"제재일: {result['metadata'].get('sanction_date', 'N/A')}")
                        print(f"유형: {result['metadata'].get('sanction_type', 'N/A')}")
                    else:
                        print(f"기관: {result['metadata'].get('institution', 'N/A')}")
                        print(f"공시일: {result['metadata'].get('disclosure_date', 'N/A')}")
                        print(f"유형: {result['metadata'].get('management_type', 'N/A')}")
                    
                    print(f"내용: {result['content'][:200]}...")
            
            # 질의응답 모드
            else:
                result = self.answer_question(user_input)
                
                print("\n🤖 답변:")
                print(result["answer"])
                
                if result["sources"]:
                    print("\n📚 참고 문서:")
                    for i, source in enumerate(result["sources"][:3]):  # 상위 3개만 표시
                        print(f"\n출처 #{i+1}:")
                        
                        # DB 타입에 따라 다른 필드 출력
                        if self.db_type == "sanctions":
                            print(f"기관: {source['metadata'].get('institution', 'N/A')}")
                            print(f"제재일: {source['metadata'].get('sanction_date', 'N/A')}")
                            print(f"유형: {source['metadata'].get('sanction_type', 'N/A')}")
                        else:
                            print(f"기관: {source['metadata'].get('institution', 'N/A')}")
                            print(f"공시일: {source['metadata'].get('disclosure_date', 'N/A')}")
                            print(f"유형: {source['metadata'].get('management_type', 'N/A')}")
                        
                        print(f"내용: {source['content']}")

    def create_vector_store_from_json(self):
        """JSON 파일에서 직접 벡터 저장소 생성"""
        try:
            print(f"📄 JSON 파일에서 벡터 저장소 생성 중: {self.create_from_json}")
            
            if not os.path.exists(self.create_from_json):
                print(f"❌ JSON 파일을 찾을 수 없습니다: {self.create_from_json}")
                return False
            
            # 임베딩 초기화
            if self.use_openai_embeddings:
                print(f"🧠 OpenAI 임베딩 API 초기화 중...")
                if not self.openai_api_key:
                    self.openai_api_key = os.getenv("OPENAI_API_KEY")
                    if not self.openai_api_key:
                        print("❌ OpenAI API 키가 설정되지 않았습니다.")
                        return False
                
                try:
                    from langchain_openai import OpenAIEmbeddings
                    self.embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-large",
                        openai_api_key=self.openai_api_key,
                        show_progress_bar=True,
                        request_timeout=60
                    )
                    print("✅ OpenAI 임베딩 초기화 완료")
                except Exception as e:
                    print(f"❌ OpenAI 임베딩 초기화 실패: {e}")
                    return False
            else:
                print(f"🧠 HuggingFace 임베딩 초기화 중: {self.embed_model_name}")
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embed_model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                print("✅ HuggingFace 임베딩 초기화 완료")
            
            # JSON 데이터 로드
            with open(self.create_from_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 문서 생성
            from langchain_core.documents import Document
            documents = []
            
            # 데이터 구조 확인
            if isinstance(data, dict) and 'documents' in data:
                docs_list = data['documents']
            elif isinstance(data, list):
                docs_list = data
            else:
                print(f"❌ 알 수 없는 JSON 데이터 구조: {type(data)}")
                return False
            
            print(f"📚 {len(docs_list)}개의 문서 처리 중...")
            
            for doc in docs_list:
                if not isinstance(doc, dict):
                    continue
                    
                # 텍스트 추출
                content = doc.get('content', {})
                if isinstance(content, dict):
                    # 제재 정보의 경우
                    full_text = content.get('full_text', '')
                    if not full_text:
                        # 상세 내용 구성
                        sanction_facts = content.get('sanction_facts', [])
                        facts_text = ""
                        for fact in sanction_facts:
                            if isinstance(fact, dict):
                                facts_text += f"\n- {fact.get('title', '')}: {fact.get('content', '')}"
                        
                        fine_info = content.get('fine', {})
                        if isinstance(fine_info, dict):
                            fine_text = fine_info.get('text', '')
                        else:
                            fine_text = str(fine_info)
                        
                        full_text = f"제재사실:\n{facts_text}\n\n제재내용: {content.get('sanction_type', '')}\n{fine_text}\n{content.get('executive_sanction', '')}"
                    text = full_text
                else:
                    text = str(content)
                
                # 메타데이터 구성
                metadata = {
                    'institution': doc.get('institution', ''),
                    'doc_id': doc.get('doc_id', ''),
                }
                
                # 문서 타입 설정
                if self.db_type == "sanctions":
                    metadata['doc_type'] = '제재정보'
                    if isinstance(content, dict):
                        metadata['sanction_type'] = content.get('sanction_type', '')
                else:
                    metadata['doc_type'] = '경영유의사항'
                    if isinstance(content, dict):
                        metadata['management_type'] = content.get('management_type', '')
                
                # 날짜 필드 추가 (date_normalizer로 정규화)
                from date_normalizer import normalize_date as _norm_date
                raw_date = (doc.get('date') or doc.get('sanction_date') or doc.get('disclosure_date') or '')
                norm_date = _norm_date(raw_date) or raw_date
                metadata['date'] = norm_date
                if norm_date and len(norm_date) >= 7:
                    try:
                        metadata['year'] = int(norm_date[:4])
                        metadata['month'] = int(norm_date[5:7])
                    except ValueError:
                        metadata['year'] = 0
                        metadata['month'] = 0
                else:
                    metadata['year'] = 0
                    metadata['month'] = 0

                # 추가 메타데이터
                doc_metadata = doc.get('metadata', {})
                if isinstance(doc_metadata, dict):
                    # 규정 정보 추가
                    if 'regulations' in doc_metadata:
                        metadata['regulations'] = doc_metadata['regulations']
                    
                    # 기타 메타데이터 복사
                    for key, value in doc_metadata.items():
                        if key not in metadata and value:
                            metadata[key] = value
                
                # 메타데이터 프리픽스 구성 (조항 + 날짜 → 임베딩 정확도 향상)
                institution = metadata.get('institution', '')
                date = metadata.get('date', '')
                doc_type = metadata.get('doc_type', '')
                sanction_type = metadata.get('sanction_type', '') or metadata.get('management_type', '')
                regulations = metadata.get('regulations', [])
                reg_str = ' / '.join(regulations) if regulations else ''

                prefix_parts = []
                if institution:
                    prefix_parts.append(f"[기관: {institution}]")
                if date:
                    prefix_parts.append(f"[날짜: {date}]")
                if doc_type:
                    prefix_parts.append(f"[유형: {doc_type}]")
                if sanction_type:
                    prefix_parts.append(f"[제재: {sanction_type}]")
                if reg_str:
                    prefix_parts.append(f"[조항: {reg_str}]")
                prefix = " ".join(prefix_parts)

                # 텍스트가 너무 짧으면 (스캔 PDF 등) 메타데이터로 대체 청크 생성
                if len(text.strip()) < 50:
                    text = (
                        f"기관명: {institution}\n"
                        f"제재조치일: {date}\n"
                        f"문서유형: {doc_type}\n"
                        f"제재유형: {sanction_type}\n"
                        f"관련조항: {reg_str}\n"
                        f"(원문 스캔 문서)"
                    )

                # 256자 단위로 청킹 (overlap 30)
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=256,
                    chunk_overlap=30,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    if chunk.strip():
                        # 프리픽스를 청크 앞에 붙여 임베딩 (메타데이터 컨텍스트 포함)
                        enriched = f"{prefix}\n{chunk}" if prefix else chunk
                        documents.append(Document(page_content=enriched, metadata=metadata))

            print(f"📄 {len(documents)}개의 청크 준비 완료")

            if not documents:
                print("❌ 문서를 생성할 수 없습니다.")
                return False

            # 배치 크기 계산 (OpenAI 토큰 제한 고려)
            batch_size = 256  # 한 번에 처리할 문서 수
            
            # FAISS 벡터 저장소 생성
            print("🔄 벡터 저장소 생성 중...")
            if self.use_faiss:
                from langchain_community.vectorstores import FAISS
                
                # 첫 번째 배치로 벡터 저장소 초기화
                first_batch = documents[:batch_size]
                print(f"📦 첫 번째 배치 처리 중 (1-{len(first_batch)})")
                self.vector_store = FAISS.from_documents(
                    first_batch,
                    self.embeddings
                )
                
                # 나머지 배치 처리
                for i in range(batch_size, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    end_idx = min(i + batch_size, len(documents))
                    print(f"📦 배치 처리 중 ({i+1}-{end_idx})")
                    
                    # 배치의 텍스트와 메타데이터 분리
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    
                    # 배치 추가
                    self.vector_store.add_texts(
                        texts,
                        metadatas=metadatas
                    )
                
                # FAISS 벡터 저장소를 파일로 저장
                if self.vector_db_path:
                    faiss_dir = os.path.join(self.vector_db_path, "faiss")
                    os.makedirs(faiss_dir, exist_ok=True)
                    
                    # index.faiss와 index.pkl 파일 저장
                    print(f"💾 FAISS 벡터 저장소를 파일로 저장 중: {faiss_dir}")
                    self.vector_store.save_local(faiss_dir)
                    print("✅ FAISS 벡터 저장소 파일 저장 완료")
                
                print("✅ FAISS 벡터 저장소 생성 완료")

                # BM25 인덱스 빌드 및 저장
                if BM25_AVAILABLE:
                    self._build_bm25(documents)
                    self._save_bm25(faiss_dir)
            else:
                from langchain_community.vectorstores import Chroma
                self.vector_store = Chroma.from_documents(
                    documents,
                    self.embeddings
                )
                print("✅ Chroma 벡터 저장소 생성 완료")

            # 벡터 저장소 테스트
            self.check_vector_store()
            
            return True
            
        except Exception as e:
            print(f"❌ 벡터 저장소 생성 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# 사용 예시
if __name__ == "__main__":
    # 사용할 벡터 DB 선택
    db_type = input("사용할 벡터 DB를 선택하세요 (1: 제재정보, 2: 경영유의사항): ")
    
    if db_type == "2":
        vector_db_path = "./data/vector_db/fss_management"
        print("경영유의사항 벡터 DB를 사용합니다.")
    else:
        vector_db_path = "./data/vector_db/fss_sanctions"
        print("제재정보 벡터 DB를 사용합니다.")
    
    # LLM 선택
    use_anthropic = input("Anthropic Claude API를 사용하시겠습니까? (y/n): ").lower() == 'y'
    
    if use_anthropic:
        # API 키 입력
        anthropic_api_key = input("Anthropic API 키를 입력하세요: ")
        
        rag_system = FSSRagSystem(
            vector_db_path=vector_db_path,
            embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            use_anthropic=True,
            anthropic_api_key=anthropic_api_key,
            top_k=5,
            use_openai_embeddings=False,  # 로컬 임베딩 사용
            use_openai_llm=False
        )
    else:
        rag_system = FSSRagSystem(
            vector_db_path=vector_db_path,
            embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            llm_model_name="gpt-4o",
            top_k=5,
            use_openai_embeddings=False,  # 로컬 임베딩 사용
            use_openai_llm=True
        )
    
    # 대화형 모드 시작
    rag_system.interactive_mode() 