"""
금융감독원 경영유의 사항 JSON 코퍼스를 적응형 청킹(Adaptive Chunking)으로 처리하는 모듈
- 문서 구조와 내용에 따라 적응적으로 청크 크기 조정
- 메타데이터 보존 및 문맥 유지를 위한 최적화
- 벡터 저장소 구축
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import chromadb
import traceback
from langchain.docstore.document import Document
import datetime
from date_normalizer import normalize_date as _normalize_date

# .env 파일에서 환경 변수 로드
load_dotenv()


class AdaptiveChunkingStrategy:
    """문서 특성에 따라 청킹 전략을 적응적으로 결정하는 클래스"""
    
    def __init__(
        self,
        default_chunk_size: int = 512,
        default_chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024,
    ):
        """초기화"""
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # 기본 텍스트 분할기
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.default_chunk_size,
            chunk_overlap=self.default_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def determine_chunk_parameters(self, document: Dict[str, Any]) -> Tuple[int, int]:
        """문서의 특성에 따라 최적의 청크 크기와 겹침을 결정"""
        content = document.get('content', {}).get('full_text', '')
        
        # 문서 길이 기반 결정
        content_length = len(content)
        
        # 매우 짧은 문서는 분할하지 않음
        if content_length < self.min_chunk_size * 2:
            return content_length, 0
        
        # 매우 긴 문서는 작은 청크로 분할
        if content_length > 10000:
            return min(384, self.default_chunk_size), 30
            
        # 기본 설정 사용
        return self.default_chunk_size, self.default_chunk_overlap
    
    def get_splitter_for_document(self, document: Dict[str, Any]) -> RecursiveCharacterTextSplitter:
        """문서에 맞는 분할기 생성"""
        chunk_size, chunk_overlap = self.determine_chunk_parameters(document)
        
        if chunk_size == self.default_chunk_size and chunk_overlap == self.default_chunk_overlap:
            return self.default_splitter
        
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_by_structure(self, document: Dict[str, Any]) -> List[str]:
        """문서 구조를 고려한 청킹"""
        content = document.get('content', {})
        full_text = content.get('full_text', '')
        
        # 문서가 충분히 짧으면 분할하지 않음
        if len(full_text) < self.min_chunk_size * 2:
            return [full_text]
        
        # 문서 특성에 맞는 분할기 사용
        splitter = self.get_splitter_for_document(document)
        chunks = splitter.split_text(full_text)
        
        return chunks


class FSSAdaptiveChunker:
    """금융감독원 경영유의 문서 적응형 청킹 및 벡터 저장소 구축"""
    
    def __init__(
        self,
        input_json: str,
        output_dir: str = "./data/vector_db",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_openai_embeddings: bool = True,  # 기본값을 True로 변경
        default_chunk_size: int = 512,
        default_chunk_overlap: int = 50,
        use_faiss: bool = False  # FAISS 사용 여부 추가
    ):
        """
        초기화
        
        Args:
            input_json: 입력 JSON 파일 경로
            output_dir: 출력 디렉토리
            model_name: 임베딩 모델 이름 (use_openai_embeddings=False인 경우)
            use_openai_embeddings: OpenAI 임베딩 API 사용 여부
            default_chunk_size: 기본 청크 크기
            default_chunk_overlap: 기본 청크 겹침 크기
            use_faiss: FAISS 벡터 저장소 사용 여부
        """
        self.input_json = input_json
        self.output_dir = output_dir
        self.model_name = model_name
        self.use_openai_embeddings = use_openai_embeddings
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.use_faiss = use_faiss
        
        # OpenAI API 키 로드
        if use_openai_embeddings:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                print("⚠️ OpenAI API 키가 없습니다. HuggingFace 임베딩으로 전환합니다.")
                self.use_openai_embeddings = False
        
        # 적응형 청킹 전략 초기화
        self.chunking_strategy = AdaptiveChunkingStrategy(
            default_chunk_size=default_chunk_size,
            default_chunk_overlap=default_chunk_overlap
        )
        
        # 임베딩 모델 초기화
        if use_openai_embeddings and self.openai_api_key:
            print(f"🧠 OpenAI 임베딩 API 초기화 중...")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=self.openai_api_key
            )
            print(f"✅ OpenAI 임베딩 초기화 완료")
        else:
            print(f"🧠 HuggingFace 임베딩 모델 초기화 중: {self.model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ 임베딩 모델 초기화 완료")
        
        # 데이터 및 벡터 저장소
        self.corpus = None
        self.db = None
        
        # 청크 통계
        self.chunk_stats = {
            "total_docs": 0,
            "total_chunks": 0,
            "avg_chunks_per_doc": 0,
            "min_chunks": float('inf'),
            "max_chunks": 0,
        }
    
    def load_corpus(self) -> List[Dict[str, Any]]:
        """JSON 코퍼스 로드"""
        try:
            print(f"📄 JSON 코퍼스 로드 중: {self.input_json}")
            
            # 디렉토리 확인
            input_dir = os.path.dirname(self.input_json)
            if not os.path.exists(input_dir):
                print(f"⚠️ 디렉토리가 존재하지 않습니다. 생성합니다: {input_dir}")
                os.makedirs(input_dir, exist_ok=True)
            
            # 파일 존재 여부 확인
            if not os.path.exists(self.input_json):
                print(f"⚠️ JSON 파일이 존재하지 않습니다: {self.input_json}")
                print("테스트 목적으로 샘플 데이터를 생성합니다.")
                return self.create_sample_corpus()
            
            with open(self.input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "documents" in data:
                documents = data["documents"]
                print(f"✅ {len(documents)}개 문서 로드 완료")
                
                # 관련 문서만 필터링 (전자금융 관련 문서)
                relevant_docs = [doc for doc in documents if doc.get('is_relevant', False)]
                print(f"🔍 전자금융 관련 문서: {len(relevant_docs)}개")
                
                self.corpus = relevant_docs
                return relevant_docs
            else:
                print("❌ JSON 파일에 'documents' 키가 없습니다.")
                return self.create_sample_corpus()
        
        except Exception as e:
            print(f"❌ 코퍼스 로드 실패: {e}")
            print("테스트 목적으로 샘플 데이터를 생성합니다.")
            return self.create_sample_corpus()
    
    def create_sample_corpus(self) -> List[Dict[str, Any]]:
        """테스트용 샘플 코퍼스 생성"""
        print("🔄 테스트 샘플 데이터 생성 중...")
        
        sample_documents = [
            {
                "doc_id": "SAMPLE_MGMT_001",
                "source_file": "sample_file_1.pdf",
                "institution": "테스트금융주식회사",
                "disclosure_date": "2023.01.15",
                "is_relevant": True,
                "found_keywords": ["전자금융", "정보처리위탁"],
                "content": {
                    "management_type": "전자금융 관련 경영유의사항",
                    "management_details": [
                        {
                            "title": "1. 전자금융 안전조치 미흡",
                            "content": "전자금융거래법 제21조에 따른 안전조치 의무를 소홀히 하여 고객정보 유출 위험을 초래함. 금융회사는 전자금융거래의 안전성 확보를 위하여 인력, 시설, 전자적 장치 등의 정보기술부문, 전자금융업무 영위를 위한 내부통제절차 등에 관하여 금융위원회가 정하는 기준을 준수해야 함에도 불구하고 보안 취약점을 방치하였음."
                        },
                        {
                            "title": "2. 정보처리 업무 위탁 관리 부실",
                            "content": "정보처리 업무 위탁 관리가 부실하여 위탁업체에 대한 관리·감독이 제대로 이루어지지 않았음. 특히 외부 개발업체에 대한 보안관리가 미흡하여 개발 단계에서 보안 취약점이 발생했으며, 이에 대한 조치가 지연되었음."
                        }
                    ],
                    "full_text": "테스트금융주식회사 경영유의사항 공시\n\n1. 전자금융 안전조치 미흡\n전자금융거래법 제21조에 따른 안전조치 의무를 소홀히 하여 고객정보 유출 위험을 초래함. 금융회사는 전자금융거래의 안전성 확보를 위하여 인력, 시설, 전자적 장치 등의 정보기술부문, 전자금융업무 영위를 위한 내부통제절차 등에 관하여 금융위원회가 정하는 기준을 준수해야 함에도 불구하고 보안 취약점을 방치하였음.\n\n2. 정보처리 업무 위탁 관리 부실\n정보처리 업무 위탁 관리가 부실하여 위탁업체에 대한 관리·감독이 제대로 이루어지지 않았음. 특히 외부 개발업체에 대한 보안관리가 미흡하여 개발 단계에서 보안 취약점이 발생했으며, 이에 대한 조치가 지연되었음."
                },
                "metadata": {
                    "doc_type": "경영유의사항",
                    "char_count": 500,
                    "regulations": ["전자금융거래법 제21조", "정보처리 업무 위탁에 관한 규정 제7조"]
                },
                "quality_score": 4
            },
            {
                "doc_id": "SAMPLE_MGMT_002",
                "source_file": "sample_file_2.pdf",
                "institution": "샘플은행",
                "disclosure_date": "2023.02.20",
                "is_relevant": True,
                "found_keywords": ["신용정보법"],
                "content": {
                    "management_type": "신용정보 관련 경영유의사항",
                    "management_details": [
                        {
                            "title": "1. 신용정보 관리 소홀",
                            "content": "신용정보법 제19조에 따른 신용정보의 관리 및 보호 의무를 소홀히 하여 신용정보 유출 사고가 발생함. 신용정보회사등은 신용정보의 분실·도난·유출·변조 또는 훼손을 방지하기 위하여 내부관리규정 준수, 접근통제 등 안전성 확보에 필요한 조치를 취해야 하나 이를 이행하지 않았음."
                        },
                        {
                            "title": "2. 위탁업체 관리 부실",
                            "content": "신용정보법 제17조에 따른 위탁업체 관리 의무를 준수하지 않았음. 업무 위탁 시 위탁업체의 기술적·물리적·관리적 보안 대책 수립 여부를 확인하지 않고 보안 약정을 체결하지 않는 등 위탁업체 관리가 미흡하였음."
                        }
                    ],
                    "full_text": "샘플은행 경영유의사항 공시\n\n1. 신용정보 관리 소홀\n신용정보법 제19조에 따른 신용정보의 관리 및 보호 의무를 소홀히 하여 신용정보 유출 사고가 발생함. 신용정보회사등은 신용정보의 분실·도난·유출·변조 또는 훼손을 방지하기 위하여 내부관리규정 준수, 접근통제 등 안전성 확보에 필요한 조치를 취해야 하나 이를 이행하지 않았음.\n\n2. 위탁업체 관리 부실\n신용정보법 제17조에 따른 위탁업체 관리 의무를 준수하지 않았음. 업무 위탁 시 위탁업체의 기술적·물리적·관리적 보안 대책 수립 여부를 확인하지 않고 보안 약정을 체결하지 않는 등 위탁업체 관리가 미흡하였음."
                },
                "metadata": {
                    "doc_type": "경영유의사항",
                    "char_count": 450,
                    "regulations": ["신용정보법 제17조", "신용정보법 제19조"]
                },
                "quality_score": 5
            }
        ]
        
        # 샘플 데이터 저장
        os.makedirs(os.path.dirname(self.input_json), exist_ok=True)
        
        with open(self.input_json, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "source": "샘플 데이터",
                    "created_at": "2023-06-13"
                },
                "documents": sample_documents
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 샘플 데이터 생성 완료: {len(sample_documents)}개 문서")
        return sample_documents
    
    def create_adaptive_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """적응형 청킹 수행"""
        all_chunks = []
        chunks_per_doc = []
        
        print(f"🔪 {len(documents)}개 문서 적응형 청킹 시작...")
        
        for doc in tqdm(documents):
            # 적응형 청킹 전략 사용
            chunks = self.chunking_strategy.split_by_structure(doc)
            
            # 각 청크에 메타데이터 추가
            doc_chunks = []
            for i, chunk_text in enumerate(chunks):
                # 리스트 타입을 문자열로 변환
                found_keywords = doc.get('found_keywords', [])
                keywords_str = ', '.join(found_keywords) if found_keywords else ''
                
                # 문서 타입에 따라 필드명 결정
                doc_type = doc.get('metadata', {}).get('doc_type', '')

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
                    "keywords": keywords_str,  # 리스트 대신 문자열로 저장
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_text": chunk_text
                }
                doc_chunks.append(chunk)
            
            # 통계 업데이트
            if len(doc_chunks) > 0:
                chunks_per_doc.append(len(doc_chunks))
                self.chunk_stats["min_chunks"] = min(self.chunk_stats["min_chunks"], len(doc_chunks))
                self.chunk_stats["max_chunks"] = max(self.chunk_stats["max_chunks"], len(doc_chunks))
            
            all_chunks.extend(doc_chunks)
        
        # 최종 통계 계산
        self.chunk_stats["total_docs"] = len(documents)
        self.chunk_stats["total_chunks"] = len(all_chunks)
        if chunks_per_doc:
            self.chunk_stats["avg_chunks_per_doc"] = sum(chunks_per_doc) / len(chunks_per_doc)
        else:
            self.chunk_stats["min_chunks"] = 0
        
        print(f"✅ 총 {len(all_chunks)}개 청크 생성 완료")
        print(f"📊 평균 청크 수 (문서당): {self.chunk_stats['avg_chunks_per_doc']:.2f}개")
        print(f"📊 최소/최대 청크 수: {self.chunk_stats['min_chunks']}개/{self.chunk_stats['max_chunks']}개")
        
        return all_chunks
    
    def build_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """벡터 저장소 구축"""
        try:
            # 출력 디렉토리 생성
            os.makedirs(self.output_dir, exist_ok=True)
            
            print(f"🔄 벡터 저장소 구축 시작...")
            
            # 텍스트와 메타데이터 분리
            texts = []
            metadatas = []
            
            for chunk in chunks:
                text = chunk.get("chunk_text", "")
                
                # 유효한 텍스트만 추가
                if not text.strip():
                    continue
                    
                texts.append(text)
                
                # 안전한 메타데이터 구성 (직접 필요한 필드만 선택)
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
                
                metadatas.append(safe_metadata)
            
            if not texts:
                print("❌ 유효한 청크가 없습니다.")
                return
            
            # Chroma DB 생성
            self.db = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=self.output_dir
            )
            
            # 저장
            self.db.persist()
            
            print(f"✅ 벡터 저장소 구축 완료: {self.output_dir}")
            print(f"📊 임베딩된 청크 수: {len(texts)}")
            
            # 정보 저장
            import datetime
            info = {
                "model_name": self.model_name,
                "chunk_strategy": "adaptive",
                "document_count": self.chunk_stats["total_docs"],
                "chunk_count": len(texts),
                "avg_chunks_per_doc": self.chunk_stats["avg_chunks_per_doc"],
                "min_chunks": self.chunk_stats["min_chunks"],
                "max_chunks": self.chunk_stats["max_chunks"],
                "created_at": datetime.datetime.now().isoformat(),
            }
            
            with open(os.path.join(self.output_dir, "vector_store_info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"❌ 벡터 저장소 구축 실패: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def test_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """테스트 쿼리 수행"""
        if not self.db:
            print("❌ 벡터 저장소가 초기화되지 않았습니다.")
            return []
        
        print(f"🔍 쿼리: '{query}'")
        
        try:
            results = self.db.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 쿼리 실행 중 오류: {str(e)}")
            return []
    
    def process(self):
        """전체 처리 프로세스 실행"""
        print(f"🔄 경영유의사항 처리 시작...")
        
        # 1. 데이터 로드
        documents = self.load_corpus()
        
        # 2. 적응형 청킹
        chunked_docs = self.create_adaptive_chunks(documents)
        
        # 3. 결과 저장
        output_json = os.path.join(self.output_dir, "fss_management_parsed.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(chunked_docs, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 경영유의사항 청킹 결과 저장 완료: {output_json}")
        
        # 4. 벡터 저장소 생성
        self.create_vector_store(batch_size=100)
        
        print(f"🎉 처리 완료!")
    
    def create_vector_store(self, batch_size: int = 100):
        """벡터 저장소 생성 (배치 처리 추가)"""
        print("🔢 벡터 저장소 생성 중...")
        
        # 출력 파일 경로
        output_json = os.path.join(self.output_dir, "fss_management_parsed.json")
        
        try:
            # 데이터 로드
            with open(output_json, 'r', encoding='utf-8') as f:
                chunks = json.load(f)  # <- 바로 chunk 리스트임

                documents = []
                for chunk in chunks:
                    text = chunk.get("chunk_text", "") or chunk.get("content", "")
                    if not text.strip():
                        continue

                    doc = Document(
                        page_content=text,
                        metadata={
                            'id': chunk.get('id', ''),
                            'doc_id': chunk.get('doc_id', ''),
                            'institution': chunk.get('institution', ''),
                            'date': chunk.get('date', ''),
                            'year': chunk.get('year', 0),
                            'month': chunk.get('month', 0),
                            'doc_type': chunk.get('doc_type', ''),
                            'keywords': chunk.get('keywords', '')
                        }
                    )
                    documents.append(doc)

            
            print(f"총 {len(documents)}개 문서 준비 완료")
            
            # 임베딩 초기화
            if self.use_openai_embeddings:
                print("🧠 OpenAI 임베딩 API 초기화 중...")
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=self.openai_api_key
                )
            else:
                print(f"🧠 HuggingFace 임베딩 모델 초기화 중: {self.model_name}")
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            # 벡터 저장소 생성
            if self.use_faiss:
                # FAISS 벡터 저장소 디렉토리
                faiss_dir = os.path.join(self.output_dir, "faiss")
                os.makedirs(faiss_dir, exist_ok=True)
                
                print(f"FAISS 벡터 저장소 생성 중: {faiss_dir}")
                
                # 초기 벡터 저장소 생성 (첫 배치 사용)
                first_batch = documents[:batch_size]
                vectorstore = FAISS.from_documents(
                    documents=first_batch,
                    embedding=embeddings
                )
                
                # 나머지 배치 추가
                for i in tqdm(range(batch_size, len(documents), batch_size), desc="FAISS에 문서 추가 중"):
                    batch = documents[i:i+batch_size]
                    vectorstore.add_documents(batch)
                
                
                # 로컬 파일에 저장
                vectorstore.save_local(faiss_dir)
                
                print(f"✅ FAISS 벡터 저장소 생성 완료")
                
                # 벡터 저장소 정보 저장
                vector_store_info = {
                    'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'document_count': len(documents),
                    'embed_model': self.model_name if not self.use_openai_embeddings else 'text-embedding-3-large',
                    'use_openai': self.use_openai_embeddings,
                    'vector_store_type': 'FAISS'
                }
                
                with open(os.path.join(self.output_dir, 'vector_store_info.json'), 'w', encoding='utf-8') as f:
                    json.dump(vector_store_info, f, ensure_ascii=False, indent=2)
                
            else:
                # Chroma 벡터 저장소 생성
                print(f"Chroma 벡터 저장소 생성 중: {self.output_dir}")
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=self.output_dir
                )
                vectorstore.persist()
                
                print(f"✅ Chroma 벡터 저장소 생성 완료")
                
                # 벡터 저장소 정보 저장
                vector_store_info = {
                    'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'document_count': len(documents),
                    'embed_model': self.model_name if not self.use_openai_embeddings else 'text-embedding-3-large',
                    'use_openai': self.use_openai_embeddings,
                    'vector_store_type': 'Chroma'
                }
                
                with open(os.path.join(self.output_dir, 'vector_store_info.json'), 'w', encoding='utf-8') as f:
                    json.dump(vector_store_info, f, ensure_ascii=False, indent=2)
            
            print(f"📊 벡터 저장소 정보: {len(documents)}개 문서 임베딩")
            
        except Exception as e:
            print(f"❌ 벡터 저장소 생성 실패: {e}")


# 사용 예시
if __name__ == "__main__":
    chunker = FSSAdaptiveChunker(
        input_json="./data/FSS_MANAGEMENT/fss_management_parsed.json",
        output_dir="./data/vector_db/fss_management",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_openai_embeddings=False,  # OpenAI 임베딩 사용으로 변경
        default_chunk_size=512,
        default_chunk_overlap=50,
        use_faiss=True  # FAISS 사용으로 변경
    )
    
    chunker.process() 