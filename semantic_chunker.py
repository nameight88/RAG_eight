"""
금융감독원 제재 JSON 코퍼스를 시맨틱 청킹하고 벡터 저장소에 임베딩하는 모듈
- 문서를 의미 있는 청크로 분할
- 벡터 임베딩 생성
- 벡터 데이터베이스에 저장소 구축
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
# 임베딩 모델 임포트
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import chromadb
from langchain.docstore.document import Document
import datetime

# .env 파일에서 환경 변수 로드
load_dotenv()


class FSSSemanticChunker:
    """금융감독원 제재 정보를 시맨틱 청킹하는 클래스"""
    
    def __init__(
        self,
        input_json: str,
        output_dir: str,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_openai_embeddings: bool = True,  # 기본값을 True로 변경
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_faiss: bool = False  # FAISS 사용 여부 추가
    ):
        """
        초기화
        
        Args:
            input_json: 입력 JSON 파일 경로
            output_dir: 출력 디렉토리
            model_name: 시맨틱 청킹에 사용할 임베딩 모델 이름 (use_openai_embeddings=False인 경우)
            use_openai_embeddings: OpenAI 임베딩 API 사용 여부
            chunk_size: 청크 크기
            chunk_overlap: 청크 겹침 크기
            use_faiss: FAISS 벡터 저장소 사용 여부
        """
        self.input_json = input_json
        self.output_dir = output_dir
        self.model_name = model_name
        self.use_openai_embeddings = use_openai_embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_faiss = use_faiss
        
        # OpenAI API 키 로드
        if use_openai_embeddings:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                print("⚠️ OpenAI API 키가 없습니다. HuggingFace 임베딩으로 전환합니다.")
                self.use_openai_embeddings = False

        # Langchain 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
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
                self.corpus = documents
                return documents
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
                "doc_id": "SAMPLE_001",
                "source_file": "sample_file_1.pdf",
                "institution": "테스트금융주식회사",
                "sanction_date": "2023.01.15",
                "content": {
                    "sanction_type": "과태료",
                    "fine": {"amount": 5000000, "unit": "원", "text": "과태료 5백만원"},
                    "executive_sanction": "CEO 주의적 경고",
                    "sanction_facts": [
                        {"title": "전자금융 안전조치 미흡", "content": "전자금융거래법 제21조에 따른 안전조치 의무를 소홀히 하여 고객정보 유출 위험을 초래함"}
                    ],
                    "full_text": "테스트금융(주)는 전자금융거래법 제21조에 따른 안전조치 의무를 소홀히 하여 고객정보 유출 위험을 초래하였으므로 과태료 5백만원의 제재를 결정함"
                },
                "metadata": {
                    "doc_type": "제재내용공개",
                    "char_count": 200,
                    "regulations": ["전자금융거래법 제21조", "전자금융감독규정 제13조"]
                },
                "quality_score": 4
            },
            {
                "doc_id": "SAMPLE_002",
                "source_file": "sample_file_2.pdf",
                "institution": "샘플은행",
                "sanction_date": "2023.02.20",
                "content": {
                    "sanction_type": "기관경고",
                    "fine": {"amount": 0, "unit": "원", "text": ""},
                    "executive_sanction": "CIO 문책경고",
                    "sanction_facts": [
                        {"title": "정보보호 인력 부족", "content": "전자금융감독규정에 따른 정보보호 인력을 충분히 확보하지 않고 운영함"},
                        {"title": "보안 취약점 방치", "content": "알려진 보안 취약점에 대한 조치를 6개월 이상 지연하여 조치함"}
                    ],
                    "full_text": "샘플은행은 전자금융감독규정에 따른 정보보호 인력을 충분히 확보하지 않고 운영하였으며, 알려진 보안 취약점에 대한 조치를 6개월 이상 지연하여 조치한 사실이 확인되어 기관경고 조치함"
                },
                "metadata": {
                    "doc_type": "제재내용공개",
                    "char_count": 300,
                    "regulations": ["전자금융감독규정 제36조", "전자금융거래법 제21조의2"]
                },
                "quality_score": 5
            },
            {
                "doc_id": "SAMPLE_003",
                "source_file": "sample_file_3.hwp",
                "institution": "예시보험",
                "sanction_date": "2023.03.10",
                "content": {
                    "sanction_type": "과태료",
                    "fine": {"amount": 10000000, "unit": "원", "text": "과태료 1천만원"},
                    "executive_sanction": "",
                    "sanction_facts": [
                        {"title": "개인정보 유출사고", "content": "고객 개인정보 관리 소홀로 인해 약 1,000명의 고객정보가 외부에 유출됨"}
                    ],
                    "full_text": "예시보험은 고객 개인정보 관리 소홀로 인해 약 1,000명의 고객정보가 외부에 유출된 사실이 확인되어 과태료 1천만원을 부과함. 해당 사고는 내부 직원의 관리 소홀 및 시스템 보안 취약점으로 인해 발생하였음."
                },
                "metadata": {
                    "doc_type": "제재내용공개",
                    "char_count": 250,
                    "regulations": ["개인정보보호법 제29조", "신용정보법 제19조"]
                },
                "quality_score": 3
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
    
    def preprocess_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """문서 전처리"""
        # 필요한 필드 추출
        processed = {
            "id": doc.get("doc_id", ""),
            "institution": doc.get("institution", ""),
            "sanction_date": doc.get("sanction_date", ""),
            "sanction_type": doc.get("content", {}).get("sanction_type", ""),
            "fine_amount": doc.get("content", {}).get("fine", {}).get("amount", 0),
            "fine_text": doc.get("content", {}).get("fine", {}).get("text", ""),
            "sanction_facts": doc.get("content", {}).get("sanction_facts", []),
            "full_text": doc.get("content", {}).get("full_text", ""),
            "regulations": doc.get("metadata", {}).get("regulations", []),
            "quality_score": doc.get("quality_score", 0),
        }
        
        # 제재 사실 텍스트 추출
        facts_text = ""
        for fact in processed["sanction_facts"]:
            if isinstance(fact, dict):
                title = fact.get("title", "")
                content = fact.get("content", "")
                if title and content:
                    facts_text += f"{title}\n{content}\n\n"
                elif title:
                    facts_text += f"{title}\n\n"
                elif content:
                    facts_text += f"{content}\n\n"
        
        # 규정 텍스트 생성
        regulations_text = "\n".join(processed["regulations"]) if processed["regulations"] else ""
        
        # 통합 텍스트 생성
        combined_text = (
            f"금융기관: {processed['institution']}\n"
            f"제재일자: {processed['sanction_date']}\n"
            f"제재유형: {processed['sanction_type']}\n"
            f"과태료: {processed['fine_text']}\n\n"
            f"제재사실:\n{facts_text}\n"
            f"관련규정:\n{regulations_text}\n\n"
            f"{processed['full_text']}"
        )
        
        processed["combined_text"] = combined_text
        return processed
    
    def create_semantic_chunks(self, preprocessed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """시맨틱 청킹 수행"""
        all_chunks = []
        
        print(f"🔪 {len(preprocessed_docs)}개 문서 청킹 시작...")
        for doc in tqdm(preprocessed_docs):
            # 텍스트 분할
            chunks = self.text_splitter.split_text(doc["combined_text"])
            
            # 각 청크에 메타데이터 추가
            doc_chunks = []
            for i, chunk_text in enumerate(chunks):
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
                doc_chunks.append(chunk)
            
            all_chunks.extend(doc_chunks)
        
        print(f"✅ 총 {len(all_chunks)}개 청크 생성 완료")
        return all_chunks
    
    def create_vector_store(self, batch_size: int = 100):
        """벡터 저장소 생성 (배치 처리 추가)"""
        print("🔢 벡터 저장소 생성 중...")
        
        # 출력 파일 경로
        output_json = os.path.join(self.output_dir, "fss_sanctions_parsed.json")
        
        try:
            # 데이터 로드
            with open(output_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 문서 배열 생성
            documents = []
            for item in data:
                for chunk in item['chunks']:
                    doc = Document(
                        page_content=chunk['content'],
                        metadata={
                            'id': chunk['id'],
                            'institution': item['institution'],
                            'sanction_date': item['sanction_date'],
                            'sanction_type': item['sanction_type'],
                            'violation_regulation': item.get('violation_regulation', '')
                        }
                    )
                    documents.append(doc)
            
            print(f"총 {len(documents)}개 문서 준비 완료")
            
            # 임베딩 초기화
            if self.use_openai_embeddings:
                print("🧠 OpenAI 임베딩 API 초기화 중...")
                embeddings = OpenAIEmbeddings(
                    #model="text-embedding-3-small",
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
                self.db = vectorstore
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
    
    def test_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """테스트 쿼리 수행"""
        if not self.db:
            print("❌ 벡터 저장소가 초기화되지 않았습니다.")
            return []
        
        print(f"🔍 쿼리: '{query}'")
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
    
    def process(self) -> None:
        """전체 처리 과정 실행"""
        # 1. 코퍼스 로드
        documents = self.load_corpus()
        if not documents:
            return
        
        # 2. 문서 전처리
        print("🔄 문서 전처리 중...")
        preprocessed_docs = [self.preprocess_document(doc) for doc in tqdm(documents)]
        
        # 3. 시맨틱 청킹
        chunks = self.create_semantic_chunks(preprocessed_docs)
        # 3.5. 청크 결과 저장
        output_json_path = os.path.join(self.output_dir, "fss_sanctions_parsed.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(
                [
                    {
                        "institution": doc["institution"],
                        "sanction_date": doc["sanction_date"],
                        "sanction_type": doc["sanction_type"],
                        "chunks": [
                            {
                                "id": chunk["id"],
                                "content": chunk["chunk_text"]
                            }
                            for chunk in chunks if chunk["doc_id"] == doc["id"]
                        ]
                    }
                    for doc in preprocessed_docs
                ],
                f,
                ensure_ascii=False,
                indent=2
            )
        print("✅ 청크 JSON 저장 완료")
        # 4. 벡터 저장소 구축
        self.create_vector_store(batch_size=100)
        
        print(f"🎉 처리 완료!")
        
        # 5. 테스트 쿼리 실행
        test_queries = [
            "전자금융 관련 과태료 부과 사례",
            "개인정보 유출 관련 제재",
            "정보보호 위반으로 인한 제재",
        ]
        
        for query in test_queries:
            results = self.test_query(query)
            print(f"\n📊 테스트 쿼리 '{query}' 결과:")
            for i, result in enumerate(results[:2]):  # 상위 2개만 출력
                print(f"  #{i+1} (점수: {result['score']:.4f})")
                print(f"  기관: {result['metadata']['institution']}")
                print(f"  제재일: {result['metadata']['sanction_date']}")
                print(f"  내용: {result['content'][:150]}...\n")


# 사용 예시
if __name__ == "__main__":
    chunker = FSSSemanticChunker(
        input_json="./data/FSS_SANCTION/fss_sanctions_parsed.json",  # 상대 경로 수정
        output_dir="./data/vector_db/fss_sanctions",  # 상대 경로 수정
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 한국어 지원 다국어 모델로 변경
        #use_openai_embeddings=True, # OpenAI 임베딩 사용 여부를 True로 변경
        use_openai_embeddings=False,
        chunk_size=512,
        chunk_overlap=50,
        use_faiss=True # FAISS 사용 여부
    )
    
    chunker.process() 