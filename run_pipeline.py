"""
금융감독원 제재/경영유의사항 RAG 파이프라인
- 코퍼스 로드 및 청킹
- 벡터 저장소 구축
- 대화형 RAG 시스템 실행
"""

import os
import argparse
import json
from typing import Dict, List, Any, Optional
import shutil
from datetime import datetime
from dotenv import load_dotenv
from semantic_chunker import FSSSemanticChunker
from adaptive_chunker import FSSAdaptiveChunker
from rag_system import FSSRagSystem

# .env 파일에서 환경 변수 로드
load_dotenv()


def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='금융감독원 제재/경영유의사항 RAG 파이프라인')
    
    # 입력 파일/디렉토리 설정
    parser.add_argument('--sanctions-json', type=str, default='./data/fss_sanctions_parsed.json',
                        help='제재 정보 파싱된 JSON 파일 경로')
    parser.add_argument('--management-json', type=str, default='./data/fss_management_parsed.json',
                        help='경영유의사항 파싱된 JSON 파일 경로')
    
    # 출력 디렉토리 설정
    parser.add_argument('--vector-db-dir', type=str, default='./data/vector_db',
                        help='벡터 저장소 디렉토리')
    
    # 임베딩 설정
    parser.add_argument('--embed-model', type=str, 
                        default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        help='임베딩 모델 이름')
    parser.add_argument('--use-openai-embeddings', action='store_true', default=True,
                        help='OpenAI 임베딩 API 사용 (환경 변수 OPENAI_API_KEY 필요)')
    parser.add_argument('--use-faiss', action='store_true', default=True,
                        help='Chroma 대신 FAISS 벡터 저장소 사용')
    
    # 청킹 설정
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='청크 크기')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                        help='청크 겹침 크기')
    
    # 처리 단계 선택
    parser.add_argument('--skip-sanctions', action='store_true',
                        help='제재 정보 처리 건너뛰기')
    parser.add_argument('--skip-management', action='store_true',
                        help='경영유의사항 처리 건너뛰기')
    parser.add_argument('--interactive', action='store_true',
                        help='처리 후 대화형 모드 실행')
    
    # 샘플 데이터 옵션
    parser.add_argument('--use-sample-data', action='store_true',
                        help='샘플 데이터 사용 (실제 데이터가 없는 경우)')
    
    return parser.parse_args()


def create_sample_data(sanctions_json: str, management_json: str):
    """샘플 데이터 생성"""
    # 디렉토리 생성
    os.makedirs(os.path.dirname(sanctions_json), exist_ok=True)
    os.makedirs(os.path.dirname(management_json), exist_ok=True)
    
    # 제재 정보 샘플 데이터
    if not os.path.exists(sanctions_json):
        print(f"🔄 샘플 제재 정보 데이터 생성: {sanctions_json}")
        sample_sanctions = {
            "metadata": {
                "source": "샘플 데이터",
                "created_at": datetime.now().isoformat()
            },
            "documents": [
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
                }
            ]
        }
        
        with open(sanctions_json, 'w', encoding='utf-8') as f:
            json.dump(sample_sanctions, f, ensure_ascii=False, indent=2)
    
    # 경영유의사항 샘플 데이터
    if not os.path.exists(management_json):
        print(f"🔄 샘플 경영유의사항 데이터 생성: {management_json}")
        sample_management = {
            "metadata": {
                "source": "샘플 데이터",
                "created_at": datetime.now().isoformat()
            },
            "documents": [
                {
                    "doc_id": "SAMPLE_MGMT_001",
                    "source_file": "sample_mgmt_1.pdf",
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
                }
            ]
        }
        
        with open(management_json, 'w', encoding='utf-8') as f:
            json.dump(sample_management, f, ensure_ascii=False, indent=2)
    
    print("✅ 샘플 데이터 생성 완료")


def process_sanctions(args):
    """제재 정보 처리"""
    print("\n🔄 제재 정보 처리 시작...")
    
    # 디렉토리 설정
    sanctions_db_dir = os.path.join(os.path.abspath(args.vector_db_dir), "fss_sanctions")
    os.makedirs(sanctions_db_dir, exist_ok=True)
    
    # 시맨틱 청커 초기화 및 처리
    sanctions_chunker = FSSSemanticChunker(
        input_json=os.path.abspath(args.sanctions_json),
        output_dir=sanctions_db_dir,
        model_name=args.embed_model,
        use_openai_embeddings=args.use_openai_embeddings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_faiss=args.use_faiss  # FAISS 사용 여부 추가
    )
    
    sanctions_chunker.process()
    
    print("✅ 제재 정보 처리 완료")


def process_management(args):
    """경영유의사항 처리"""
    print("\n🔄 경영유의사항 처리 시작...")
    
    # 디렉토리 설정
    management_db_dir = os.path.join(os.path.abspath(args.vector_db_dir), "fss_management")
    os.makedirs(management_db_dir, exist_ok=True)
    
    # 적응형 청커 초기화 및 처리
    management_chunker = FSSAdaptiveChunker(
        input_json=os.path.abspath(args.management_json),
        output_dir=management_db_dir,
        model_name=args.embed_model,
        use_openai_embeddings=args.use_openai_embeddings,
        default_chunk_size=args.chunk_size,
        default_chunk_overlap=args.chunk_overlap,
        use_faiss=args.use_faiss  # FAISS 사용 여부 추가
    )
    
    management_chunker.process()
    
    print("✅ 경영유의사항 처리 완료")


def run_interactive_mode(args):
    """대화형 모드 실행"""
    print("\n🤖 대화형 모드 시작...")
    
    # DB 유형 선택
    print("\n사용할 벡터 저장소를 선택하세요:")
    print("1. 제재 정보")
    print("2. 경영유의사항")
    choice = input("선택 (기본값: 1): ").strip() or "1"
    
    if choice == "2":
        db_path = os.path.join(os.path.abspath(args.vector_db_dir), "fss_management")
        print("경영유의사항 벡터 저장소 사용")
    else:
        db_path = os.path.join(os.path.abspath(args.vector_db_dir), "fss_sanctions")
        print("제재 정보 벡터 저장소 사용")
    
    # API 선택
    print("\n사용할 LLM API를 선택하세요:")
    print("1. OpenAI API (기본값)")
    print("2. Anthropic Claude API")
    api_choice = input("선택 (기본값: 1): ").strip() or "1"
    
    use_anthropic = (api_choice == "2")
    use_openai = (api_choice == "1")
    
    # RAG 시스템 초기화
    if use_anthropic:
        anthropic_api_key = os.getenv("ANTHROPIC_APIKEY")
        if not anthropic_api_key:
            anthropic_api_key = input("Anthropic API 키를 입력하세요: ")
        
        rag_system = FSSRagSystem(
            vector_db_path=db_path,
            embed_model_name=args.embed_model,
            use_openai_embeddings=args.use_openai_embeddings,
            use_anthropic=True,
            anthropic_api_key=anthropic_api_key,
            top_k=5,
            use_faiss=args.use_faiss,  # FAISS 사용 여부 추가
            use_openai_llm=False
        )
    else:  # OpenAI API 사용 (기본값)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("⚠️ 환경 변수 OPENAI_API_KEY를 찾을 수 없습니다.")
            return
            
        rag_system = FSSRagSystem(
            vector_db_path=db_path,
            embed_model_name=args.embed_model,
            use_openai_embeddings=args.use_openai_embeddings,
            llm_model_name="gpt-3.5-turbo",  # OpenAI 모델 사용
            top_k=5,
            use_faiss=args.use_faiss,  # FAISS 사용 여부 추가
            use_openai_llm=True  # OpenAI LLM 사용
        )
    
    # 대화형 모드 실행
    rag_system.interactive_mode()


def create_vector_stores(args):
    """벡터 저장소 생성 및 저장"""
    print("\n🔄 벡터 저장소 생성 시작...")
    
    # OpenAI API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️ OpenAI API 키가 설정되지 않았습니다.")
        return False
    
    try:
        # 제재 정보 벡터 저장소 생성
        if not args.skip_sanctions:
            sanctions_dir = os.path.join(args.vector_db_dir, "fss_sanctions")
            os.makedirs(sanctions_dir, exist_ok=True)
            
            # RAG 시스템으로 벡터 저장소 생성
            from rag_system import FSSRagSystem
            sanctions_rag = FSSRagSystem(
                vector_db_path=sanctions_dir,
                embed_model_name=args.embed_model,
                use_openai_embeddings=args.use_openai_embeddings,
                use_anthropic=False,
                use_faiss=args.use_faiss,
                create_from_json=args.sanctions_json
            )
            
            if not sanctions_rag.vector_store:
                print("❌ 제재 정보 벡터 저장소 생성 실패")
                return False
            
            # 벡터 저장소 정보 저장
            info_path = os.path.join(sanctions_dir, "vector_store_info.json")
            info = {
                "created_at": datetime.now().isoformat(),
                "embed_model": "text-embedding-3-large" if args.use_openai_embeddings else args.embed_model,
                "use_openai": args.use_openai_embeddings,
                "vector_store_type": "FAISS" if args.use_faiss else "Chroma"
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            
            print("✅ 제재 정보 벡터 저장소 생성 완료")
        
        # 경영유의사항 벡터 저장소 생성
        if not args.skip_management:
            management_dir = os.path.join(args.vector_db_dir, "fss_management")
            os.makedirs(management_dir, exist_ok=True)
            
            # RAG 시스템으로 벡터 저장소 생성
            from rag_system import FSSRagSystem
            management_rag = FSSRagSystem(
                vector_db_path=management_dir,
                embed_model_name=args.embed_model,
                use_openai_embeddings=args.use_openai_embeddings,
                use_anthropic=False,
                use_faiss=args.use_faiss,
                create_from_json=args.management_json
            )
            
            if not management_rag.vector_store:
                print("❌ 경영유의사항 벡터 저장소 생성 실패")
                return False
            
            # 벡터 저장소 정보 저장
            info_path = os.path.join(management_dir, "vector_store_info.json")
            info = {
                "created_at": datetime.now().isoformat(),
                "embed_model": "text-embedding-3-large" if args.use_openai_embeddings else args.embed_model,
                "use_openai": args.use_openai_embeddings,
                "vector_store_type": "FAISS" if args.use_faiss else "Chroma"
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            
            print("✅ 경영유의사항 벡터 저장소 생성 완료")
        
        print("\n🎉 벡터 저장소 생성 완료!")
        print("생성된 벡터 저장소를 GitHub에 커밋하세요.")
        return True
        
    except Exception as e:
        print(f"❌ 벡터 저장소 생성 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("🚀 금융감독원 제재/경영유의사항 RAG 파이프라인 시작")
    
    # 인자 파싱
    args = parse_arguments()
    
    # 경로 정규화
    args.sanctions_json = os.path.abspath(args.sanctions_json)
    args.management_json = os.path.abspath(args.management_json)
    args.vector_db_dir = os.path.abspath(args.vector_db_dir)
    
    # 필요한 디렉토리 생성
    os.makedirs(os.path.dirname(args.sanctions_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.management_json), exist_ok=True)
    os.makedirs(args.vector_db_dir, exist_ok=True)
    
    # OpenAI API 키 확인
    if args.use_openai_embeddings:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("⚠️ 환경 변수 OPENAI_API_KEY를 찾을 수 없습니다.")
            print("OpenAI 임베딩을 사용하려면 .env 파일에 OPENAI_API_KEY를 설정하세요.")
            print("HuggingFace 임베딩으로 전환합니다.")
            args.use_openai_embeddings = False
    
    # 샘플 데이터 생성 (필요한 경우)
    if args.use_sample_data:
        create_sample_data(args.sanctions_json, args.management_json)
    
    # 벡터 저장소 생성
    create_vector_stores(args)
    
    # 대화형 모드 실행
    if args.interactive:
        run_interactive_mode(args)
    
    print("\n🎉 파이프라인 처리 완료!")

if __name__ == "__main__":
    main() 