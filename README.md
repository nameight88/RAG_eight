# 금융감독원 제재/경영유의사항 RAG 시스템

금융감독원 제재 내역 및 경영유의사항을 활용한 검색 증강 생성(RAG) 시스템입니다. 이 시스템은 PDF, HWP, HWPX 파일에서 추출한 금융감독원 문서를 파싱하고, 벡터 저장소에 임베딩하여 질의응답 시스템을 구축합니다.

## 🌟 주요 기능

- **문서 파싱**: PDF, HWP, HWPX 파일에서 텍스트 및 구조화된 정보 추출
- **적응형 청킹**: 문서 구조와 내용에 따라 최적화된 방식으로 청크 분할
- **시맨틱 청킹**: 의미 기반으로 문서를 분할하여 질의응답 품질 향상
- **벡터 임베딩**: 문서 청크를 벡터로 변환하여 벡터 저장소에 저장
- **질의응답 시스템**: LLM을 활용한 RAG 기반 질의응답 제공
- **웹 인터페이스**: Streamlit을 이용한 사용자 친화적 웹 인터페이스

## 📁 프로젝트 구조

```
parser_py/
├── adaptive_chunker.py       # 경영유의사항 문서 적응형 청킹 모듈
├── semantic_chunker.py       # 제재 문서 시맨틱 청킹 모듈
├── rag_system.py             # RAG 질의응답 시스템
├── streamlit_app.py          # Streamlit 웹 인터페이스
├── run_pipeline.py           # 전체 파이프라인 실행 스크립트
├── requirements.txt          # 필요 라이브러리
├── document_parser.py        # 문서 파싱 기본 클래스
├── fss_crawler_*.py          # 금감원 웹사이트 크롤러
└── fss_doc_*_parser_*.py     # 특정 문서 유형 파서
```

## 🛠️ 설치 방법

1. 가상환경 생성 및 활성화 (선택사항이지만 권장):

```bash
# 가상환경 생성
python -m venv rag_env

# 가상환경 활성화
# Windows
rag_env\Scripts\activate
# macOS/Linux
source rag_env/bin/activate
```

2. 필요 패키지 설치:

```bash
pip install -r requirements.txt
```

3. 추가 필수 패키지 설치:

```bash
# Accelerate 설치 (LLM 가속을 위해)
pip install "accelerate>=0.26.0"

# Watchdog 설치 (Streamlit 파일 감시를 위해)
pip install watchdog
```

4. HWP 파일 처리를 위한 추가 패키지 설치:

```bash
pip install --no-deps hwp5
```

5. 시스템 구성:
   - Python 3.8 이상
   - 최소 8GB RAM
   - 권장: CUDA 지원 GPU (LLM 실행 시)

## 🚀 사용 방법

### 1. 데이터 디렉토리 구조 확인

```bash
# 데이터 디렉토리 생성
mkdir -p data/vector_db/fss_sanctions data/vector_db/fss_management
```

### 2. 벡터 저장소 구축

먼저 문서를 청킹하고 벡터 저장소를 구축합니다:

```bash
# 제재 문서 벡터 저장소 구축
python semantic_chunker.py

# 경영유의사항 문서 벡터 저장소 구축
python adaptive_chunker.py
```

### 3. RAG 시스템 실행 (터미널)

터미널에서 대화형으로 RAG 시스템을 실행할 수 있습니다:

```bash
python rag_system.py
```

메시지가 표시되면 사용할 벡터 DB 유형을 선택합니다 (1: 제재정보, 2: 경영유의사항).

### 4. Streamlit 웹 인터페이스 실행

웹 인터페이스를 통해 RAG 시스템을 사용하려면:

```bash
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 웹 인터페이스를 사용할 수 있습니다.

### 5. 전체 파이프라인 실행

크롤링부터 파싱, 청킹, 벡터 저장소 구축까지 전체 파이프라인을 실행하려면:

```bash
python run_pipeline.py
```

## 💬 질의응답 예시

시스템은 다음과 같은 질문에 답변할 수 있습니다:

- "최근 1년간 제재받은 은행 알려줘"
- "신용정보법 위반 사례 있어?"
- "주의 조치 받은 금융사 중 내부통제 문제였던 경우?"
- "과징금 부과된 보험사 알려줘"
- "전자금융 관련 경영유의사항은?"

## 📊 시스템 구성도

```
[문서 파일(PDF/HWP/HWPX)] → [문서 파싱] → [청킹(적응형/시맨틱)] → [벡터 임베딩]
                                                              ↓
[사용자 질문] → [쿼리 처리] → [벡터 검색] → [관련 문서 검색] → [LLM 답변 생성] → [사용자 응답]
```

## 📝 참고사항

- 벡터 저장소는 `./data/vector_db/` 디렉토리에 저장됩니다
- 기본 LLM 모델은 `beomi/llama-2-ko-7b`입니다
- 기본 임베딩 모델은 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`입니다

## ⚠️ 문제 해결

### LLM 로드 관련 문제

LLM 로드 시 다음과 같은 오류가 발생하면:
```
⚠️ LLM 로드 실패: Using `low_cpu_mem_usage=True`, a `device_map` or a `tp_plan` requires Accelerate
```

Accelerate 패키지를 설치하세요:
```bash
pip install "accelerate>=0.26.0"
```

### 메모리 부족 문제

LLM 로드 시 메모리 부족 오류가 발생하면:
1. 더 작은 LLM 모델을 선택하세요 (예: 1.3B 크기 모델)
2. `rag_system.py` 파일에서 `model_kwargs` 에 `device_map="auto"` 대신 `device="cpu"` 로 변경하세요
3. 가능하면 GPU가 있는 시스템에서 실행하세요

### Streamlit 관련 문제

Streamlit 실행 시 문제가 발생하면:
1. Watchdog 패키지 설치: `pip install watchdog`
2. 정확한 경로에서 실행되고 있는지 확인하세요: `cd /path/to/parser_py`
3. 직접 Python 모듈로 실행: `python -m streamlit run streamlit_app.py`

### 벡터 저장소 디렉토리 문제

벡터 저장소가 존재하지 않는다는 오류가 나타나면:
1. 데이터 디렉토리 구조 확인: `mkdir -p data/vector_db/fss_sanctions data/vector_db/fss_management`
2. 청킹 모듈을 실행하여 벡터 저장소 구축: `python semantic_chunker.py`

## 🧰 개선 방향

- 🔄 다양한 LLM 모델 지원
- 📈 검색 알고리즘 개선
- 🧩 더 정교한 청킹 전략
- 🌐 더 다양한 문서 소스 지원 