"""
Streamlit을 이용한 금융감독원 제재/경영유의사항 RAG 시스템 웹 인터페이스
- 벡터 저장소 기반 RAG 시스템 연결
- 챗봇 인터페이스 제공
- 검색 결과 시각화
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any, Union
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# RAG 시스템 가져오기
from rag_system import FSSRagSystem

# 페이지 설정
st.set_page_config(
    page_title="금융감독원 제재정보 챗봇",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 스타일 추가
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
}
.sub-header {
    font-size: 1.5rem;
    color: #3B82F6;
}
.source-card {
    background-color: #F3F4F6;
    border-left: 4px solid #3B82F6;
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
}
.source-title {
    font-weight: bold;
    color: #1E3A8A;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    font-size: 1.1rem;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.user-message {
    background-color: #E0F2FE;
    margin-left: 2rem;
    border-left: 4px solid #0EA5E9;
    font-weight: 500;
}
.bot-message {
    background-color: #F0FDF4;
    margin-right: 2rem;
    border-left: 4px solid #10B981;
    font-weight: 500;
}
.message-content {
    font-size: 1.1rem;
    color: #111827;
}
.message-role {
    font-weight: bold;
    margin-right: 0.5rem;
    color: #1F2937;
}
.system-message {
    background-color: #FEF3C7;
    margin: 1rem 0;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border-left: 4px solid #F59E0B;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db_path" not in st.session_state:
    st.session_state.vector_db_path = "./data/vector_db/fss_sanctions"
if "embed_model" not in st.session_state:
    st.session_state.embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
if "use_openai_embeddings" not in st.session_state:
    st.session_state.use_openai_embeddings = True  # 기본값을 True로 변경
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-3.5-turbo"  # OpenAI 모델로 변경
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "use_anthropic" not in st.session_state:
    st.session_state.use_anthropic = False  # 기본값을 False로 설정
if "use_faiss" not in st.session_state:
    st.session_state.use_faiss = True  # FAISS 사용 여부 (기본값 True)
if "use_openai_llm" not in st.session_state:
    st.session_state.use_openai_llm = True  # OpenAI LLM 사용 여부 (기본값 True)
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False

# 벡터 저장소 로드 함수 (최초 1회만 실행)
def load_vector_store():
    if not st.session_state.vector_store_loaded:
        with st.spinner("벡터 저장소 로드 중..."):
            try:
                # 벡터 저장소만 로드하는 RAG 시스템 생성
                st.session_state.rag_system = FSSRagSystem(
                    vector_db_path=st.session_state.vector_db_path,
                    embed_model_name=st.session_state.embed_model,
                    use_openai_embeddings=st.session_state.use_openai_embeddings,
                    use_anthropic=False,  # 항상 False로 설정 (LLM은 별도로 초기화)
                    use_faiss=st.session_state.use_faiss  # FAISS 사용 여부
                )
                
                # 벡터 저장소 로드 성공 여부 확인
                if st.session_state.rag_system.vector_store:
                    st.session_state.vector_store_loaded = True
                    return True
                else:
                    st.sidebar.error("벡터 저장소 로드에 실패했습니다. 경로를 확인해주세요.")
                    st.session_state.vector_store_loaded = False
                    return False
            except Exception as e:
                st.sidebar.error(f"벡터 저장소 로드 실패: {str(e)}")
                st.session_state.vector_store_loaded = False
                return False
    return True

# 사이드바에 RAG 시스템 설정
with st.sidebar:
    st.markdown("### 시스템 설정")
    
    # 벡터 저장소 선택
    vector_db_options = {
        "제재 정보": "./data/vector_db/fss_sanctions",
        "경영유의사항": "./data/vector_db/fss_management",
    }
    
    vector_db = st.selectbox(
        "벡터 저장소 선택",
        options=list(vector_db_options.keys()),
        index=0,
        key="vector_db_selector"
    )
    
    # 벡터 저장소 경로가 변경된 경우 재로드 필요
    if st.session_state.vector_db_path != vector_db_options[vector_db]:
        st.session_state.vector_db_path = vector_db_options[vector_db]
        st.session_state.vector_store_loaded = False
    
    # 임베딩 설정
    st.markdown("### 임베딩 설정")

    # OpenAI 임베딩 사용 여부 활성화
    use_openai = st.checkbox("OpenAI 임베딩 API 사용", value=st.session_state.use_openai_embeddings)

    if use_openai != st.session_state.use_openai_embeddings:
        st.session_state.use_openai_embeddings = use_openai
        st.session_state.vector_store_loaded = False  # 임베딩 모델이 변경되어 재로드 필요

    # OpenAI API 키 확인
    if st.session_state.use_openai_embeddings:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            st.success("✅ OpenAI API 키가 환경 변수에서 로드되었습니다.")
        else:
            st.error("❌ 환경 변수 OPENAI_API_KEY를 찾을 수 없습니다.")
            st.info("OpenAI API 키를 설정하거나 HuggingFace 임베딩을 사용하세요.")

    # 벡터 저장소 타입 선택
    vector_store_type = st.radio(
        "벡터 저장소 타입",
        ["FAISS", "Chroma"],
        index=0 if st.session_state.use_faiss else 1
    )

    use_faiss = (vector_store_type == "FAISS")
    if use_faiss != st.session_state.use_faiss:
        st.session_state.use_faiss = use_faiss
        st.session_state.vector_store_loaded = False  # 벡터 저장소 타입이 변경되어 재로드 필요

    # HuggingFace 임베딩 모델 선택 (OpenAI를 사용하지 않을 경우에만 표시)
    if not st.session_state.use_openai_embeddings:
        embed_model_options = {
            "MiniLM-L12-v2 (다국어)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "BGE-Large-Ko": "BAAI/bge-large-ko-v1.5",
        }

        embed_model = st.selectbox(
            "임베딩 모델",
            options=list(embed_model_options.keys()),
            index=0,
        )

        # 임베딩 모델이 변경된 경우 재로드 필요
        if st.session_state.embed_model != embed_model_options[embed_model]:
            st.session_state.embed_model = embed_model_options[embed_model]
            st.session_state.vector_store_loaded = False
    
    # 벡터 저장소 로드 버튼
    if not st.session_state.vector_store_loaded:
        if st.button("벡터 저장소 로드", type="primary"):
            if load_vector_store():
                st.sidebar.success("✅ 벡터 저장소 로드 완료")
    else:
        st.success("✅ 벡터 저장소가 로드되었습니다")
    
    st.markdown("---")
    
    # LLM 선택 - Anthropic API 또는 OpenAI API
    st.markdown("### LLM 설정")

    llm_provider = st.radio(
        "LLM 제공자",
        ["OpenAI", "Anthropic"],
        index=0 if not st.session_state.use_anthropic else 1
    )

    st.session_state.use_anthropic = (llm_provider == "Anthropic")
    st.session_state.use_openai_llm = (llm_provider == "OpenAI")

    # OpenAI 모델 선택
    if st.session_state.use_openai_llm:
        openai_models = {
            "GPT-3.5 Turbo": "gpt-3.5-turbo",
            "GPT-4 Turbo": "gpt-4-turbo",
            "GPT-4o": "gpt-4o",
        }
        
        openai_model = st.selectbox(
            "OpenAI 모델",
            options=list(openai_models.keys()),
            index=0,
        )
        
        st.session_state.llm_model = openai_models[openai_model]
        
        # OpenAI API 키 확인
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            st.success("✅ OpenAI API 키가 환경 변수에서 로드되었습니다.")
        else:
            st.error("❌ 환경 변수 OPENAI_API_KEY를 찾을 수 없습니다.")

    # Anthropic API 키 확인 및 표시
    elif st.session_state.use_anthropic:
        anthropic_api_key = os.getenv("ANTHROPIC_APIKEY")
        if anthropic_api_key:
            st.success("✅ Anthropic API 키가 환경 변수에서 로드되었습니다.")
        else:
            st.error("❌ 환경 변수 ANTHROPIC_APIKEY를 찾을 수 없습니다.")

    # 검색 문서 수
    # top_k = st.slider("검색 문서 수", min_value=1, max_value=10, value=5)
    # st.session_state.top_k = top_k
    
    # LLM 초기화 버튼
    if st.session_state.vector_store_loaded:
        if st.button("LLM 초기화", type="primary"):
            with st.spinner("LLM 초기화 중..."):
                try:
                    if st.session_state.use_anthropic:
                        anthropic_api_key = os.getenv("ANTHROPIC_APIKEY")
                        if not anthropic_api_key:
                            st.error("환경 변수 ANTHROPIC_APIKEY를 찾을 수 없습니다.")
                        else:
                            # 기존 RAG 시스템에 LLM 초기화
                            st.session_state.rag_system.use_anthropic = True
                            st.session_state.rag_system.anthropic_api_key = anthropic_api_key
                            st.session_state.rag_system.use_openai_llm = False
                            st.session_state.rag_system.use_faiss = st.session_state.use_faiss
                            st.session_state.rag_system.initialize_llm()
                            if st.session_state.rag_system.llm:
                                st.success("Anthropic Claude API로 LLM 초기화 완료!")
                            else:
                                st.error("LLM 초기화 실패. 로그를 확인해주세요.")
                    else:
                        # OpenAI API 사용
                        openai_api_key = os.getenv("OPENAI_API_KEY")
                        if not openai_api_key:
                            st.error("환경 변수 OPENAI_API_KEY를 찾을 수 없습니다.")
                        else:
                            # 기존 RAG 시스템에 LLM 초기화
                            st.session_state.rag_system.use_anthropic = False
                            st.session_state.rag_system.use_openai_llm = True
                            st.session_state.rag_system.llm_model_name = st.session_state.llm_model
                            st.session_state.rag_system.use_faiss = st.session_state.use_faiss
                            st.session_state.rag_system.initialize_llm()
                            if st.session_state.rag_system.llm:
                                st.success(f"OpenAI {st.session_state.llm_model}로 LLM 초기화 완료!")
                            else:
                                st.error("LLM 초기화 실패. API 키를 확인하고 로그를 확인해주세요.")
                except Exception as e:
                    st.error(f"LLM 초기화 실패: {str(e)}")

    st.markdown("---")
    
    # 빠른 질문 예시
    st.markdown("### 질문 예시")
    example_questions = [
        "최근 1년간 제재받은 은행 알려줘",
        "신용정보법 위반 사례 있어?",
        "주의 조치 받은 금융사 중 내부통제 문제였던 경우?",
        "과징금 부과된 보험사 알려줘",
        "전자금융 관련 경영유의사항은?"
    ]
    
    # 예시 질문을 세션 상태에 추가하는 함수
    def set_example_question(question):
        # 동일한 질문이 이미 마지막으로 입력된 경우 중복 방지
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            last_user_msg = None
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break
            
            if last_user_msg == question:
                # 이미 동일한 질문이 존재하면 다시 처리하지 않음
                return
        
        st.session_state.chat_input = question
        st.rerun()
    
    for q in example_questions:
        if st.sidebar.button(q):
            set_example_question(q)

# 메인 화면
st.markdown('<h1 class="main-header">금융감독원 제재정보 챗봇</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">금융 제재 및 경영유의사항 질의응답 시스템</h2>', unsafe_allow_html=True)

# 시스템 상태 확인
system_status_shown = False
if not st.session_state.vector_store_loaded:
    st.markdown('<div class="system-message">⚠️ 사이드바에서 벡터 저장소를 로드해주세요.</div>', unsafe_allow_html=True)
    system_status_shown = True
elif st.session_state.rag_system and not st.session_state.rag_system.llm:
    st.markdown('<div class="system-message">⚠️ 사이드바에서 LLM을 초기화해주세요.</div>', unsafe_allow_html=True)
    system_status_shown = True

# 벡터 저장소 자동 로드 시도는 제거 (중복 메시지 방지)

# 채팅 기록 표시
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.container():
            st.markdown(f'<div class="chat-message user-message"><span class="message-role">🧑‍💼</span><span class="message-content">{message["content"]}</span></div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f'<div class="chat-message bot-message"><span class="message-role">🤖</span><span class="message-content">{message["content"]}</span></div>', unsafe_allow_html=True)
            
            if "sources" in message and message["sources"]:
                with st.expander("참고 문서 보기"):
                    for j, source in enumerate(message["sources"][:3]):
                        metadata = source.get("metadata", {})
                        
                        # DB 타입에 따라 날짜와 유형을 결정합니다.
                        date_to_display = metadata.get('sanction_date') or metadata.get('disclosure_date', 'N/A')
                        type_to_display = metadata.get('sanction_type') or metadata.get('management_type', 'N/A')

                        st.markdown(f"""
                        <div class="source-card">
                            <p class="source-title">출처 #{j+1}</p>
                            <p><strong>기관:</strong> {metadata.get('institution', 'N/A')}</p>
                            <p><strong>날짜:</strong> {date_to_display}</p>
                            <p><strong>유형:</strong> {type_to_display}</p>
                            <p>{source.get('content', '')[:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)

# 사용자 입력
user_input = st.chat_input("질문을 입력하세요...", key="chat_input")

# 사용자 입력 처리
if user_input:
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 벡터 저장소와 LLM 상태 확인
    if not st.session_state.vector_store_loaded:
        error_msg = "벡터 저장소가 로드되지 않았습니다. 사이드바에서 '벡터 저장소 로드' 버튼을 클릭해주세요."
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": error_msg,
            "sources": []
        })
        st.error(error_msg)
        st.rerun()
    elif st.session_state.rag_system and not st.session_state.rag_system.llm:
        error_msg = "LLM이 초기화되지 않았습니다. 사이드바에서 'LLM 초기화' 버튼을 클릭해주세요."
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": error_msg,
            "sources": []
        })
        st.error(error_msg)
        st.rerun()
    else:
        # 시스템 응답 생성
        with st.spinner("답변 생성 중..."):
            try:
                response = st.session_state.rag_system.answer_question(user_input)
                
                answer = response.get("answer", "")
                sources = response.get("sources", [])
                
                # 챗봇 응답 추가
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
                # 화면 갱신
                st.rerun()
                
            except Exception as e:
                # 오류 메시지를 채팅 기록에 추가하고 무한 루프 방지
                error_msg = f"오류가 발생했습니다: {str(e)}"
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "sources": []
                })
                st.error(error_msg)

# 대화 기록 지우기 버튼
if st.session_state.chat_history and st.button("대화 기록 지우기"):
    st.session_state.chat_history = []
    st.rerun() 