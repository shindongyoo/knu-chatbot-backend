# app/search_engine.py

import os
from dotenv import load_dotenv

# LangChain에서 FAISS와 임베딩 모델을 사용하기 위한 라이브러리
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document # LangChain의 Document 객체 타입 힌트용

# .env 파일 로드
load_dotenv()

# --- 1. 경로 설정 및 임베딩 모델 초기화 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vector DB 생성 시 사용했던 것과 "동일한" 임베딩 모델을 LangChain 방식으로 초기화합니다.
# (이 모델은 DB를 만들 때 사용한 모델과 반드시 일치해야 합니다!)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# --- 2. 각 Vector DB를 로딩합니다 ---

# 공지사항 DB 로딩
NOTICES_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'notices')
notices_db = None # 초기값 설정
try:
    notices_db = FAISS.load_local(
        folder_path=NOTICES_DB_DIR,
        embeddings=embeddings,
        index_name="notices_title_index",  # 🚨 'notices_title_index.faiss' 파일 이름 (확장자 제외)
        allow_dangerous_deserialization=True
    )
    print("✅ Notices Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Notices Vector DB 로딩 실패: {e}")
    # vector_db = None  # 주석 처리 또는 제거

# 취업 정보 DB 로딩
JOBS_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'jobs')
jobs_db = None # 초기값 설정
try:
    jobs_db = FAISS.load_local(
        folder_path=JOBS_DB_DIR,
        embeddings=embeddings,
        index_name="jobs_openai_index", # 🚨 'jobs_openai_index.faiss' 파일 이름 (확장자 제외)
        allow_dangerous_deserialization=True
    )
    print("✅ Jobs Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Jobs Vector DB 로딩 실패: {e}")
    # vector_db = None  # 주석 처리 또는 제거


# --- 3. 질문의 의도를 파악하여 DB를 선택하는 라우터 함수 ---
def route_query_to_db(query: str) -> FAISS:
    """
    사용자 질문의 키워드를 기반으로 적절한 Vector DB를 선택하여 반환합니다.
    """
    job_keywords = ["취업", "인턴", "채용", "회사", "직무", "자소서", "면접", "구인", "모집", "공고", "일자리"]
    
    # 질문에 취업 관련 키워드가 포함되어 있는지 확인
    if any(keyword in query for keyword in job_keywords):
        if jobs_db:
            print(f"[🔍 DB 라우팅] '{query}' -> 취업 정보 DB 선택")
            return jobs_db
        else:
            print("[🔍 DB 라우팅] 취업 정보 DB가 로드되지 않아 공지사항 DB 사용")
            return notices_db # 취업 DB가 없으면 공지사항 DB라도 사용
    else:
        if notices_db:
            print(f"[🔍 DB 라우팅] '{query}' -> 공지사항 DB 선택")
            return notices_db
        else:
            print("[🔍 DB 라우팅] 공지사항 DB가 로드되지 않아 취업 정보 DB 사용")
            return jobs_db # 공지사항 DB가 없으면 취업 DB라도 사용


# --- 4. 메인 검색 함수 ---
def search_similar_documents(query: str, top_k: int = 5):
    """
    라우팅된 Vector DB에서 query와 유사한 문서를 검색하여,
    LangChain RAG 체인에 적합한 형태로 context를 반환합니다.
    """
    # 4-1. 라우터를 통해 어떤 DB를 검색할지 결정합니다.
    selected_db = route_query_to_db(query)
    
    if not selected_db:
        # 두 DB 모두 로딩되지 않았을 경우
        print("경고: 로드된 Vector DB가 없습니다.")
        return "관련 정보를 찾을 수 없습니다 (Vector DB 로딩 실패).", []

    # 4-2. 선택된 DB에서만 검색을 수행합니다.
    results: list[Document] = selected_db.similarity_search(query, k=top_k)
    
    # 4-3. RAG 프롬프트에 들어갈 context 문자열과 field_names를 생성합니다.
    context = ""
    field_names = set()
    for doc in results:
        # Document 객체의 page_content 속성에 본문이, metadata 속성에 출처 정보가 들어있습니다.
        context += f"- 문서 제목: {doc.metadata.get('title', '제목 없음')}\n"
        context += f"  - 출처: {doc.metadata.get('source', '출처 없음')}\n"
        context += f"  - 내용: {doc.page_content}\n---\n"
        field_names.update(doc.metadata.keys()) # 모든 문서의 메타데이터 키를 수집

    return context, list(field_names)


# --------------------------------------------------------------------------
# 참고: LLM을 이용한 요약/재정렬 함수 (기존 코드 유지)
# 이 함수는 현재 메인 RAG 체인에서는 사용되지 않지만, 필요시 별도로 호출하여 사용할 수 있는 유용한 유틸리티입니다.
# 사용하려면 OpenAI 클라이언트 초기화가 필요합니다.
# import openai
# from tenacity import retry, stop_after_attempt, wait_exponential
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def get_llm_summary(query, results):
#     """검색 결과를 LLM이 맥락을 고려해 요약하고 재정렬합니다."""
#     # ... (기존 요약 함수 코드)
# --------------------------------------------------------------------------