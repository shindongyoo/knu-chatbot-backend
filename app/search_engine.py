# app/search_engine.py

import os
from dotenv import load_dotenv

# LangChain에서 FAISS와 임베딩 모델을 사용하기 위한 라이브러리
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# .env 파일 로드
load_dotenv()

# --- 1. 경로 설정 수정 ---
# 현재 파일(search_engine.py)의 위치를 기준으로 경로를 설정하여, 어디서 실행하든 정확한 위치를 찾게 합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store')

# --- 2. LangChain 컴포넌트 사용 ---
# Vector DB 생성 시 사용했던 것과 "동일한" 임베딩 모델을 LangChain 방식으로 초기화합니다.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# --- 3. Vector DB 로딩 ---
# LangChain의 FAISS.load_local을 사용하면 index와 metadata(pkl)를 한 번에 안전하게 불러옵니다.
try:
    vector_db = FAISS.load_local(
        folder_path=DB_DIR,
        embeddings=embeddings,
        index_name="notices",  # .index와 .pkl 파일의 이름 (확장자 제외)
        allow_dangerous_deserialization=True
    )
    print("✅ Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Vector DB 로딩 실패: {e}")
    vector_db = None

# --- 4. 검색 함수 간소화 ---
# FastAPI 엔드포인트에서 직접 사용할 함수입니다.
def search_similar_documents(query: str, top_k: int = 5):
    """
    로드된 Vector DB에서 query와 유사한 문서를 검색하여,
    LangChain RAG 체인에 적합한 형태로 context를 반환합니다.
    """
    if not vector_db:
        return "Vector DB가 로드되지 않았습니다.", []

    # vector_db.similarity_search는 LangChain의 Document 객체 리스트를 반환합니다.
    results = vector_db.similarity_search(query, k=top_k)
    
    # RAG 프롬프트에 들어갈 context 문자열을 생성합니다.
    # Document 객체의 page_content 속성에 본문이, metadata 속성에 출처 정보가 들어있습니다.
    context = ""
    field_names = set()
    for doc in results:
        context += f"- 문서 제목: {doc.metadata.get('title', '제목 없음')}\n"
        context += f"  - 출처: {doc.metadata.get('source', '출처 없음')}\n"
        context += f"  - 내용: {doc.page_content}\n---\n"
        field_names.update(doc.metadata.keys())

    return context, list(field_names)


# --------------------------------------------------------------------------
# 참고: LLM을 이용한 요약/재정렬 함수 (기존 코드 유지)
# 이 함수는 현재 메인 RAG 체인에서는 사용되지 않지만, 필요시 별도로 호출하여 사용할 수 있는 유용한 유틸리티입니다.
# 사용하려면 OpenAI 클라이언트 초기화가 필요합니다.
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# @retry(...)
# def get_llm_summary(query, results):
#     ... (기존 요약 함수 코드)
# --------------------------------------------------------------------------