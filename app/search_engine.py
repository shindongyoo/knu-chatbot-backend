# app/search_engine.py (MongoDB + Vector DB 하이브리드 검색 최종본)

import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import pickle

# main.py에서 초기화된 MongoDB 객체를 가져옵니다.
from app.main import chatbot_db 

load_dotenv()

# --- 1. Vector DB 로딩 및 임베딩 모델 초기화 ---

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def load_vector_db_manually(folder_path, index_name):
    faiss_path = os.path.join(folder_path, f"{index_name}.faiss")
    pkl_path = os.path.join(folder_path, f"{index_name}.pkl")
    if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError(f"'{folder_path}'에서 DB 파일을 찾을 수 없습니다: {index_name}")
    
    index = faiss.read_index(faiss_path)
    with open(pkl_path, "rb") as f:
        docs_data = pickle.load(f)
    
    documents = [Document(page_content=doc.pop('content', ''), metadata=doc) for doc in docs_data]
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    
    return LangChainFAISS(
        embedding=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

# Vector DB 로딩 (members_db는 제외)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
notices_title_db = None
notices_content_db = None
jobs_db = None

try:
    NOTICES_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'notices')
    notices_title_db = load_vector_db_manually(NOTICES_DB_DIR, "notices_title_index")
    print("✅ Notices (제목) Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Notices (제목) Vector DB 로딩 실패: {e}")

try:
    NOTICES_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'notices')
    notices_content_db = load_vector_db_manually(NOTICES_DB_DIR, "notices_content_index")
    print("✅ Notices (본문) Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Notices (본문) Vector DB 로딩 실패: {e}")

try:
    JOBS_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'jobs')
    jobs_db = load_vector_db_manually(JOBS_DB_DIR, "jobs_openai_index")
    print("✅ Jobs Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Jobs Vector DB 로딩 실패: {e}")


# --- 2. MongoDB에서 구성원 정보 검색 함수 ---

def search_members_in_mongodb(query: str):
    """
    질문에서 인물 이름을 추출하여 MongoDB에서 검색하고, 결과를 context 문자열로 포맷팅합니다.
    """
    match = re.search(r'([\w가-힣]{2,4})\s*(교수님|교수|조교|선생님)', query)
    if not match:
        return None

    name_to_search = match.group(1)
    members = list(chatbot_db.members.find({"name": {"$regex": name_to_search}}))
    
    if members:
        context = "### 검색된 구성원 정보:\n"
        for member in members:
            context += f"- 이름: {member.get('name', '정보 없음')}\n"
            context += f"  - 직위: {member.get('position', '정보 없음')}\n"
            context += f"  - 연구실: {member.get('lab', '정보 없음')}\n"
            context += f"  - 이메일: {member.get('email', '정보 없음')}\n"
            context += f"  - 전공 분야: {member.get('major', '정보 없음')}\n---\n"
        return context
    
    return None

# --- 3. 메인 검색 함수 (라우터 로직 통합) ---

def search_similar_documents(query: str, top_k: int = 5):
    """
    질문의 종류를 파악(라우팅)하여 MongoDB 또는 Vector DB에서 정보를 검색합니다.
    """
    member_keywords = ["교수", "교수님", "연구실", "이메일", "연락처", "조교", "선생님", "사무실", "위치"]
    job_keywords = ["취업", "인턴", "채용", "회사", "직무", "자소서", "면접"]

    # 1순위: 구성원 관련 질문이면 MongoDB에서 먼저 검색
    if any(keyword in query for keyword in member_keywords):
        print(f"[🔍 DB 라우팅] '{query}' -> MongoDB 구성원 검색 시도")
        mongo_context = search_members_in_mongodb(query)
        if mongo_context:
            return mongo_context, ['name', 'position', 'lab', 'email', 'major']

    # 2순위 또는 MongoDB에서 못 찾은 경우: Vector DB 검색
    selected_dbs = None
    if any(keyword in query for keyword in job_keywords):
        print(f"[🔍 DB 라우팅] '{query}' -> 취업 정보 Vector DB 선택")
        selected_dbs = (jobs_db,) if jobs_db else (notices_content_db,)
    else:
        print(f"[🔍 DB 라우팅] '{query}' -> 공지사항 Vector DB 선택 (제목+본문)")
        selected_dbs = (notices_title_db, notices_content_db)
    
    if not any(selected_dbs):
        return "관련 정보를 찾을 수 없습니다 (DB 로딩 실패).", []

    all_results = []
    for db in selected_dbs:
        if db:
            results = db.similarity_search_with_score(query, k=top_k)
            all_results.extend(results)
    
    unique_results = {}
    for doc, score in all_results:
        if doc.page_content not in unique_results or score < unique_results[doc.page_content][1]:
            unique_results[doc.page_content] = (doc, score)
            
    sorted_results = sorted(unique_results.values(), key=lambda item: item[1])
    
    context = ""
    field_names = set()
    for doc, score in sorted_results[:top_k]:
        context += f"유사도 점수: {score:.4f}\n"
        context += f"문서 제목: {doc.metadata.get('title', '제목 없음')}\n"
        context += f"  - 출처: {doc.metadata.get('source', '출처 없음')}\n"
        context += f"  - 내용: {doc.page_content}\n---\n"
        field_names.update(doc.metadata.keys())
        
    return context, list(field_names)