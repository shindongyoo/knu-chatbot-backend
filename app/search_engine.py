# app/search_engine.py (두 DB 모두 검색하는 최종 버전)

import os
import pickle
import faiss
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv()

# --- 1. 임베딩 모델 초기화 ---
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
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

# --- 2. 모든 Vector DB를 로딩합니다 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
notices_title_db = None
notices_content_db = None # 본문 DB를 위한 새 변수
jobs_db = None
members_db = None

# 공지사항 '제목' DB 로딩
try:
    NOTICES_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'notices')
    notices_title_db = load_vector_db_manually(NOTICES_DB_DIR, "notices_title_index")
    print("✅ Notices (제목) Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Notices (제목) Vector DB 로딩 실패: {e}")

# 공지사항 '본문' DB 로딩 (새로 추가)
try:
    NOTICES_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'notices')
    # notices_content_index.faiss 와 notices_content_index.pkl 파일이 필요합니다.
    # 이전 단계에서 notices.index -> notices_content_index.faiss
    # notices_metadata.pkl -> notices_content_index.pkl 로 이름을 변경했습니다.
    notices_content_db = load_vector_db_manually(NOTICES_DB_DIR, "notices_content_index")
    print("✅ Notices (본문) Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Notices (본문) Vector DB 로딩 실패: {e}")

# 취업 정보 DB 로딩
try:
    JOBS_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'jobs')
    jobs_db = load_vector_db_manually(JOBS_DB_DIR, "jobs_openai_index")
    print("✅ Jobs Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Jobs Vector DB 로딩 실패: {e}")
    
try:
    MEMBERS_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'members')
    members_db = load_vector_db_manually(MEMBERS_DB_DIR, "members_index")
    print("✅ Members Vector DB 로딩 성공.")
except Exception as e:
    print(f"❌ Members Vector DB 로딩 실패: {e}")


# --- 3. 라우터 함수 수정 ---
def route_query_to_db(query: str):
    """
    사용자 질문의 키워드를 분석하여 가장 적절한 Vector DB(들)을 선택하여 반환합니다.
    """
    # 1. 각 데이터베이스의 역할을 명확히 하는 키워드 목록 정의
    
    # members_db 키워드: 인물, 장소, 연락처 등 고유명사 정보
    member_keywords = [
        "교수", "교수님", "조교", "교직원", "연구실", "사무실", "이메일", 
        "연락처", "전화번호", "위치", "어디", "호관", "호실", "강의실"
    ]
    
    # jobs_db 키워드: 채용, 경력 관련 정보
    job_keywords = [
        "취업", "인턴", "채용", "회사", "직무", "자소서", "면접", 
        "구인", "모집", "공고", "일자리", "커리어", "경력"
    ]

    # notices_db 키워드: 학사, 행정, 장학금 등 일반 정보
    notice_keywords = [
        "공지", "장학금", "등록금", "신청", "기간", "제출", "마감", "안내",
        "졸업", "요건", "학점", "수강", "과목", "교과", "과정", "이수", "요람"
    ]

    # 2. 라우팅 로직: 더 구체적인 질문부터 확인 (교수/장소 -> 취업 -> 일반공지)

    # 1순위: 구성원 또는 장소 관련 질문인지 확인
    if any(keyword in query for keyword in member_keywords):
        print(f"[🔍 DB 라우팅] '{query}' -> 구성원/장소 정보 DB 선택")
        return (members_db,) if members_db else (notices_content_db,)

    # 2순위: 취업 관련 질문인지 확인
    if any(keyword in query for keyword in job_keywords):
        print(f"[🔍 DB 라우팅] '{query}' -> 취업 정보 DB 선택")
        return (jobs_db,) if jobs_db else (notices_content_db,)
    
    # 3순위: 일반 공지 및 학사 관련 질문은 공지사항 DB 검색 (제목+본문)
    # (위 두 경우에 해당하지 않는 모든 질문은 여기로 옵니다)
    print(f"[🔍 DB 라우팅] '{query}' -> 공지사항 DB 선택 (제목+본문)")
    return (notices_title_db, notices_content_db)

# --- 4. 메인 검색 함수 수정 ---
def search_similar_documents(query: str, top_k: int = 5):
    # 4-1. 라우터를 통해 검색할 DB(들)을 결정합니다.
    selected_dbs = route_query_to_db(query)
    
    if not any(selected_dbs):
        print("경고: 로드된 Vector DB가 없습니다.")
        return "관련 정보를 찾을 수 없습니다 (Vector DB 로딩 실패).", []

    # 4-2. 선택된 모든 DB에서 검색을 수행하고 결과를 합칩니다.
    all_results = []
    for db in selected_dbs:
        if db:
            results = db.similarity_search_with_score(query, k=top_k)
            all_results.extend(results)
    
    # 4-3. 중복을 제거하고 점수가 높은 순으로 정렬합니다.
    unique_results = {}
    for doc, score in all_results:
        # page_content를 기준으로 중복을 제거합니다. 동일 문서면 점수가 더 좋은(낮은) 것으로 유지합니다.
        if doc.page_content not in unique_results or score < unique_results[doc.page_content][1]:
            unique_results[doc.page_content] = (doc, score)
            
    # 점수가 낮은 순 (유사도가 높은 순)으로 정렬
    sorted_results = sorted(unique_results.values(), key=lambda item: item[1])
    
    # 4-4. 최종 context를 생성합니다.
    context = ""
    field_names = set()
    for doc, score in sorted_results[:top_k]: # 최종적으로 top_k개만 사용
        context += f"유사도 점수: {score:.4f}\n"
        context += f"문서 제목: {doc.metadata.get('title', '제목 없음')}\n"
        context += f"출처: {doc.metadata.get('source', '출처 없음')}\n"
        context += f"내용: {doc.page_content}\n---\n"
        field_names.update(doc.metadata.keys())
        
    return context, list(field_names)