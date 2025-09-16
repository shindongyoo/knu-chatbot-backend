# app/search_engine.py (최종 완성본)

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
    """
    비표준 .pkl 파일을 읽기 위해, FAISS 인덱스와 문서를 수동으로 로드하고
    LangChain FAISS 객체를 재구성하는 함수.
    """
    faiss_path = os.path.join(folder_path, f"{index_name}.faiss")
    pkl_path = os.path.join(folder_path, f"{index_name}.pkl")

    if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError(f"'{folder_path}'에서 DB 파일을 찾을 수 없습니다.")

    # 1. FAISS 인덱스를 직접 로드
    index = faiss.read_index(faiss_path)

    # 2. .pkl 파일을 직접 로드 (딕셔너리 리스트)
    with open(pkl_path, "rb") as f:
        docs_data = pickle.load(f)

    # 3. 딕셔너리 리스트를 LangChain의 Document 객체 리스트로 변환
    documents = []
    for doc_dict in docs_data:
        # 'content' 키를 page_content로, 나머지를 metadata로 사용
        content = doc_dict.pop('content', '')
        metadata = doc_dict
        documents.append(Document(page_content=content, metadata=metadata))

    # 4. LangChain의 docstore 형식으로 재구성
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
        
    # 5. 수동으로 불러온 컴포넌트들로 LangChain FAISS 객체 최종 조립
    return LangChainFAISS(
        embedding_function=embeddings.embed_query, # 최신 버전에 맞게 수정
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

# --- 2. 각 Vector DB를 로딩합니다 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
notices_db = None
jobs_db = None

try:
    NOTICES_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'notices')
    notices_db = load_vector_db_manually(NOTICES_DB_DIR, "notices_title_index")
    print("✅ Notices Vector DB 로딩 성공.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Notices Vector DB 로딩 실패: {e}")

try:
    JOBS_DB_DIR = os.path.join(BASE_DIR, '..', 'vector_store', 'jobs')
    jobs_db = load_vector_db_manually(JOBS_DB_DIR, "jobs_openai_index")
    print("✅ Jobs Vector DB 로딩 성공.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Jobs Vector DB 로딩 실패: {e}")


# --- 3. 라우터 및 검색 함수 (기존 코드와 거의 동일, selected_db 타입 힌트만 추가) ---
def route_query_to_db(query: str) -> LangChainFAISS | None:
    job_keywords = ["취업", "인턴", "채용", "회사", "직무", "자소서", "면접", "구인", "모집", "공고", "일자리"]
    if any(keyword in query for keyword in job_keywords):
        return jobs_db if jobs_db else notices_db
    else:
        return notices_db if notices_db else jobs_db

def search_similar_documents(query: str, top_k: int = 5):
    selected_db = route_query_to_db(query)
    if not selected_db:
        return "관련 정보를 찾을 수 없습니다 (Vector DB 로딩 실패).", []
    
    # similarity_search_with_score 를 사용하여 점수도 확인 (디버깅에 유용)
    results = selected_db.similarity_search_with_score(query, k=top_k)
    
    context = ""
    field_names = set()
    for doc, score in results:
        context += f"유사도 점수: {score:.4f}\n"
        context += f"문서 제목: {doc.metadata.get('title', '제목 없음')}\n"
        context += f"출처: {doc.metadata.get('source', '출처 없음')}\n"
        context += f"내용: {doc.page_content}\n---\n"
        field_names.update(doc.metadata.keys())
    return context, list(field_names)