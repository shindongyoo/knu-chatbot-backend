import os
import re
import faiss
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from app.database import chatbot_db

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # <-- "ada-002"에서 변경!
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# app/search_engine.py 의 load_vector_db_manually 함수를 이걸로 교체

def load_vector_db_manually(folder_path, index_name):
    faiss_path = os.path.join(folder_path, f"{index_name}.faiss")
    pkl_path = os.path.join(folder_path, f"{index_name}.pkl") # 이제 이게 '좋은 주소록'
    if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError(f"'{folder_path}'에서 DB 파일을 찾을 수 없습니다: {index_name}")
    
    index = faiss.read_index(faiss_path)
    with open(pkl_path, "rb") as f:
        # docs_data는 이제 [{'id':..., 'title':..., 'content':..., 'url':...}, ...] 형태의 리스트
        docs_data = pickle.load(f) 
        
    documents = []
    docstore_dict = {}
    index_to_docstore_id = {}

    # ▼▼▼ [핵심 수정: Document 생성 방식 변경] ▼▼▼
    for i, doc_dict in enumerate(docs_data):
        # DB 생성 시 사용된 'full_text'와 유사하게 page_content를 재구성
        # (DB 생성 코드의 metadata 포맷을 참고하여 필드 추가/수정 필요)
        metadata_str = (
            f"📌 제목: {doc_dict.get('title', '').strip()}\n"
            f"📅 작성일: {doc_dict.get('date', '').strip()}\n"
            f"🏢 기업명: {doc_dict.get('company', 'N/A')}\n"
        )
        content_chunk = doc_dict.get('content', '').strip()
        detail_url = doc_dict.get('url', '') # 'url' 키 사용 (DB 생성 코드 참고)

        # DB 생성 코드의 full_text 포맷과 최대한 유사하게 만듦
        reconstructed_page_content = f"{metadata_str}\n{content_chunk}\n\n🔗 자세한 내용은 링크를 참고하세요: {detail_url}"
        
        # 메타데이터에는 원본 딕셔너리 전체를 넣어도 되고, 필요한 것만 넣어도 됨
        metadata = doc_dict.copy() # 원본 복사해서 사용

        # LangChain Document 객체 생성 (page_content에 재구성된 텍스트 사용)
        doc_obj = Document(page_content=reconstructed_page_content, metadata=metadata)
        documents.append(doc_obj)
        
        # Docstore 및 매핑 생성 (기존 로직)
        doc_id = str(i)
        docstore_dict[doc_id] = doc_obj
        index_to_docstore_id[i] = doc_id
    # ▲▲▲ [수정 완료] ▲▲▲

    docstore = InMemoryDocstore(docstore_dict)

    # LangChainFAISS 객체 생성 (embedding_function 사용)
    return LangChainFAISS(
        embedding_function=embeddings, 
        index=index, 
        docstore=docstore, 
        index_to_docstore_id=index_to_docstore_id
    )

# Vector DB 로딩
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
            context += f"  - 전화번호: {member.get('phone', '정보 없음')}\n---\n"
        return context
    return None

# --- 3. 메인 검색 함수 (라우터 로직 통합) ---

def search_similar_documents(query: str, top_k: int = 3): # top_k=3으로 재설정 (진단용)
    print(f"\n--- [심층 진단] '{query}' 검색 시작 (상위 {top_k}개 요청) ---")
    member_keywords = ["교수", "교수님", "연구실", "이메일", "연락처", "조교", "선생님", "사무실", "위치", "호관", "호실"]
    job_keywords = ["취업", "인턴", "채용", "회사", "직무", "자소서", "면접", "공고"]

    # --- 교수님 질문 처리 (MongoDB) ---
    if any(keyword in query for keyword in member_keywords):
        print(f"[🔍 DB 라우팅] '{query}' -> MongoDB 검색 시도")
        mongo_context = search_members_in_mongodb(query)
        if mongo_context:
            print(f"--- [심층 진단] MongoDB 결과 반환 ---")
            return mongo_context, ['name', 'position', 'lab', 'email', 'phone']
        else:
             print(f"[🔍 DB 라우팅] MongoDB 결과 없음. Vector DB로 계속 진행...")
    
    # --- Vector DB 검색 ---
    selected_dbs = None
    db_type = ""
    if any(keyword in query for keyword in job_keywords):
        print("[진단] 취업 DB 선택")
        selected_dbs = (jobs_db,)
        db_type = "Jobs"
    else:
        print("[진단] 공지사항 DB (제목+본문) 선택")
        selected_dbs = (notices_title_db, notices_content_db)
        db_type = "Notices(Title+Content)"
    
    if not any(db for db in selected_dbs if db is not None):
        return "관련 정보를 찾을 수 없습니다 (DB 로딩 실패).", []

    all_results_with_scores = []
    print(f"--- [심층 진단] {db_type} DB에서 유사도 검색 실행 ---")
    for i, db in enumerate(selected_dbs):
        if db:
            # similarity_search_with_score는 (Document, score) 튜플 리스트 반환
            results = db.similarity_search_with_score(query, k=top_k) 
            print(f"  [DB {i+1}] 검색 완료: {len(results)}개 결과 찾음")
            for doc, score in results:
                print(f"    - 점수: {score:.4f}, 내용: {doc.page_content[:100]}...") # 점수와 내용 일부 출력
            all_results_with_scores.extend(results)

    # --- 중복 제거 및 최종 선택 ---
    unique_results = {}
    for doc, score in all_results_with_scores:
        # 내용이 같으면 점수(거리)가 더 낮은 (더 유사한) 것으로 갱신
        if doc.page_content not in unique_results or score < unique_results[doc.page_content][1]:
            unique_results[doc.page_content] = (doc, score)

    # 점수(score) 기준으로 오름차순 정렬 (점수가 낮을수록 유사함)
    sorted_results = sorted(unique_results.values(), key=lambda item: item[1])
    print(f"--- [심층 진단] 중복 제거 및 정렬 후 {len(sorted_results)}개 결과:")
    for i, (doc, score) in enumerate(sorted_results):
         print(f"  [최종 순위 {i+1}] 점수: {score:.4f}, 내용: {doc.page_content[:100]}...")

    # 최종 Context 생성 (상위 top_k개 사용)
    context = ""
    field_names = set()
    print(f"--- [심층 진단] 상위 {top_k}개를 최종 Context로 사용 ---")
    for doc, score in sorted_results[:top_k]:
        context += f"- 내용 (점수: {score:.4f}): {doc.page_content}\n---\n" # 점수 포함
        field_names.update(doc.metadata.keys())

    if not context:
        print("!!!!!!!!!!!!!! [심층 진단] 최종 컨텍스트가 비어있음 !!!!!!!!!!!!!!")
    else:
        print(f"--- [심층 진단] 최종 컨텍스트 생성 완료 ---")

    return context, list(field_names)



# app/search_engine.py의 get_graduation_info 함수 전체를 이걸로 교체

import re # re 모듈 import 확인 (없으면 추가)

def get_graduation_info(student_id_prefix: str, abeek_bool: bool):
    """
    MongoDB에서 학번(applied_year_range)과 ABEEK 상태(abeek: true/false)에 맞는 
    졸업 요건을 검색합니다. (입력 학번 '20' -> 2020 변환 로직 추가)
    """
    try:
        # 1. 컬렉션 이름 확인
        collection = chatbot_db["graduation_requirements"] 
        
        # --- [핵심 수정: 입력 학번 -> 4자리 연도 변환] ---
        search_year = -1 # 검색에 사용할 4자리 연도
        
        try:
            # 2. 사용자 입력 학번(student_id_prefix, 예: "18", "20")을 숫자로 시도
            year_prefix_num = int(student_id_prefix)
            
            # 3. 2자리 숫자이면 앞에 "20"을 붙여 4자리 연도로 만듦
            if 0 <= year_prefix_num <= 99: # 00 ~ 99 범위의 2자리 수 (또는 1자리)
                search_year = 2000 + year_prefix_num # 예: 20 -> 2020
                print(f"[학번 변환] 입력 '{student_id_prefix}' -> 검색 연도 '{search_year}'")
            else:
                # 4자리 이상 입력 시 그대로 사용 (예외 처리)
                search_year = year_prefix_num
                print(f"[학번 변환] 입력 '{student_id_prefix}'은(는) 4자리 이상이므로 그대로 '{search_year}' 사용")

        except ValueError:
            # 학번이 숫자가 아니면 검색 불가
            print(f"[경고] 입력된 학번 '{student_id_prefix}'을(를) 숫자로 변환할 수 없습니다.")
            return f"입력하신 학번 '{student_id_prefix}'이(가) 올바르지 않습니다."
        # --- [수정 완료] ---

        # MongoDB 쿼리 (abeek 조건만 사용)
        query = { "abeek": abeek_bool }
        all_reqs_for_abeek = list(collection.find(query))
        
        result = None 
        
        # 학번 범위 매칭 루프 (이전과 동일)
        print("--- [get_graduation_info] 학번 범위 매칭 시작 ---")
        for i, req_doc in enumerate(all_reqs_for_abeek):
            range_str = req_doc.get("applied_year_range", "")
            print(f"  [루프 {i+1}] 문서 범위 확인 중: '{range_str}'")

            range_start_year = -1
            range_end_year = float('inf') 

            try:
                year_numbers = re.findall(r'\d+', range_str)

                if len(year_numbers) == 1: 
                    range_start_year = int(year_numbers[0])
                elif len(year_numbers) == 2: 
                    range_start_year = int(year_numbers[0])
                    range_end_year = int(year_numbers[1])

                print(f"    -> 파싱된 범위: Start={range_start_year}, End={range_end_year}, 검색 연도={search_year}")

                # 핵심 비교 로직 (이제 search_year는 4자리)
                is_after_start = (range_start_year <= search_year)
                is_before_end = (search_year <= range_end_year)
                is_match = is_after_start and is_before_end

                print(f"    -> 비교 결과: ({range_start_year} <= {search_year}) = {is_after_start}, ({search_year} <= {range_end_year}) = {is_before_end}, 최종 매칭 = {is_match}")

                if is_match:
                    result = req_doc
                    print(f"    -> ✅ 매칭 성공! 이 문서 사용.")
                    break 
                else:
                    print(f"    -> ❌ 매칭 실패.")

            except Exception as parse_error:
                print(f"    -> ⚠️ 파싱 오류 발생: {parse_error}")
                continue 
        
        # Context 생성 및 반환 (이전과 동일)
        if result:
            requirements = result.get('requirements', {}) 
            credits = requirements.get('credits', {}) 
            courses = requirements.get('required_courses', {}) 
            english = requirements.get('english', {}) 
            grad_qual = requirements.get('graduation_qualification', {}) 
            notes_list = requirements.get('notes', []) 
            notes_str = "\n".join([f"- {note}" for note in notes_list]) if notes_list else "없음"

            context = f"""
            [검색된 맞춤형 졸업 요건 ({student_id_prefix}학번 기준)] 
            - 적용 학번(DB): {result.get('applied_year_range', 'N/A')} 기준
            - ABEEK 이수 여부: {'O' if result.get('abeek') else 'X'}
            [학점 요건]
            - 총 이수 학점: {credits.get('total', 'N/A')}학점
            - 전공 학점: {credits.get('major', 'N/A')}학점 
            - MSC 학점: {credits.get('msc', 'N/A')}학점
            - 기본소양 학점: {credits.get('basic_literacy', 'N/A')}학점
            [필수 과목 요건] 
            - 전공 기초: {courses.get('major_basic', 'N/A')}
            - 전공 필수: {courses.get('major_required', 'N/A')}
            - MSC 필수: {courses.get('msc_required', 'N/A')}
            - 기본소양 필수: {courses.get('basic_literacy_required', 'N/A')}
            [영어 요건] - {str(english)}
            [졸업 자격] - {str(grad_qual)}
            [비고] {notes_str}
            """
            return context
        else:
            fail_msg = f"{student_id_prefix}학번, ABEEK {'O' if abeek_bool else 'X'} 학생에 대한 맞춤형 졸업 요건을 DB에서 찾지 못했습니다. (적용 학번 범위를 확인해주세요)"
            return fail_msg
            
    except Exception as e:
        print(f"MongoDB 졸업 요건 검색 오류: {e}")
        return "졸업 요건 DB를 검색하는 중 오류가 발생했습니다."