# app/search_engine.py
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


import re 
import traceback # <-- 상세 오류 추적을 위해 import 추가

def get_graduation_info(student_id_prefix: str, abeek_bool: bool):
    """
    MongoDB에서 학번(applied_year_range)과 ABEEK 상태(abeek: true/false)에 맞는 
    졸업 요건을 검색합니다. (루프 내 오류 로깅 추가)
    """
    print("--- [get_graduation_info] 함수 시작 ---")
    try:
        collection = chatbot_db["graduation_requirements2"] 
        print(f"--- [get_graduation_info] 컬렉션 '{collection.name}' 선택 완료 ---")
        
        # --- [1. 학번 변환 로직] ---
        search_year = -1 
        try:
            year_prefix_num = int(student_id_prefix)
            if 0 <= year_prefix_num <= 99: 
                search_year = 2000 + year_prefix_num 
            else:
                search_year = year_prefix_num
            print(f"[학번 변환] 입력 '{student_id_prefix}' -> 검색 연도 '{search_year}'")
        except ValueError:
            return f"입력하신 학번 '{student_id_prefix}'이(가) 올바르지 않습니다."
        
        # --- [2. MongoDB 쿼리] ---
        query = { "abeek": abeek_bool }
        print(f"--- [MongoDB] 쿼리 실행 직전: {query} ---")
        all_reqs_for_abeek = list(collection.find(query))
        print(f"--- [MongoDB] 쿼리 실행 완료: {len(all_reqs_for_abeek)}개 찾음 ---")
        
        result = None 
        
        # --- [3. 학번 범위 매칭 루프] ---
        print("--- [get_graduation_info] 학번 범위 매칭 시작 ---")
        for i, req_doc in enumerate(all_reqs_for_abeek):
            range_str = req_doc.get("applied_year_range", "") 
            print(f"  [루프 {i+1}] 문서 범위 확인 중: '{range_str}'")
            
            range_start_year = -1
            range_end_year = float('inf') 
            
            # ▼▼▼ [핵심 수정] ▼▼▼
            try:
                year_numbers = re.findall(r'\d+', range_str)
                if len(year_numbers) == 1: 
                    range_start_year = int(year_numbers[0])
                elif len(year_numbers) == 2: 
                    range_start_year = int(year_numbers[0])
                    range_end_year = int(year_numbers[1])
                
                is_after_start = (range_start_year <= search_year)
                is_before_end = (search_year <= range_end_year)
                is_match = is_after_start and is_before_end
                
                print(f"    -> 파싱: {range_start_year}~{range_end_year} / 비교: ({is_after_start} AND {is_before_end}) = {is_match}") # 로그 추가

                if is_match:
                    result = req_doc
                    print(f"    -> ✅ 매칭 성공! 이 문서 사용.")
                    break 
                else:
                    print(f"    -> ❌ 매칭 실패.")

            except Exception as e: 
                print(f"!!!!!!!!!!!!!! [루프 {i+1}] 범위 매칭 중 오류 발생 !!!!!!!!!!!!!!")
                print(f"    -> 오류 내용: {e}")
                traceback.print_exc() # 상세 오류 스택 출력
                continue # 다음 루프로 넘어감
            
        # --- [3. Context 생성 (최종 상세 스키마 반영)] ---
        if result:
            # ▼▼▼ [핵심 수정: 상세 스키마 반영] ▼▼▼
            
            # 안전하게 데이터 추출 (객체가 없으면 빈 dict 반환)
            requirements = result.get('requirements', {}) 
            credits = requirements.get('credits', {})
            credit_basic = credits.get('기본소양', {}) 
            credit_msc = credits.get('전공기반', "N/A") # 전공기반은 객체가 아닌 직접 값
            credit_major = credits.get('공학전공', {})
            
            courses = requirements.get('required_courses', {}) 
            courses_basic = courses.get('기본소양', [])
            courses_msc = courses.get('전공기반', [])
            courses_major = courses.get('공학전공', [])

            english = requirements.get('english', {})
            eng_tests = english.get('tests', [])
            eng_sub = english.get('substitution', [])
            eng_notes = english.get('notes', [])

            grad_qual = requirements.get('graduation_qualification', {})
            advisor = grad_qual.get('advisor_consultation', {})
            software = grad_qual.get('software_credits', {})
            sw_courses = software.get('required_courses', [])
            sw_sub = software.get('substitution', [])
            grad_qual_notes = grad_qual.get('notes', [])

            # 리스트(Array) 정보를 콤마(,)로 구분된 문자열로 변환
            def format_list(items):
                return ", ".join(items) if items else "해당 없음"
            
            # 영어 시험 요건 포맷팅
            def format_eng_tests(tests):
                if not tests: return "해당 없음"
                return ", ".join([f"{test.get('name')}: {test.get('score')}점 이상" for test in tests])

            context = f"""
            [검색된 맞춤형 졸업 요건 ({student_id_prefix}학번, ABEEK {'O' if abeek_bool else 'X'})] 
            - 적용 학번(DB): {result.get('applied_year_range', 'N/A')} 기준

            [1. 학점 요건 (Credits)]
            - 총 이수 학점: {credits.get('total', 'N/A')}학점
            - 기본소양(교양): {credit_basic.get('min', 'N/A')}학점 이상
            - 전공기반(MSC): {credit_msc}학점
            - 공학전공(전공): {credit_major.get('total', 'N/A')}학점 (이 중 설계 {credit_major.get('design', 'N/A')}학점 포함)
            - 전공 참고: {credit_major.get('note', 'N/A')}

            [2. 필수 과목 요건 (Required Courses)] 
            - 기본소양 필수: {format_list(courses_basic)}
            - 전공기반 필수: {format_list(courses_msc)}
            - 공학전공 필수: {format_list(courses_major)}

            [3. 영어 요건 (English)]
            - 공인 시험 기준: {format_eng_tests(eng_tests)}
            - 면제 기준: {format_list(eng_sub)}
            - 비고: {format_list(eng_notes)}

            [4. 졸업 자격 (Graduation Qualification)]
            - 지도교수 상담: {advisor.get('count', 'N/A')}회 이상 ({advisor.get('note', 'N/A')})
            - 소프트웨어 이수: {software.get('min', 'N/A')}학점 이상 ({software.get('note', 'N/A')})
            - (SW 인정 과목: {format_list(sw_courses)})
            - (SW 면제 기준: {format_list(sw_sub)})
            - 졸업 자격 비고: {format_list(grad_qual_notes)}

            [종합 비고]
            {format_list(requirements.get('notes', []))}
            """
            # ▲▲▲ [수정 완료] ▲▲▲
            return context
        else:
            fail_msg = f"{student_id_prefix}학번, ABEEK {'O' if abeek_bool else 'X'} 학생에 대한 맞춤형 졸업 요건을 DB에서 찾지 못했습니다."
            return fail_msg
            
    except Exception as e:
        print(f"!!!!!!!!!!!!!! MongoDB 졸업 요건 검색 중 치명적 오류 발생 !!!!!!!!!!!!!!")
        import traceback
        traceback.print_exc()
        return "졸업 요건 DB를 검색하는 중 오류가 발생했습니다."