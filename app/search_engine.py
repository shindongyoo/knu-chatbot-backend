# app/search_engine.py
import os
import re
import faiss
import pickle
import traceback
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from app.database import chatbot_db
from langchain.tools import tool # <-- [새로 추가] AI 도구 import

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
@tool
def search_similar_documents(query: str, top_k: int = 3) -> str:
    """
    "수강신청", "장학생", "취업 정보", "교수님 정보" 등 
    '졸업 요건'을 제외한 모든 일반적인 교내 정보를 검색할 때 사용합니다.
    (예: "장학생 관련정보 알려줘", "한세경 교수님 이메일 알려줘")
    """
    print(f"\n--- [에이전트 도구 1: 일반 검색] '{query}' 검색 시작 ---")
    member_keywords = ["교수", "교수님", "연구실", "이메일", "연락처", "조교", "선생님"]
    job_keywords = ["취업", "인턴", "채용", "회사", "직무"]

    # (MongoDB 라우팅 로직은 그대로 유지)
    if any(keyword in query for keyword in member_keywords):
        print(f"[🔍 DB 라우팅] '{query}' -> MongoDB 검색 시도")
        mongo_context = search_members_in_mongodb(query)
        if mongo_context:
            return mongo_context
        else:
            print(f"[🔍 DB 라우팅] MongoDB 결과 없음. Vector DB로 계속 진행...")
    
    # (Vector DB 검색 로직은 그대로 유지)
    selected_dbs = None
    if any(keyword in query for keyword in job_keywords):
        selected_dbs = (jobs_db,)
    else:
        selected_dbs = (notices_title_db, notices_content_db)
    
    if not any(db for db in selected_dbs if db is not None):
        return "관련 정보를 찾을 수 없습니다 (DB 로딩 실패)."

    all_results_with_scores = []
    for db in selected_dbs:
        if db:
            results = db.similarity_search_with_score(query, k=top_k)
            all_results_with_scores.extend(results)

    # (중복 제거 및 정렬 로직은 그대로 유지)
    unique_results = {}
    for doc, score in all_results_with_scores:
        if doc.page_content not in unique_results or score < unique_results[doc.page_content][1]:
            unique_results[doc.page_content] = (doc, score)
    sorted_results = sorted(unique_results.values(), key=lambda item: item[1])

    # (Context 생성 로직은 그대로 유지)
    context = ""
    for doc, score in sorted_results[:top_k]:
        context += f"- 내용 (점수: {score:.4f}): {doc.page_content}\n---\n"

    if not context:
        return "검색된 참고 자료가 없습니다."
    else:
        return context


@tool
def get_graduation_info(student_id_prefix: str, abeek_bool: bool) -> str:
    """
    [진단 모드] 졸업 요건 검색 함수
    """
    print(f"\n--- [진단 시작] 학번: {student_id_prefix}, ABEEK: {abeek_bool} ---")
    
    try:
        # 1. 컬렉션 이름 확인 (가장 흔한 원인!)
        COLLECTION_NAME = "graduation_requirements" # <--- 님 DB 컬렉션 이름과 같은지 꼭 확인!
        collection = chatbot_db[COLLECTION_NAME] 
        
        # 2. 학번 변환
        search_year = -1 
        try:
            year_prefix_num = int(student_id_prefix)
            if 0 <= year_prefix_num <= 99: 
                search_year = 2000 + year_prefix_num 
            else:
                search_year = year_prefix_num
            print(f"[1. 학번 변환] 입력 '{student_id_prefix}' -> 검색용 연도 '{search_year}'")
        except ValueError:
            return f"입력하신 학번 '{student_id_prefix}'이(가) 올바르지 않습니다."
        
        # 3. DB 쿼리 실행
        query = { "abeek": abeek_bool }
        print(f"[2. DB 쿼리] 조건: {query}")
        
        all_reqs_for_abeek = list(collection.find(query))
        print(f"[3. 쿼리 결과] 총 {len(all_reqs_for_abeek)}개의 문서를 찾았습니다.")
        
        if len(all_reqs_for_abeek) == 0:
            print("⚠️ [경고] 해당 조건의 문서가 0개입니다. 컬렉션 이름이나 데이터(abeek 필드)를 확인하세요.")
            return f"DB에서 ABEEK 상태가 {abeek_bool}인 문서를 하나도 찾지 못했습니다."

        result = None 
        
        # 4. 매칭 루프
        print("[4. 범위 매칭 시작]")
        for i, req_doc in enumerate(all_reqs_for_abeek):
            range_str = req_doc.get("applied_year_range", "필드없음")
            print(f"  [{i+1}번 문서] 범위: '{range_str}'")
            
            try:
                year_numbers = re.findall(r'\d+', str(range_str))
                if not year_numbers:
                    print("    -> ⚠️ 숫자 추출 실패")
                    continue

                range_start = int(year_numbers[0])
                # 숫자가 1개면(예: 2025~) 끝은 무한대, 2개면(예: 2018~2022) 두번째 숫자
                range_end = int(year_numbers[1]) if len(year_numbers) > 1 else float('inf')
                
                # 비교 로직
                is_match = (range_start <= search_year <= range_end)
                
                print(f"    -> 파싱: {range_start} ~ {range_end}")
                print(f"    -> 비교: {range_start} <= {search_year} <= {range_end} ? 결과: {is_match}")

                if is_match:
                    result = req_doc
                    print("    -> 🎉 정답 문서를 찾았습니다!")
                    break 
            except Exception as e:
                print(f"    -> ❌ 에러 발생: {e}")
                continue
            
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
            print("[5. 결과] 매칭되는 문서를 찾지 못했습니다.")
            return f"{student_id_prefix}학번({search_year}), ABEEK {abeek_bool} 조건에 맞는 범위를 찾을 수 없습니다."
            
    except Exception as e:
        print(f"!!!!!!!!!!!!!! 치명적 오류 !!!!!!!!!!!!!!")
        traceback.print_exc()
        return "오류가 발생했습니다."


@tool
def search_curriculum_subjects(student_id_prefix: str, abeek_bool: bool, grade: int = None, semester: int = None, subject_type: str = None, module: str = None) -> str:
    """
    [설명서] 이 도구는 '교과과정', '개설 과목', '수업 목록'을 검색할 때 사용합니다.
    학번(student_id_prefix)과 ABEEK(abeek_bool) 정보가 필수입니다.
    
    [필터링 옵션]
    - grade (int): 학년 (1, 2, 3, 4)
    - semester (int): 학기 (1, 2)
    - subject_type (str): 과목 구분 (예: "전공기반", "공학전공", "기본소양")
    - module (str): **중요** 사용자가 "스마트 계통", "전력전자" 등 특정 분야/모듈을 언급하면 이 파라미터에 입력하세요.
    """
    print(f"\n--- [에이전트 도구 3: 교과과정 검색] 학번: {student_id_prefix}, ABEEK: {abeek_bool}, 모듈: {module} ---")
    try:
        # ▼▼▼ [수정 1] 컬렉션 이름을 데이터가 있는 곳으로 변경 ▼▼▼
        collection = chatbot_db["graduation_requirements"] 
        # ▲▲▲ [수정 완료] ▲▲▲
        
        # (학번 변환 로직 - 기존과 동일)
        search_year = -1 
        try:
            year_prefix_num = int(student_id_prefix)
            if 0 <= year_prefix_num <= 99: search_year = 2000 + year_prefix_num 
            else: search_year = year_prefix_num
        except ValueError:
            return f"입력 학번 '{student_id_prefix}'이(가) 올바르지 않습니다."
        
        # (학번 범위 검색 로직 - 기존과 동일)
        query_doc = { "abeek": abeek_bool }
        all_reqs = list(collection.find(query_doc))
        result_doc = None 
        for req_doc in all_reqs:
            range_str = req_doc.get("applied_year_range", "") 
            start_year, end_year = -1, float('inf') 
            try:
                year_numbers = re.findall(r'\d+', range_str)
                if len(year_numbers) == 1: start_year = int(year_numbers[0])
                elif len(year_numbers) == 2: start_year, end_year = int(year_numbers[0]), int(year_numbers[1])
                if (start_year <= search_year <= end_year):
                    result_doc = req_doc
                    break 
            except Exception: continue 
        
        if not result_doc:
            return f"{student_id_prefix}학번, ABEEK {'O' if abeek_bool else 'X'}에 대한 '교과과정' 문서를 찾지 못했습니다."
        
        # --- [과목 필터링] ---
        # ▼▼▼ [수정 2] curriculum 객체 안에서 subjects 가져오기 ▼▼▼
        # 데이터 구조: document -> curriculum -> subjects
        curriculum_data = result_doc.get('curriculum', {})
        if not curriculum_data:
             return "해당 학번의 문서에 'curriculum' 데이터가 없습니다."
             
        subjects = curriculum_data.get('subjects', [])
        # ▲▲▲ [수정 완료] ▲▲▲
        
        if not subjects:
            return "교과과정 문서를 찾았으나, 과목(subjects) 리스트가 비어있습니다."

        filtered_subjects = subjects
        
        # 필터링 로직 (학년, 학기, 구분, 모듈)
        if grade:
            filtered_subjects = [s for s in filtered_subjects if s.get('grade') == grade]
        if semester:
            filtered_subjects = [s for s in filtered_subjects if s.get('semester') == semester]
        if subject_type:
            filtered_subjects = [s for s in filtered_subjects if subject_type in s.get('type', '')]
        if module:
            print(f"    -> 필터링: '모듈'에 '{module}' 포함")
            filtered_subjects = [s for s in filtered_subjects if module in s.get('module', '')]
            
        if not filtered_subjects:
            conditions = []
            if grade: conditions.append(f"{grade}학년")
            if semester: conditions.append(f"{semester}학기")
            if subject_type: conditions.append(f"구분:{subject_type}")
            if module: conditions.append(f"모듈:{module}")
            condition_str = ", ".join(conditions)
            return f"조건({condition_str})에 맞는 과목을 찾지 못했습니다."
            
        # 결과 포맷팅
        context = f"[검색된 교과과정 ({student_id_prefix}학번, ABEEK {'O' if abeek_bool else 'X'})]\n"
        context += f"- 적용 학번: {result_doc.get('applied_year_range', 'N/A')}\n"
        context += f"- 검색된 과목 수: {len(filtered_subjects)}\n\n"
        
        for i, sub in enumerate(filtered_subjects[:30]): 
            module_info = f", 모듈: {sub.get('module')}" if sub.get('module') else ""
            context += f"  - {sub.get('course_name')} (학년:{sub.get('grade')}/학기:{sub.get('semester')}, 구분:{sub.get('type')}{module_info}, 학점:{sub.get('credits')})\n"
        
        if len(filtered_subjects) > 30:
            context += f"\n... (외 {len(filtered_subjects) - 30}개 과목이 더 있습니다)"

        return context
            
    except Exception as e:
        print(f"!!!!!!!!!!!!!! 교과과정 검색 중 치명적 오류 발생 !!!!!!!!!!!!!!")
        traceback.print_exc()
        return "교과과정 DB를 검색하는 중 오류가 발생했습니다."