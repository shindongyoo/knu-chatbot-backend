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
from openai import OpenAI

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

def optimize_search_query(query: str) -> str:
    """
    사용자의 애매한 질문을 공지사항/규정 DB 검색에 적합한 '핵심 키워드 문장'으로 변환합니다.
    단, 기업명이나 교수명 같은 고유명사는 절대 변경하지 않습니다.
    예: "학교 좀 쉬고 싶어" -> "휴학 신청 절차 및 기간"
    예: "돈 주는거 뭐 있어?" -> "장학금 종류 및 신청 안내"
    """
    try:
        client = OpenAI() # 환경변수 API KEY 사용
        
        system_prompt = """당신은 검색어 최적화 도구입니다. 
        사용자의 질문을 대학교 공지사항이나 규정집에서 검색하기 좋은 '공식 용어'로 변환하세요.

        [절대 규칙]
        1. **기업명(삼성, 현대, LG 등)이나 교수님 성함(한세경 등) 같은 고유명사는 절대로 변경하거나 삭제하지 마세요.** 그대로 포함시켜야 합니다.
        2. "삼성 채용" -> "삼성 채용 공고 모집 요강" (O)
        3. "삼성 채용" -> "대학교 취업 안내" (X - 기업명이 사라짐!)
        4. "쉬고 싶어" -> "휴학 신청 절차" (O - 애매한 표현은 변환)
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"질문: {query}"}
            ],
            temperature=0
        )
        optimized_query = response.choices[0].message.content.strip()
        print(f"    [검색어 변환] '{query}' -> '{optimized_query}'")
        return optimized_query
    except Exception as e:
        print(f"    [변환 실패] 원본 사용: {e}")
        return query

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
            context += f"  - 전공분야: {member.get('major', '정보 없음')}\n"
            context += f"  - 이메일: {member.get('email', '정보 없음')}\n"
            context += f"  - 전화번호: {member.get('phone', '정보 없음')}\n---\n"
        return context
    return None

# --- 3. 메인 검색 함수 (라우터 로직 통합) ---
@tool
def search_similar_documents(query: str, top_k: int = 3) -> str:
    """
    "수강신청", "장학생", "교수님 정보", "학사 일정" 등 
    '졸업 요건'이나 '교과과정'을 제외한 모든 일반적인 교내 정보를 검색할 때 사용합니다.
    질문이 애매해도 찰떡같이 알아듣고 검색합니다.
    """
    print(f"\n--- [에이전트 도구 1: 일반 검색] 원본 질문: '{query}' ---")
    
    # 1. 질문을 DB 친화적으로 변환 (여기가 핵심!)
    optimized_query = optimize_search_query(query)
    
    # 2. 교수님 키워드 확인 (기존 로직 유지)
    member_keywords = ["교수", "교수님", "연구실", "이메일", "연락처", "조교", "선생님"]
    job_keywords = ["취업", "인턴", "채용", "회사", "직무"]

    # (MongoDB 라우팅 로직)
    if any(keyword in query for keyword in member_keywords):
        print(f"    [라우팅] 교수님 검색 모드")
        mongo_context = search_members_in_mongodb(query) # 원본 이름 사용 (이름은 변환하면 안됨)
        if mongo_context:
            return mongo_context
        else:
            print(f"    [라우팅] MongoDB 결과 없음. Vector DB로 계속 진행...")
    
    # 3. Vector DB 검색 (변환된 optimized_query 사용!)
    selected_dbs = None
    if any(keyword in query for keyword in job_keywords):
        selected_dbs = (jobs_db,)
    else:
        selected_dbs = (notices_title_db, notices_content_db)
    
    if not any(db for db in selected_dbs if db is not None):
        return "관련 정보를 찾을 수 없습니다 (DB 로딩 실패)."

    all_results_with_scores = []
    
    print(f"    [Vector DB 검색] 키워드: '{optimized_query}'")
    for db in selected_dbs:
        if db:
            # 여기서 변환된 쿼리로 검색합니다!
            results = db.similarity_search_with_score(optimized_query, k=top_k)
            all_results_with_scores.extend(results)

    # (중복 제거 및 정렬 로직 - 기존과 동일)
    unique_results = {}
    for doc, score in all_results_with_scores:
        if doc.page_content not in unique_results or score < unique_results[doc.page_content][1]:
            unique_results[doc.page_content] = (doc, score)
    sorted_results = sorted(unique_results.values(), key=lambda item: item[1])

    # (Context 생성 - 기존과 동일)
    context = ""
    for doc, score in sorted_results[:top_k]:
        # 너무 관련 없는 것(점수 1.6 이상)은 필터링 (선택 사항)
        if score < 1.6: 
            context += f"- 내용 (점수: {score:.4f}): {doc.page_content}\n---\n"

    if not context:
        return f"'{query}'(변환: {optimized_query})에 대한 관련 정보를 찾지 못했습니다."
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
        COLLECTION_NAME = "graduation_requirements2" # <--- 님 DB 컬렉션 이름과 같은지 꼭 확인!
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
def search_curriculum_subjects(student_id_prefix: str = None, abeek_bool: bool = None, grade: int = None, semester: int = None, subject_type: str = None, module: str = None) -> str:
    """
    [설명서] '교과과정', '개설 과목', '수업 목록'을 검색합니다.
    [중요] 사용자가 '모듈'(예: 스마트계통, 전력전자)을 물어보면 다른 인자 없이 module만 입력해도 됩니다.
    """
    print(f"\n--- [도구 실행] 교과과정 검색 (모듈: {module}) ---")
    
    try:
        collection = chatbot_db["graduation_requirements2"] 
        
        target_docs = []

        # 1. 문서 확보 전략
        if module:
            print("    -> 모듈 검색 모드: 전체 문서 스캔")
            # 모듈 검색은 학번/ABEEK 무시하고 전체 문서 스캔
            target_docs = list(collection.find({}))
        
        elif student_id_prefix:
            # 학번 검색 모드
            search_year = 2000 + int(student_id_prefix) if int(student_id_prefix) < 100 else int(student_id_prefix)
            query = {"abeek": abeek_bool} if abeek_bool is not None else {}
            all_docs = list(collection.find(query))
            
            for doc in all_docs:
                range_str = doc.get("applied_year_range", "")
                try:
                    nums = re.findall(r'\d+', range_str)
                    if not nums: continue
                    start = int(nums[0])
                    end = int(nums[1]) if len(nums) > 1 else float('inf')
                    if start <= search_year <= end:
                        target_docs.append(doc)
                        break 
                except: continue

        if not target_docs:
            # 안전장치: 혹시 모르니 그냥 다 가져와봄 (데이터가 적으므로 가능)
            target_docs = list(collection.find({}))

        # 2. 과목 필터링
        results = []
        search_module = module.replace(" ", "") if module else ""

        print(f"--- [디버깅] 문서 {len(target_docs)}개 내부 탐색 시작 ---")

        for doc in target_docs:
            # ▼▼▼ [핵심 수정] 데이터 위치 자동 탐색 ▼▼▼
            # 1순위: 최상위에 curriculum이 있는 경우
            curriculum = doc.get('curriculum', {})
            subjects = curriculum.get('subjects', [])
            
            # 2순위: requirements 안에 curriculum이 있는 경우 (지금 님의 DB 상황!)
            if not subjects:
                requirements = doc.get('requirements', {})
                curriculum = requirements.get('curriculum', {})
                subjects = curriculum.get('subjects', [])
            # ▲▲▲ [수정 완료] ▲▲▲
            
            for sub in subjects:
                # 모듈 체크
                if module:
                    sub_module = sub.get('module', '').replace(" ", "")
                    if search_module not in sub_module:
                        continue 

                # 나머지 필터
                if grade and sub.get('grade') != grade: continue
                if semester and sub.get('semester') != semester: continue
                
                # 결과 포맷팅
                mod_info = f" [모듈: {sub.get('module')}]" if sub.get('module') else ""
                info = f"- {sub.get('course_name')} (학년: {sub.get('grade')}, 구분: {sub.get('type')}){mod_info}"
                
                if info not in results:
                    results.append(info)

        if not results:
            return f"조건(모듈: {module})에 맞는 과목을 찾을 수 없습니다. DB 구조를 확인해주세요."

        return f"[검색 결과] 총 {len(results)}개 과목 발견:\n" + "\n".join(results[:30])

    except Exception as e:
        traceback.print_exc()
        return "검색 중 오류 발생"
    
@tool
def search_professors_by_keyword(keyword: str) -> str:
    """
    "스마트 계통", "전력전자", "반도체", "인공지능" 등 특정 '분야'나 '모듈' 키워드로 
    관련된 교수님 정보를 찾을 때 사용합니다. (이름 검색 아님)
    """
    print(f"\n--- [에이전트 도구 4: 교수님 분야 검색] 키워드: {keyword} ---")
    try:
        collection = chatbot_db["members"] 
        
        # 정규식으로 '전공(major)' 또는 '연구실(lab)'에 키워드가 포함된 교수 검색
        query = {
            "$or": [
                {"name": {"$regex": keyword, "$options": "i"}},     # 이름으로 찾기 (필수!)
                {"major": {"$regex": keyword, "$options": "i"}},    # 전공으로 찾기
                {"lab": {"$regex": keyword, "$options": "i"}},      # 연구실로 찾기
                {"position": {"$regex": keyword, "$options": "i"}}, # 직위로 찾기
                {"email": {"$regex": keyword, "$options": "i"}}     # 이메일로 찾기
            ]
        }
        
        results = list(collection.find(query))
        
        if not results:
            return f"'{keyword}' 분야와 관련된 교수님 정보를 찾지 못했습니다."
        
        context = f"[검색된 '{keyword}' 관련 교수님 목록 (총 {len(results)}명)]\n"
            
        # 결과 포맷팅
        for member in results:
            context += f"- 이름: {member.get('name', '정보없음')}\n"
            context += f"  - 직위: {member.get('position', '정보없음')}\n"
            context += f"  - 연구실: {member.get('lab', '정보없음')}\n"
            context += f"  - 전공분야: {member.get('major', '정보없음')}\n"
            context += f"  - 이메일: {member.get('email', '정보없음')}\n"
            context += f"  - 전화번호: {member.get('phone', '정보없음')}\n"
            context += "---\n"
            
        return context

    except Exception as e:
        print(f"교수님 분야 검색 오류: {e}")
        traceback.print_exc()
        return "교수님 검색 중 오류가 발생했습니다."

@tool
def get_employment_stats(year: int = 2023) -> str:
    """
    [설명서] '취업률', '취업 통계', '진로 현황', '어떤 회사 갔어?', '대기업 취업 비율' 등
    학과 졸업생들의 취업 실적과 통계 데이터를 검색할 때 사용합니다.
    기본적으로 2023년 데이터를 검색합니다.
    """
    print(f"\n--- [에이전트 도구 4: 취업 통계 검색] 연도: {year} ---")
    
    try:
        # 1. 컬렉션 이름 확인 (MongoDB에 이 컬렉션이 있어야 합니다!)
        collection = chatbot_db["employment_rate_2023"] 
        
        # 2. 연도(year)로 문서 검색
        query = {"year": year}
        result = collection.find_one(query)
        
        if not result:
            # 특정 연도가 없으면 가장 최신 데이터를 가져오도록 유도하거나 전체 목록 확인
            return f"{year}년도 취업 통계 데이터를 찾을 수 없습니다."
            
        # 3. 데이터 파싱 및 Context 생성
        stats = result.get('stats', {})
        
        # 3-1. 전체 현황
        overall = stats.get('1_overall_status', {})
        overall_text = (
            f"- 졸업자: {overall.get('graduates')}명, 취업자: {overall.get('employed')}명\n"
            f"- 진학: {overall.get('advanced_study')}명, 미취업: {overall.get('unemployed')}명\n"
            f"- 📈 취업률: {overall.get('employment_rate')} (진학률: {overall.get('advancement_rate')})"
        )
        
        # 3-2. 기업 형태별 요약
        company_summary = stats.get('3_company_type_summary', {})
        dist_list = company_summary.get('distribution', [])
        dist_text = ", ".join([f"{d['type']}: {d['ratio']}({d['count']}명)" for d in dist_list])
        
        # 3-3. 상세 취업처 (리스트 포맷팅 헬퍼 함수)
        def format_companies(company_list):
            if not company_list: return "없음"
            # 예: "현대자동차(5), LG전자(3)"
            return ", ".join([f"{c['name']}({c['count']}명)" for c in company_list])

        details = stats.get('4_employment_details', {})
        large_ent = format_companies(details.get('large_enterprise', []))
        medium_ent = format_companies(details.get('medium_enterprise', []))
        small_ent = format_companies(details.get('small_medium_enterprise', []))
        public_inst = format_companies(details.get('public_institution', []))
        
        # 4. 최종 Context 조합
        context = f"""
        [검색된 {year}년도 전기공학과 취업 통계]
        
        1. 전체 현황
        {overall_text}
        
        2. 기업 형태별 분포
        - {dist_text}
        
        3. 주요 취업처 상세 (기업명 및 인원)
        - 🏢 대기업: {large_ent}
        - 🏭 중견기업: {medium_ent}
        - 🏘️ 중소기업: {small_ent}
        - 🏛️ 공공기관/공기업: {public_inst}
        """
        
        return context

    except Exception as e:
        print(f"취업 통계 검색 오류: {e}")
        traceback.print_exc()
        return "취업 통계 정보를 가져오는 중 오류가 발생했습니다."