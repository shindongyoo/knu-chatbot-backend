# app/main.py
import os
import json
import time
import openai
import pytesseract
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from PIL import Image
from dotenv import load_dotenv
load_dotenv() # .env 파일을 여기서 먼저 로드합니다.
# --- 서비스 초기화 ---
# OpenAI 클라이언트 초기화 (최신 v1.x 방식)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ▼▼▼ [수정] database.py에서 DB 객체들을 import 합니다. ▼▼▼
from app.database import chatbot_db, r
# ▼▼▼ [수정] DB 초기화가 끝난 후에 search_engine을 import 합니다. ▼▼▼
from app.search_engine import search_similar_documents
from app.search_engine import get_graduation_info

# FastAPI 앱 설정
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    user_id: str
    session_id: str
    question: str

def get_recent_history(session_id: str, n=5) -> list[dict]: # n=3 -> n=5 (조절 가능)
    """
    Redis에서 최근 N개의 대화 기록을 'messages' API 형식으로 불러옵니다.
    (예: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}])
    """
    if not r: return [] # 빈 리스트 반환
    try:
        key = f"chat:{session_id}"
        logs = r.lrange(key, -n * 2, -1) # 최근 N*2 (Q, A) 항목
        
        messages = []
        for item in logs:
            parsed = json.loads(item)
            q = parsed.get("question")
            a = parsed.get("answer")
            
            # 저장된 순서(Q, A)를 보장하기 위해 확인 후 추가
            if q: messages.append({"role": "user", "content": q})
            if a: messages.append({"role": "assistant", "content": a})
        
        return messages # list[dict] 반환
    
    except Exception as e:
        print(f"Redis 히스토리 조회 오류: {e}")
        return [] # 오류 시 빈 리스트 반환

def save_chat_history(user_id: str, session_id: str, question: str, answer: str):
    if not r: return
    try:
        # 1. 세션 목록을 저장하는 키를 사용자별로 분리합니다.
        #    'sessions_sorted' -> 'user:{user_id}:sessions_sorted'
        r.zadd(f"user:{user_id}:sessions_sorted", {session_id: time.time()})

        # 2. 개별 대화 내용은 session_id로 이미 고유하므로 그대로 둡니다.
        key = f"chat:{session_id}"
        log_entry = json.dumps({
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        })
        r.rpush(key, log_entry)
        r.expire(key, 3 * 24 * 60 * 60)
    except Exception as e:
        print(f"Redis 히스토리 저장 오류: {e}")

@app.get("/")
def root():
    return {"message": "KNU Chatbot backend is running"}

def transform_query(question: str) -> list[str]:
    """AI를 사용해 사용자의 애매한 질문을 구체적인 검색어로 변환합니다."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 사용자의 질문 1개를 Vector DB에서 검색하기 좋은 구체적인 검색어 3개로 바꾸는 쿼리 변환 전문가입니다. 각 검색어는 쉼표(,)로 구분해서 응답하세요."},
                {"role": "user", "content": f"다음 질문을 검색어로 바꿔줘: {question}"}
            ],
            temperature=0
        )
        transformed_queries = response.choices[0].message.content.strip()
        queries = [q.strip() for q in transformed_queries.split(',')][:3] # 최대 3개
        print(f"[쿼리 변환] 원본: '{question}' -> 변환: {queries}")
        return queries
    except Exception as e:
        print(f"쿼리 변환 실패: {e}")
        return [question] # 실패하면 원래 질문으로 검색


@app.post("/stream")
def stream_answer(req: QuestionRequest):
    question = req.question
    session_id = req.session_id
    user_id = req.user_id

    

    def event_generator():
        try:
            state_key = f"state:{session_id}"
            current_state = r.get(state_key)

            # --- [유지] 1. '졸업요건' 2단계 처리 ---
            if current_state == "awaiting_grad_info":
                print(f"[상태 감지] 'awaiting_grad_info' 상태입니다. 입력: {question}")
                try:
                    student_id, abeek_status_input = question.strip().split('/')
                    abeek_bool_value = True if abeek_status_input.upper() == 'O' else False
                    print(f"[정보 파싱] 학번: {student_id}, ABEEK(쿼리용): {abeek_bool_value}")
                
                except Exception as e:
                    error_text = '입력 형식이 잘못되었습니다. 학번(예: 18)과 ABEEK 이수 여부(O/X)를 \'18/O\' 형식으로 다시 입력해주세요.'
                    yield f"data: {json.dumps({'text': error_text})}\n\n"
                    return

                context = get_graduation_info(student_id, abeek_bool_value)
                r.delete(state_key) # 상태 초기화

                system_prompt = "당신은 경북대학교 졸업 요건 안내 전문가입니다. 오직 '검색된 졸업 요건' 자료에만 근거하여 사용자에게 맞춤형 졸업 정보를 안내하세요."
                user_prompt = f"""
                ### 검색된 졸업 요건:
                {context if context else "일치하는 졸업 요건을 찾지 못했습니다."}
                ### 사용자의 질문 (요약):
                {student_id}학번, ABEEK {abeek_status_input} 학생의 졸업 요건
                ### 답변:
                """
                
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True, temperature=0.2
                )
                
                collected_answer = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        collected_answer += delta
                        yield f"data: {json.dumps({'text': delta})}\n\n"
                
                save_chat_history(user_id, session_id, "졸업 요건 질문", collected_answer)
                return # 작업 완료

            # --- [유지] 2. '졸업요건' 1단계 의도 감지 ---
            if "졸업요건" in question or "졸업 요건" in question:
                print(f"[의도 감지] '졸업요건' 질문을 감지했습니다.")
                r.set(state_key, "awaiting_grad_info", ex=300) 
                follow_up_question = "졸업 요건을 확인하기 위해, 학번(예: 18, 19, 20...)과 ABEEK 이수 여부(O/X)를 **'18/O'** 형식으로 입력해주세요."
                yield f"data: {json.dumps({'text': follow_up_question})}\n\n"
                return # 작업 완료

            # --- [핵심 수정] 3. '일반 질문' 처리 ---
            else:
                print(f"[일반 질문] '{question}'에 대한 RAG 절차를 시작합니다.")
                
                # 1. [통합] API가 이해하는 'list[dict]' 형태로 대화 기록을 "한 번만" 불러옵니다.
                history_messages = get_recent_history(session_id, n=5)
                
                # 1. search_engine에서 교수님 키워드 리스트를 가져옵니다. (가져올 수 없다면 직접 정의)
                member_keywords = ["교수", "교수님", "연구실", "이메일", "연락처", "조교", "선생님"]
                context = None
                
                if any(keyword in question for keyword in member_keywords):
                    # 1-A. 교수님 질문이면, search_similar_documents를 "딱 한 번만" 호출합니다.
                    print("[라우팅] 교수님 질문으로 판단. MongoDB 검색 시도.")
                    context, _ = search_similar_documents(question)
                
                else:
                    # 1-B. 교수님 질문이 "아닌" 경우에만 "질문 해석 AI"를 사용합니다.
                    print("[라우팅] 일반 질문으로 판단. AI 쿼리 변환 시작.")
                    search_queries = transform_query(question)
                    all_contexts = []
                    for q in search_queries:
                        context_part, _ = search_similar_documents(q)
                        if context_part:
                            all_contexts.append(context_part)
                    context = "\n---\n".join(all_contexts)
                    
              
                MAX_CONTEXT_LENGTH = 7000
                if len(context) > MAX_CONTEXT_LENGTH:
                    context = context[:MAX_CONTEXT_LENGTH]

                # ▼▼▼ [수정된 system_prompt] ▼▼▼
                system_prompt = """당신은 경북대학교 전기공학과 학생들을 돕는 '자율 AI 비서'입니다. 당신은 사용자의 '이전 대화 기록'을 기억하고 맥락을 파악할 수 있습니다.

                [당신의 임무]
                당신은 '검색된 참고 자료'와 '이전 대화 기록', '당신의 내부 지식'을 모두 활용하여 사용자에게 가장 도움이 되는 답변을 해야 합니다.

                [행동 지침]
                1.  **[1단계: 질문 의도 파악]** 먼저 '이전 대화 기록'을 확인하여, 사용자의 새 질문이 "그럼", "거기서", "그분은" 등 이전 대화에 이어지는 **'후속 질문'**인지, 아니면 **'새로운 질문'**인지 판단합니다.
                
                2.  **[2단계: 답변 생성]**
                    * **(A) '후속 질문'일 경우:** (예: "그럼 이메일은?")
                        - **'이전 대화 기록'을 최우선**으로 참고하여 답변합니다.
                        - 이 경우 '검색된 참고 자료'는 비어있거나 관련 없을 수 있습니다. **자료를 무시하고 기억을 바탕으로 답하세요.**
                        - 예: (이전에 '한세경 교수'를 답했다면) "네, 한세경 교수님의 이메일 주소는 XXX@knu.ac.kr입니다."
                    
                    * **(B) '새로운 질문'일 경우:** (예: "장학생 정보 알려줘")
                        - **'검색된 참고 자료'를 최우선**으로 평가합니다.
                        - (B-1) 자료가 유용하면: 자료에 근거하여 답변합니다.
                        - (B-2) 자료가 쓸모없으면 (관련 없거나 비어있음): 자료를 무시하고, [규칙 3]에 따라 행동합니다.

                3.  **[3단계: 잡담 또는 정보 없음]**
                    * 사용자의 질문이 '안녕?' 같은 **일상 대화**이거나, (B-2)처럼 쓸모없는 자료를 받은 '새로운 질문'일 경우, **당신의 내부 지식**을 활용하여 자유롭고 친절하게 대화하세요. (단, 없는 교내 정보를 지어내지는 마세요.)
                """
                
                # ▲▲▲ [수정 완료] ▲▲▲
                
                # ▼▼▼ [수정 3] 'user_prompt'에서는 '이전 대화 기록' 섹션을 제거합니다.
                #    맥락(Context)과 현재 질문(Question)만 남깁니다.
                user_prompt = f"""
                ### 검색된 참고 자료 (참고만 하세요):
                {context}

                ### 사용자의 질문:
                {question}

                ### 답변:
                """
                
                # API에 전달할 'messages' 리스트 재구성
                messages_to_send = []
                messages_to_send.append({"role": "system", "content": system_prompt})
                # [중요] 이전 대화 기록(list[dict])을 먼저 추가합니다.
                messages_to_send.extend(history_messages) 
                # [중요] RAG 컨텍스트가 포함된 'user_prompt'를 마지막에 추가합니다.
                messages_to_send.append({"role": "user", "content": user_prompt})

                # [디버깅용] API에 전달되는 최종 메시지 목록 확인
                print("--- [API Request] API로 다음 메시지들을 전송합니다: ---")
                for msg in messages_to_send:
                    content_preview = (msg['content'][:150] + '...') if len(msg['content']) > 150 else msg['content']
                    print(f"  {msg['role']}: {content_preview.replace('  ', ' ')}")
                print("--------------------------------------------------")

                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages_to_send, # <-- (변경) 재구성된 리스트를 전달
                    stream=True,
                    temperature=0.7 
                )
                
                collected_answer = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        collected_answer += delta
                        yield f"data: {json.dumps({'text': delta})}\n\n"
                
                save_chat_history(user_id, session_id, question, collected_answer)

        except Exception as e:
            print(f"!!!!!!!!!!!!!! 스트림 중 심각한 오류 발생 !!!!!!!!!!!!!!")
            import traceback
            traceback.print_exc()
            error_message = json.dumps({"error": "답변 생성 중 오류가 발생했습니다."})
            yield f"data: {error_message}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ask")
async def ask(req: QuestionRequest):
    try:
        # ▼▼▼ [수정 1] API가 이해하는 'list[dict]' 형태로 대화 기록을 불러옵니다.
        history_messages = get_recent_history(req.session_id, n=5) # n=5는 조절 가능

        context, _ = search_similar_documents(req.question)
        
        system_prompt = """당신은 경북대학교 전기공학과 학생들을 돕는 '자율 AI 비서'입니다. 당신은 사용자의 '이전 대화 기록'을 기억하고 맥락을 파악할 수 있습니다.

        [당신의 임무]
        당신은 '검색된 참고 자료'와 '이전 대화 기록', '당신의 내부 지식'을 모두 활용하여 사용자에게 가장 도움이 되는 답변을 해야 합니다.

        [행동 지침]
        1.  **[1단계: 질문 의도 파악]** 먼저 '이전 대화 기록'을 확인하여, 사용자의 새 질문이 "그럼", "거기서", "그분은" 등 이전 대화에 이어지는 **'후속 질문'**인지, 아니면 **'새로운 질문'**인지 판단합니다.
        
        2.  **[2단계: 답변 생성]**
            * **(A) '후속 질문'일 경우:** (예: "그럼 이메일은?")
                - **'이전 대화 기록'을 최우선**으로 참고하여 답변합니다.
                - 이 경우 '검색된 참고 자료'는 비어있거나 관련 없을 수 있습니다. **자료를 무시하고 기억을 바탕으로 답하세요.**
                - 예: (이전에 '한세경 교수'를 답했다면) "네, 한세경 교수님의 이메일 주소는 XXX@knu.ac.kr입니다."
            
            * **(B) '새로운 질문'일 경우:** (예: "장학생 정보 알려줘")
                - **'검색된 참고 자료'를 최우선**으로 평가합니다.
                - (B-1) 자료가 유용하면: 자료에 근거하여 답변합니다.
                - (B-2) 자료가 쓸모없으면 (관련 없거나 비어있음): 자료를 무시하고, [규칙 3]에 따라 행동합니다.

        3.  **[3단계: 잡담 또는 정보 없음]**
            * 사용자의 질문이 '안녕?' 같은 **일상 대화**이거나, (B-2)처럼 쓸모없는 자료를 받은 '새로운 질문'일 경우, **당신의 내부 지식**을 활용하여 자유롭고 친절하게 대화하세요. (단, 없는 교내 정보를 지어내지는 마세요.)
        """
        
        # 2. 'user' 프롬프트에는 질문과 참고 자료(데이터)만 전달합니다.
        user_prompt = f"""
        ### 검색된 참고 자료:
        {context}

        ### 사용자의 질문:
        {req.question}

        ### 답변:
        """

        messages_to_send = []
        messages_to_send.append({"role": "system", "content": system_prompt})
        messages_to_send.extend(history_messages) # 이전 기록 추가
        messages_to_send.append({"role": "user", "content": user_prompt}) # RAG + 새 질문 추가

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_to_send, # <-- (변경)
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        
        save_chat_history(req.user_id, req.session_id, req.question, answer)

        return JSONResponse(content={"answer": answer})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": f"답변 생성 중 오류: {e}"}, status_code=500)
    
# 2025-09-21

# URL 경로에 user_id를 받도록 변경: @app.get("/sessions") -> @app.get("/sessions/{user_id}")
@app.get("/sessions/{user_id}")
async def get_sessions(user_id: str):
    if not r:
        return JSONResponse(content={"error": "Redis not connected"}, status_code=500)
    try:
        # 해당 사용자의 세션 목록만 가져오도록 키 변경
        session_ids = r.zrevrange(f"user:{user_id}:sessions_sorted", 0, 4) # 최근 5개
        
        sessions_with_titles = []
        for session_id in session_ids:
            first_log_raw = r.lrange(f"chat:{session_id}", 0, 0)
            if first_log_raw:
                first_question = json.loads(first_log_raw[0]).get("question", "알 수 없는 대화")
                sessions_with_titles.append({
                    "session_id": session_id,
                    "title": first_question[:50]
                })
        return JSONResponse(content={"sessions": sessions_with_titles})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ▼▼▼ 이 함수 전체를 복사해서 붙여넣으세요 ▼▼▼
@app.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    user_id: str,
    cursor: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100)
):
    """
    특정 세션의 대화 기록을 페이지네이션으로 불러옵니다.
    소유권 확인을 위해 user_id가 필요합니다.
    """
    if not r:
        return JSONResponse(
            status_code=503,
            content={"error": "Redis service is unavailable"}
        )

    try:
        # 1. 소유권 확인: 이 session_id가 해당 user_id의 세션 목록에 있는지 확인
        session_key = f"user:{user_id}:sessions_sorted"
        if r.zscore(session_key, session_id) is None:
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden: You do not have access to this session."}
            )

        # 2. 페이지네이션을 위해 Redis list에서 데이터 슬라이싱
        chat_key = f"chat:{session_id}"
        start_index = cursor
        end_index = cursor + limit - 1
        logs_raw = r.lrange(chat_key, start_index, end_index)

        # 3. 데이터 형식 변환: [{"question":...}, {"answer":...}] -> [{"role":"user",...}, {"role":"assistant",...}]
        messages = []
        for item_raw in logs_raw:
            item = json.loads(item_raw)
            question = item.get("question")
            answer = item.get("answer")
            timestamp = item.get("timestamp")

            if question:
                messages.append({"role": "user", "text": question, "timestamp": timestamp})
            if answer:
                messages.append({"role": "assistant", "text": answer, "timestamp": timestamp})

        # 4. 다음 페이지 커서 계산
        next_cursor = None
        # 요청한 limit만큼의 데이터를 성공적으로 가져왔다면, 다음 페이지가 있을 수 있음
        if len(logs_raw) == limit:
            new_cursor = cursor + limit
            # 전체 로그 개수를 확인하여 다음 페이지가 실제로 있는지 최종 확인
            total_logs = r.llen(chat_key)
            if new_cursor < total_logs:
                next_cursor = new_cursor

        # 5. 최종 응답 반환
        return JSONResponse(content={
            "session_id": session_id,
            "messages": messages,
            "next_cursor": next_cursor
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {e}"}
        )
        
# main.py 파일에 추가

@app.delete("/delete_session/{session_id}")
async def delete_session(session_id: str, user_id: str):
    """
    특정 세션의 대화 기록을 삭제합니다.
    소유권 확인을 위해 user_id가 필요합니다.
    """
    if not r:
        return JSONResponse(status_code=503, content={"error": "Redis service is unavailable"})

    try:
        # 1. 소유권 확인
        session_list_key = f"user:{user_id}:sessions_sorted"
        if r.zscore(session_list_key, session_id) is None:
            return JSONResponse(status_code=403, content={"error": "Forbidden"})

        # 2. 세션 목록에서 해당 session_id 제거
        r.zrem(session_list_key, session_id)

        # 3. 실제 대화 내용 데이터 삭제
        chat_key = f"chat:{session_id}"
        r.delete(chat_key)

        print(f"✅ 세션 삭제 성공: user='{user_id}', session='{session_id}'")
        return JSONResponse(status_code=200, content={"message": "Session deleted successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# app/main.py 파일에 추가

@app.get("/sessions/latest/{user_id}")
async def get_latest_session_history(user_id: str):
    """
    특정 사용자의 가장 최근 대화 세션 기록 전체를 반환합니다.
    """
    if not r:
        return JSONResponse(status_code=503, content={"error": "Redis service is unavailable"})

    try:
        # 1. 해당 사용자의 가장 최근 session_id를 1개만 가져옵니다.
        session_list_key = f"user:{user_id}:sessions_sorted"
        latest_session_ids = r.zrevrange(session_list_key, 0, 0)
        
        if not latest_session_ids:
            # 이 사용자의 대화 기록이 아예 없는 경우
            return JSONResponse(content={"session_id": None, "messages": []})

        latest_session_id = latest_session_ids[0]

        # 2. 해당 session_id의 전체 대화 기록을 가져옵니다.
        chat_key = f"chat:{latest_session_id}"
        logs_raw = r.lrange(chat_key, 0, -1)

        messages = []
        for item_raw in logs_raw:
            item = json.loads(item_raw)
            question, answer = item.get("question"), item.get("answer")
            if question: messages.append({"role": "user", "text": question})
            if answer: messages.append({"role": "assistant", "text": answer})

        # 3. 프론트엔드가 대화를 이어갈 수 있도록 session_id와 메시지 목록을 함께 반환합니다.
        return JSONResponse(content={
            "session_id": latest_session_id,
            "messages": messages
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    


@app.get("/debug-db-files")
def debug_db_files():
    """
    서버에 배포된 vector_store 폴더의 실제 파일 목록과 크기를 확인하는 진단용 엔드포인트.
    """
    base_path = os.path.join(os.path.dirname(__file__), '..', 'vector_store')
    file_info = {}
    
    if not os.path.exists(base_path):
        return {"error": f"vector_store 폴더를 찾을 수 없습니다: {base_path}"}
        
    for root, dirs, files in os.walk(base_path):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                # 파일 경로에서 /app/ 부분을 기준으로 상대 경로를 만듭니다.
                relative_path = os.path.relpath(file_path, start=os.path.join(os.path.dirname(__file__), '..'))
                file_size_bytes = os.path.getsize(file_path)
                file_info[relative_path] = f"{file_size_bytes} bytes"
            except Exception as e:
                file_info[file_path] = f"크기 확인 오류: {e}"
                
    return file_info