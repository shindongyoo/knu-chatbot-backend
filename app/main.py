# app/main.py
import os
import json
import time
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import openai
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract

from dotenv import load_dotenv
load_dotenv() # .env 파일을 여기서 먼저 로드합니다.

# --- 서비스 초기화 ---
# OpenAI 클라이언트 초기화 (최신 v1.x 방식)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ▼▼▼ [수정] database.py에서 DB 객체들을 import 합니다. ▼▼▼
from app.database import chatbot_db, r

# ▼▼▼ [수정] DB 초기화가 끝난 후에 search_engine을 import 합니다. ▼▼▼
from app.search_engine import search_similar_documents

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

def get_recent_history(session_id: str, n=3) -> str:
    if not r: return ""
    try:
        key = f"chat:{session_id}"
        logs = r.lrange(key, -n * 2, -1)
        dialogue = []
        for item in logs:
            parsed = json.loads(item)
            q, a = parsed.get("question"), parsed.get("answer")
            if q: dialogue.append(f"사용자: {q}")
            if a: dialogue.append(f"챗봇: {a}")
        return "\n".join(dialogue)
    except Exception as e:
        print(f"Redis 히스토리 조회 오류: {e}")
        return ""

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

# 이 함수를 새로 추가해야 합니다.
def transform_query(query: str) -> list[str]:
    """AI를 사용해 사용자의 애매한 질문을 구체적인 검색어로 변환합니다."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 사용자의 질문을 Vector DB에서 검색하기 좋은 구체적인 검색어 3개로 바꾸는 쿼리 변환 전문가입니다. 각 검색어는 쉼표(,)로 구분해서 응답하세요."},
                {"role": "user", "content": f"다음 질문을 검색어로 바꿔줘: {query}"}
            ],
            temperature=0
        )
        transformed_queries = response.choices[0].message.content.strip()
        # "채용 공고, 인턴십, 신입 모집" -> ["채용 공고", "인턴십", "신입 모집"]
        return [q.strip() for q in transformed_queries.split(',')][:3] # 최대 3개
    except Exception as e:
        print(f"쿼리 변환 실패: {e}")
        return [query] # 실패하면 원래 질문으로 검색

@app.post("/stream")
async def stream_answer(req: QuestionRequest):
    question = req.question
    session_id = req.session_id
    user_id = req.user_id

    def event_generator():
        try:
            # 이 부분이 빠져서 발생했던 NameError를 해결한 코드입니다.
            recent = get_recent_history(session_id)
            
            # 1. AI를 호출해 질문을 검색어로 변환
            print(f"원본 질문: {question}")
            search_queries = transform_query(question)
            
            # 2. 변환된 모든 검색어로 DB에서 자료를 찾음
            all_contexts = []
            for q in search_queries:
                context, _ = search_similar_documents(q) # top_k=3이라면 총 3*3=9개 검색
                if context:
                    all_contexts.append(context)
            
            # 3. 모든 검색 결과를 하나로 합침 (중복 제거는 search_engine.py에서 이미 처리됨)
            final_context = "\n---\n".join(all_contexts)
            # ▲▲▲ [수정 완료] ▲▲▲
            
            # 4. (이후 프롬프트는 그대로...)
            
            # 컨텍스트 길이 제한
            MAX_CONTEXT_LENGTH = 7000
            if len(final_context) > MAX_CONTEXT_LENGTH:
                print(f"⚠️ [경고] 컨텍스트가 너무 깁니다. {len(context)}자를 {MAX_CONTEXT_LENGTH}자로 자릅니다.")
                context = context[:MAX_CONTEXT_LENGTH]


            system_prompt = """당신은 경북대학교 안내 챗봇입니다. 당신의 가장 중요한 임무는 '검색된 참고 자료'를 바탕으로 사용자의 질문에 **'정확하게'** 답변하는 것입니다.

            [답변 생성 3단계 규칙]
            1.  **[1단계: 엄격한 관련성 판단]** 먼저, '검색된 참고 자료'가 '사용자의 질문'과 **'정확히 일치하는지'** 엄격하게 판단합니다.
                * (예: 질문이 '삼성 채용'인데 자료가 'LG 채용'이면, 이것은 관련 없는 자료입니다.)
                * (예: 질문이 '졸업 요건'인데 자료가 '졸업식 안내'이면, 이것은 관련 없는 자료입니다.)

            2.  **[2단계: 답변 생성]**
                * **(A) '정확히 일치하는' 자료를 찾은 경우:** 해당 자료에 근거해서만 답변을 생성합니다.
                * **(B) '관련 없는' 자료만 찾은 경우 (가장 중요!):** 자료의 내용을 **절대 요약하거나 언급하지 마세요.** 이것은 '딴소리'입니다. 대신, "죄송합니다, 질문하신 내용과 정확히 일치하는 정보를 찾을 수 없습니다."라고만 답변하세요.

            3.  **[3단계: 거절 조건]**
                * '검색된 참고 자료' 섹션이 문자 그대로 완전히 비어 있을 경우에도 "죄송합니다, 관련된 정보를 찾을 수 없습니다."라고 답변하세요.
                * **절대 당신의 기존 지식을 사용해서는 안 됩니다.**
            """

            # 2. 'user' 프롬프트에는 질문과 참고 자료(데이터)만 깔끔하게 전달합니다.
            user_prompt = f"""
            ### 검색된 참고 자료:
            {context}

            ### 이전 대화 기록:
            {recent}

            ### 사용자의 질문:
            {question}

            ### 답변:
            """
            
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},  # <-- 수정: AI의 역할과 규칙
                    {"role": "user", "content": user_prompt}     # <-- 수정: 질문과 데이터
                ],
                stream=True,
                temperature=0.2
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
        recent = get_recent_history(req.session_id)
        context, _ = search_similar_documents(req.question)

        system_prompt = """당신은 경북대학교 안내 챗봇입니다. 당신의 가장 중요한 임무는 '검색된 참고 자료'를 바탕으로 사용자의 질문에 **'정확하게'** 답변하는 것입니다.

        [답변 생성 3단계 규칙]
        1.  **[1단계: 엄격한 관련성 판단]** 먼저, '검색된 참고 자료'가 '사용자의 질문'과 **'정확히 일치하는지'** 엄격하게 판단합니다.
            * (예: 질문이 '삼성 채용'인데 자료가 'LG 채용'이면, 이것은 관련 없는 자료입니다.)
            * (예: 질문이 '졸업 요건'인데 자료가 '졸업식 안내'이면, 이것은 관련 없는 자료입니다.)

        2.  **[2단계: 답변 생성]**
            * **(A) '정확히 일치하는' 자료를 찾은 경우:** 해당 자료에 근거해서만 답변을 생성합니다.
            * **(B) '관련 없는' 자료만 찾은 경우 (가장 중요!):** 자료의 내용을 **절대 요약하거나 언급하지 마세요.** 이것은 '딴소리'입니다. 대신, "죄송합니다, 질문하신 내용과 정확히 일치하는 정보를 찾을 수 없습니다."라고만 답변하세요.

        3.  **[3단계: 거절 조건]**
            * '검색된 참고 자료' 섹션이 문자 그대로 완전히 비어 있을 경우에도 "죄송합니다, 관련된 정보를 찾을 수 없습니다."라고 답변하세요.
            * **절대 당신의 기존 지식을 사용해서는 안 됩니다.**
        """
        
        # 2. 'user' 프롬프트에는 질문과 참고 자료(데이터)만 전달합니다.
        user_prompt = f"""
        ### 검색된 참고 자료:
        {context}

        ### 이전 대화 기록:
        {recent}

        ### 사용자의 질문:
        {req.question}

        ### 답변:
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},  # <-- 수정: AI의 역할과 규칙
                {"role": "user", "content": user_prompt}     # <-- 수정: 질문과 데이터
            ],
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
    
    # app/main.py 파일 맨 아래에 추가하세요.

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