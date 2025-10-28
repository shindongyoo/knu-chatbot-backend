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

# app/main.py의 stream_answer 함수를 이걸로 통째로 덮어쓰세요.

@app.post("/stream")
async def stream_answer(req: QuestionRequest):
    question = req.question
    session_id = req.session_id
    user_id = req.user_id

    def event_generator():
        try:
            # 이 부분이 빠져서 발생했던 NameError를 해결한 코드입니다.
            recent = get_recent_history(session_id)
            
            context, _ = search_similar_documents(question)
            
            # 컨텍스트가 너무 길 경우를 대비한 안전장치
            MAX_CONTEXT_LENGTH = 7000
            if len(context) > MAX_CONTEXT_LENGTH:
                print(f"⚠️ [경고] 컨텍스트가 너무 깁니다. {len(context)}자를 {MAX_CONTEXT_LENGTH}자로 자릅니다.")
                context = context[:MAX_CONTEXT_LENGTH]


            system_prompt = """당신은 경북대학교 안내 챗봇입니다. 당신의 가장 중요한 임무는 '검색된 참고 자료'를 바탕으로 사용자에게 **최대한 도움이 되는 답변**을 생성하는 것입니다.

            [답변 생성 규칙]
            1.  **답변의 1순위:** 당신의 답변은 **반드시 '검색된 참고 자료'에 있는 내용에 근거**해야 합니다. 당신의 기존 지식을 사용해서는 안 됩니다.
            2.  **자료 요약의 의무:** '검색된 참고 자료'는 이미 데이터베이스에서 사용자의 질문과 가장 관련성이 높다고 판단된 내용입니다. **당신의 임무는 이 자료를 친절하게 요약해서 전달하는 것입니다.**
            3.  **유연한 답변:** 만약 사용자의 질문("A")이 "취업 정보"처럼 광범위한데, 찾은 자료("B")가 "[삼성전자] 채용 공고"처럼 구체적이더라도, "정보를 찾을 수 없다"고 답변해서는 안 됩니다.
                * **대신, "찾아낸 관련 정보"를 알려주는 방식으로 답변하세요.**
                * **예시:** "질문하신 내용과 관련하여 [삼성전자] 채용 공고 정보를 찾았습니다. 주요 내용은..."
            4.  **유일한 거절 조건:** **오직 '검색된 참고 자료' 섹션이 문자 그대로 완전히 비어 있을 경우에만** "죄송합니다, 관련된 정보를 찾을 수 없습니다."라고 답변하세요. 
                * '검색된 참고 자료'에 단 하나의 단어라도 있다면, 당신은 반드시 그 내용을 바탕으로 답변을 구성해야 합니다."""

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

        system_prompt = """당신은 경북대학교 안내 챗봇입니다. 당신의 가장 중요한 임무는 '검색된 참고 자료'를 바탕으로 사용자에게 **최대한 도움이 되는 답변**을 생성하는 것입니다.

        [답변 생성 규칙]
        1.  **답변의 1순위:** 당신의 답변은 **반드시 '검색된 참고 자료'에 있는 내용에 근거**해야 합니다. 당신의 기존 지식을 사용해서는 안 됩니다.
        2.  **자료 요약의 의무:** '검색된 참고 자료'는 이미 데이터베이스에서 사용자의 질문과 가장 관련성이 높다고 판단된 내용입니다. **당신의 임무는 이 자료를 친절하게 요약해서 전달하는 것입니다.**
        3.  **유연한 답변:** 만약 사용자의 질문("A")이 "취업 정보"처럼 광범위한데, 찾은 자료("B")가 "[삼성전자] 채용 공고"처럼 구체적이더라도, "정보를 찾을 수 없다"고 답변해서는 안 됩니다.
            * **대신, "찾아낸 관련 정보"를 알려주는 방식으로 답변하세요.**
            * **예시:** "질문하신 내용과 관련하여 [삼성전자] 채용 공고 정보를 찾았습니다. 주요 내용은..."
        4.  **유일한 거절 조건:** **오직 '검색된 참고 자료' 섹션이 문자 그대로 완전히 비어 있을 경우에만** "죄송합니다, 관련된 정보를 찾을 수 없습니다."라고 답변하세요."""

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