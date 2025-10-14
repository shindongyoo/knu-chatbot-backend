# app/main.py
import os
import json
import certifi
import redis
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import openai
import time
from fastapi import FastAPI, Request, UploadFile, File, Form, Query

load_dotenv()

# --- 서비스 초기화 ---
# OpenAI 클라이언트 초기화 (최신 v1.x 방식)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MongoDB 설정
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
chatbot_db = mongo_client.chatbot_database

from app.search_engine import search_similar_documents

# Redis 설정
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True
    )
    r.ping()
    print("✅ Redis 연결 성공.")
except Exception as e:
    print(f"❌ Redis 연결 실패: {e}")
    r = None

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

# 아래 코드를 @app.post("/ask") 위에 붙여넣으세요.

@app.post("/stream")
async def stream_answer(req: QuestionRequest):
    question = req.question
    session_id = req.session_id
    user_id = req.user_id # <-- user_id를 요청 본문에서 추출

    def event_generator():
        try:
            recent = get_recent_history(session_id)
            context, _ = search_similar_documents(question)
            
            prompt = f"""당신은 경북대학교에 대한 질문에 답변하는 친절한 챗봇입니다. 아래 제공된 '검색된 참고 자료'를 바탕으로 사용자의 질문에 답변해주세요. 자료에 없는 내용은 답변하지 마세요.
            ### 이전 대화 기록:
            {recent}
            ### 검색된 참고 자료:
            {context}
            ### 사용자의 질문:
            {question}
            ### 답변:
            """
            
            # OpenAI API 호출 (최신 v1.x 스트리밍 방식)
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 경북대학교 지식 기반 챗봇입니다."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )

            collected_answer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    collected_answer += delta
                    # 프론트엔드로 데이터 전송
                    yield f"data: {json.dumps({'text': delta})}\n\n"
            
            save_chat_history(user_id, session_id, question, collected_answer)

        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
            error_message = json.dumps({"error": "답변 생성 중 오류가 발생했습니다."})
            yield f"data: {error_message}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/ask")
async def ask(req: QuestionRequest):
    try:
        recent = get_recent_history(req.session_id)
        context, _ = search_similar_documents(req.question)

        prompt = f"""당신은 경북대학교에 대한 질문에 답변하는 친절한 챗봇입니다. 아래 제공된 '검색된 참고 자료'를 바탕으로 사용자의 질문에 답변해주세요. 자료에 없는 내용은 답변하지 마세요.
        ### 이전 대화 기록:
        {recent}
        ### 검색된 참고 자료:
        {context}
        ### 사용자의 질문:
        {req.question}
        ### 답변:
        """

        # OpenAI API 호출 (최신 v1.x 방식)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 경북대학교 지식 기반 챗봇입니다."},
                {"role": "user", "content": prompt}
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
