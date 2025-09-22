# app/main.py (최신 버전 최종 완성본)

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

load_dotenv()
from app.search_engine import search_similar_documents

# --- 서비스 초기화 ---
# OpenAI 클라이언트 초기화 (최신 v1.x 방식)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MongoDB 설정
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
chatbot_db = mongo_client.chatbot_database

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