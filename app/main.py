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

# --- .env 로드 및 디버깅 (서버 시작 시 한 번만 실행) ---
load_dotenv()
print("--- 디버깅 정보 ---")
print(f".env 파일 로드 성공 여부: {load_dotenv()}")
print(f"읽어온 OPENAI_API_KEY 값: {os.getenv('OPENAI_API_KEY')}")
print("--------------------")

# search_engine을 여기서 import해야 DB 로딩 메시지가 먼저 뜹니다.
from app.search_engine import search_similar_documents

# --- 서비스 초기화 ---

# OpenAI 클라이언트 초기화 (최신 v1.x 방식)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MongoDB 설정
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where()
)
chatbot_db = mongo_client.chatbot_database

# Redis 설정
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True
    )
    r.ping() # 연결 테스트
    print("✅ Redis 연결 성공.")
except Exception as e:
    print(f"❌ Redis 연결 실패: {e}")
    r = None # Redis 연결 실패 시 r을 None으로 설정

# FastAPI 앱 설정
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://knu-chatbot.github.io"], # 실제 프론트엔드 주소에 맞게 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델
class QuestionRequest(BaseModel):
    session_id: str
    question: str

# --- 헬퍼 함수 ---

def get_recent_history(session_id: str, n=3) -> str:
    if not r: return "" # Redis 연결 실패 시 빈 문자열 반환
    try:
        key = f"chat:{session_id}"
        logs = r.lrange(key, -n * 2, -1) # 질문/답변 쌍을 고려하여 2*n개 가져오기
        dialogue = []
        for item in logs:
            parsed = json.loads(item)
            q = parsed.get("question")
            a = parsed.get("answer")
            if q: dialogue.append(f"사용자: {q}")
            if a: dialogue.append(f"챗봇: {a}")
        return "\n".join(dialogue) if dialogue else ""
    except Exception as e:
        print(f"Redis 히스토리 조회 오류: {e}")
        return ""

def save_chat_history(session_id: str, question: str, answer: str):
    if not r: return # Redis 연결 실패 시 저장 안함
    try:
        key = f"chat:{session_id}"
        timestamp = datetime.utcnow().isoformat()
        log_entry = json.dumps({
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        })
        r.rpush(key, log_entry)
        r.expire(key, 3 * 24 * 60 * 60) # 3일 후 만료
    except Exception as e:
        print(f"Redis 히스토리 저장 오류: {e}")

# --- API 엔드포인트 ---

@app.get("/")
def root():
    return {"message": "KNU Chatbot backend is running"}

@app.post("/ask")
async def ask(req: QuestionRequest):
    try:
        # 1. 컨텍스트 및 히스토리 준비
        recent_history = get_recent_history(req.session_id)
        context, _ = search_similar_documents(req.question)

        # 2. 프롬프트 생성
        prompt = f"""당신은 경북대학교에 대한 질문에 친절하게 답변하는 챗봇입니다. 아래 제공된 '검색된 참고 자료'를 바탕으로 사용자의 질문에 답변해주세요. 자료에 없는 내용은 답변하지 마세요.

### 이전 대화 기록:
{recent_history}

### 검색된 참고 자료:
{context}

### 사용자의 질문:
{req.question}

### 답변:
"""
        print(f"[ASK] 최종 프롬프트(앞 500자): {prompt[:500]}")

        # 3. OpenAI API 호출 (최신 v1.x 방식)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 경북대학교 지식 기반 챗봇입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()

        # 4. 대화 기록 저장
        save_chat_history(req.session_id, req.question, answer)

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": f"답변 생성 중 오류: {e}"}, status_code=500)

# (파일 업로드, 히스토리 조회 등 다른 엔드포인트는 기존 코드를 유지하셔도 좋습니다)
# ... 기존의 /upload, /history 엔드포인트 코드 ...