# app/main.py (진짜 최종 완성본)

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

load_dotenv()
from app.search_engine import search_similar_documents

# --- 서비스 초기화 ---
# OpenAI 클라이언트 초기화 (구버전 v0.28.1 방식)
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    allow_origins=["*"], # 테스트를 위해 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    session_id: str
    question: str

def get_recent_history(session_id: str, n=3) -> str:
    # ... (이하 기존 코드와 동일) ...
    if not r: return ""
    try:
        key = f"chat:{session_id}"
        logs = r.lrange(key, -n * 2, -1)
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
    # ... (이하 기존 코드와 동일) ...
    if not r: return
    try:
        key = f"chat:{session_id}"
        timestamp = datetime.utcnow().isoformat()
        log_entry = json.dumps({
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        })
        r.rpush(key, log_entry)
        r.expire(key, 3 * 24 * 60 * 60)
    except Exception as e:
        print(f"Redis 히스토리 저장 오류: {e}")

@app.get("/")
def root():
    return {"message": "KNU Chatbot backend is running"}

@app.post("/stream")
async def stream_answer(req: QuestionRequest):
    question = req.question
    session_id = req.session_id

    def event_generator():
        try:
            recent = get_recent_history(session_id, n=3)
            context, field_names = search_similar_documents(question)

            prompt = f"""당신은 경북대학교에 대한 질문에 답변하는 친절한 챗봇입니다. 아래 '검색된 참고 자료'를 바탕으로 사용자의 질문에 답변해주세요.
            
            ### 이전 대화 기록:
            {recent}

            ### 검색된 참고 자료:
            {context}

            ### 사용자의 질문:
            {question}

            ### 답변:
            """
            
            # OpenAI API 호출 (구버전 v0.28.1 스트리밍 방식)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "너는 친절한 경북대 전기과 졸업요건 안내 챗봇이야."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )

            collected_answer = ""
            for chunk in response:
                # 구버전 응답 방식에 맞게 수정
                delta = chunk['choices'][0]['delta'].get("content", "")
                if delta:
                    collected_answer += delta
                    # 프론트엔드로 데이터 전송
                    yield f"data: {json.dumps({'text': delta})}\n\n"
            
            # 전체 답변이 완성된 후 Redis에 저장
            save_chat_history(session_id, question, collected_answer)

        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            error_message = json.dumps({"error": "답변 생성 중 오류가 발생했습니다."})
            yield f"data: {error_message}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ... 기존의 /ask, /upload, /history 엔드포인트 코드 ...
# (다른 엔드포인트들도 openai v0.28.1 방식으로 수정해야 하지만, 우선 스트리밍부터 확인)