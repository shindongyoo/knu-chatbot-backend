import os
import re
import json
import certifi
import redis
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
#from keybert import KeyBERT
#from kiwipiepy import Kiwi

# Load environment variables
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    tlsAllowInvalidCertificates=True,
    tlsAllowInvalidHostnames=True
)
chatbot_db = mongo_client.chatbot_database

# Redis setup (로컬)
r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)


# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class QuestionRequest(BaseModel):
    session_id: str
    question: str

# 키워드 추출기
#kw_model = KeyBERT()
#kiwi = Kiwi()

def extract_keywords_korean(question: str, top_n=5) -> list:
    return [question.strip()]


def get_context_and_fields(question: str):
    keywords = extract_keywords_korean(question)
    regex = '|'.join(keywords)

    collections = chatbot_db.list_collection_names()
    unique_results = {}
    all_docs = []

    for coll_name in collections:
        coll = chatbot_db[coll_name]
        docs = coll.find({
            "$or": [
                {field: {"$regex": regex, "$options": "i"}} for field in [
                    "title", "content", "name", "position", "major", "section",
                    "body", "date", "type", "phone", "email", "homepage", "lab"
                ]
            ]
        })
        for doc in docs:
            key = str(doc.get("_id", ""))
            if key not in unique_results:
                unique_results[key] = True
                all_docs.append(doc)

    MAX_DOCS = 10
    priority_docs, other_docs = [], []
    for doc in all_docs:
        combined_text = " ".join([
            doc.get("name", ""), doc.get("title", ""),
            doc.get("body", ""), doc.get("content", "")
        ])
        if any(k in combined_text for k in keywords):
            priority_docs.append(doc)
        else:
            other_docs.append(doc)

    selected_docs = (priority_docs + other_docs)[:MAX_DOCS]

    context = ""
    field_names = set()
    for doc in selected_docs:
        context += "- 문서 정보:\n"
        for key, value in doc.items():
            if value:
                context += f"  {key}: {value}\n"
                field_names.add(key)
        context += "|\n"

    return context, field_names

# 최근 대화 불러오기 함수 (마지막 n개 Q/A)
def get_recent_history(session_id: str, n=4) -> str:
    key = f"chat:{session_id}"
    logs = r.lrange(key, -n, -1)  # 마지막 n개
    dialogue = []
    for item in logs:
        parsed = json.loads(item)
        q = parsed.get("question")
        a = parsed.get("answer")
        if q:
            dialogue.append(f"사용자: {q}")
        if a:
            dialogue.append(f"챗봇: {a}")
    return "\n".join(dialogue) if dialogue else ""

@app.post("/ask", response_class=JSONResponse)
async def ask(req: QuestionRequest):
    try:
        # 최근 대화 불러오기 (ex: 4개)
        recent = get_recent_history(req.session_id, n=4)
        context, field_names = get_context_and_fields(req.question)
        prompt = (
            (f"이전 대화 기록:\n{recent}\n\n" if recent else "") +
            f"사용자의 질문: '{req.question}'\n\n"
            f"아래는 관련 문서들의 다양한 정보입니다:\n{context}\n\n"
            f"각 문서에는 다음과 같은 정보가 포함되어 있습니다: {', '.join(sorted(field_names))}.\n"
            f"가능한 모든 필드 값을 활용해서 질문에 답변해 주세요.\n"
            f"특히 lab, phone, email, homepage, url, content 등이 포함되어 있을 경우 반드시 응답에 포함해 주세요.\n"
            f"문서 제목과 링크도 자연스럽게 포함해 주세요.\n"
            f"질문과 관련 없는 문서는 제외하세요.\n"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 친절한 경북대 전기과 졸업요건 안내 챗봇이야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        answer = response.choices[0].message.content.strip()
        r.rpush(f"chat:{req.session_id}", json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "question": req.question,
            "answer": answer
        }))
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        print("❗ 예외 발생:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/stream")
async def stream_answer(req: Request):
    body = await req.json()
    session_id = body.get("session_id")
    question = body.get("question")

    # 최근 대화 내역
    recent = get_recent_history(session_id, n=4)
    context, field_names = get_context_and_fields(question)

    def event_generator():
        full_answer = ""
        prompt = (
            (f"이전 대화 기록:\n{recent}\n\n" if recent else "") +
            f"사용자의 질문: '{question}'\n\n"
            f"{context}\n\n"
            f"각 문서에는 다음과 같은 정보가 포함되어 있습니다: {', '.join(sorted(field_names))}."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 친절한 경북대 전기과 졸업요건 안내 챗봇이야."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta.get("content", "")
            full_answer += delta
            yield f"data: {delta}\n\n"
        r.rpush(f"chat:{session_id}", json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": full_answer
        }))

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    key = f"chat:{session_id}"
    logs = r.lrange(key, 0, -1)
    return JSONResponse(content={"history": [json.loads(item) for item in logs]})

@app.post("/history", response_class=JSONResponse)
async def post_history(req: Request):
    try:
        body = await req.json()
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse(content={"error": "session_id is required"}, status_code=400)

        key = f"chat:{session_id}"
        logs = r.lrange(key, 0, -1)

        history = []
        for item in logs:
            parsed = json.loads(item)
            history.append({"role": "user", "content": parsed["question"]})
            history.append({"role": "bot", "content": parsed["answer"]})

        return JSONResponse(content=history)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "KNU Chatbot backend is running"}
