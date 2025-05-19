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

# Redis setup
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

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

# Common helper function to extract documents and context
def get_context_and_fields(question: str):
    keyword = question.strip()
    cleaned = re.sub(r"[은는이가을를도에의와과로]", "", keyword)
    words = re.findall(r"[가-힣a-zA-Z0-9]+", cleaned)
    regex = f"{keyword}|{'|'.join(words)}"

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
        if any(word in combined_text for word in words):
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

# Regular GPT response API
@app.post("/ask", response_class=JSONResponse)
async def ask(req: QuestionRequest):
    try:
        context, field_names = get_context_and_fields(req.question)

        prompt = (
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

# Streaming SSE GPT response
@app.post("/stream")
async def stream_answer(req: Request):
    body = await req.json()
    session_id = body.get("session_id")
    question = body.get("question")

    context, field_names = get_context_and_fields(question)

    def event_generator():
        full_answer = ""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 친절한 경북대 전기과 졸업요건 안내 챗봇이야."},
                {"role": "user", "content": (
                    f"사용자의 질문: '{question}'\n\n{context}\n\n"
                    f"각 문서에는 다음과 같은 정보가 포함되어 있습니다: {', '.join(sorted(field_names))}."
                )}
            ],
            stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta.get("content", "")
            full_answer += delta

        r.rpush(f"chat:{session_id}", json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": full_answer
        }))

        yield f"data: {full_answer}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# 대화 기록 조회
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    key = f"chat:{session_id}"
    logs = r.lrange(key, 0, -1)
    return JSONResponse(content={"history": [json.loads(item) for item in logs]})
