import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
import certifi
from fastapi.responses import JSONResponse
from openai import OpenAI
import re
import redis
import json
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(user_question, context_text, field_names):
    prompt = (
        f"사용자의 질문: '{user_question}'\n\n"
        f"아래는 관련 문서들의 다양한 정보입니다:\n{context_text}\n\n"
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
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()

#mongo db
MONGO_URI = os.getenv("MONGO_URI")

app = FastAPI()

mongo_client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    tlsAllowInvalidCertificates=True,
    tlsAllowInvalidHostnames=True
)
db = mongo_client.KNUChatbot
collection = db.notices

#대화기록
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

class QuestionRequest(BaseModel):
    session_id: str
    question: str

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    key = f"chat:{session_id}"
    logs = r.lrange(key, 0, -1)
    return JSONResponse(content={"history": [json.loads(item) for item in logs]})

@app.post("/ask", response_class=JSONResponse)
async def ask(req: QuestionRequest):
    try:
        keyword = req.question.strip()
        cleaned = re.sub(r"[은는이가을를도에의와과로]", "", keyword)
        words = re.findall(r"[가-힣a-zA-Z0-9]+", cleaned)
        regex = f"{keyword}|{'|'.join(words)}"


        chatbot_db = mongo_client.chatbot_database
        collections = chatbot_db.list_collection_names()

        unique_results = {}
        all_docs = []

        for coll_name in collections:
            coll = chatbot_db[coll_name]
            docs = coll.find({
                "$or": [
                    {"title": {"$regex": regex, "$options": "i"}},
                    {"content": {"$regex": regex, "$options": "i"}},
                    {"name": {"$regex": regex, "$options": "i"}},
                    {"position": {"$regex": regex, "$options": "i"}},
                    {"major": {"$regex": regex, "$options": "i"}},
                    {"section": {"$regex": regex, "$options": "i"}},
                    {"body": {"$regex": regex, "$options": "i"}},
                    {"date": {"$regex": regex, "$options": "i"}},
                    {"type": {"$regex": regex, "$options": "i"}},
                    {"phone": {"$regex": regex, "$options": "i"}},
                    {"email": {"$regex": regex, "$options": "i"}},
                    {"homepage": {"$regex": regex, "$options": "i"}},
                    {"lab": {"$regex": regex, "$options": "i"}}
                ]
            })

            for doc in docs:
                key = str(doc.get("_id", ""))
                    

                if key not in unique_results:
                    unique_results[key] = True
                    all_docs.append({
                        "title": doc.get("title", "제목 없음"),
                        "url": doc.get("url", ""),
                        "major": doc.get("major", ""),
                        "position": doc.get("position", ""),
                        "section": doc.get("section", ""),
                        "body": doc.get("body", ""),
                        "content": doc.get("content", ""),
                        "date": doc.get("date", ""),
                        "type": doc.get("type", ""),
                        "phone": doc.get("phone", ""),
                        "email": doc.get("email", ""),
                        "homepage": doc.get("homepage", ""),
                        "lab": doc.get("lab", "")
                    })


        MAX_DOCS = 10
        priority_docs = []
        other_docs = []

        for doc in all_docs:
            combined_text = " ".join([
                doc.get("title", ""),
                doc.get("name", ""),
                doc.get("body", ""),
                doc.get("content", "")
            ])
            if any(word in combined_text for word in words):
                priority_docs.append(doc)
            else:
                other_docs.append(doc)

        selected_docs = (priority_docs + other_docs)[:MAX_DOCS]


        if selected_docs:
            context = ""
            field_names = set()

            for doc in selected_docs:
                context += "- 문서 정보:\n"
                for key, value in doc.items():
                    if value:
                        context += f"  {key}: {value}\n"
                        field_names.add(key)
                context += "|\n"

            answer = generate_answer(req.question, context, field_names)
            
            #Redis 대화저장
            chat = {
                "timestamp": datetime.utcnow().isoformat(),
                "question": req.question,
                "answer": answer
            }
            r.rpush(f"chat:{req.session_id}", json.dumps(chat)) #리스트로 누적적
            
            return JSONResponse(content={"answer": answer})
        else:
            return JSONResponse(content={"answer": f"'{keyword}' 관련된 문서를 찾지 못했습니다."})

    except Exception as e:
        print("❗ 예외 발생:", e)
        return JSONResponse(content={"error": str(e)})
    
#프론트 연결
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (필요 시 GitHub Pages 도메인만 설정 가능)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)