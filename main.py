import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
import certifi
from fastapi.responses import JSONResponse
import cohere


if not os.getenv("COHERE_API_KEY"):
    raise RuntimeError("COHERE_API_KEY가 설정되지 않았습니다!")
COHERE_MODEL = "command-nightly"


#cohere ai
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

def generate_answer(user_question, context_text):
    prompt = (
        f"사용자의 질문: '{user_question}'\n\n"
        f"아래는 관련 문서 제목과 링크입니다:\n{context_text}\n\n"
        f"이 문서 목록을 기반으로 질문과 관련된 정보들을 요약하지 말고, 문서 제목과 링크를 자연스럽게 정리해서 보여주세요.\n"
        f"관련 문서가 있다면, 그 링크도 자연스럽게 포함해주세요.\n"
        f"답변은 친절하고 이해하기 쉽게 작성해주세요."
    )

    response = cohere_client.chat(
        message=prompt,
        model="command-nightly",
        temperature=0.5,
        max_tokens=400
    )

    return response.text.strip()


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

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/ask", response_class=JSONResponse)
async def ask(req: QuestionRequest):
    try:
        keyword = req.question.strip()
        words = keyword.split()
        regex = "|".join(words)

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
                    {"body": {"$regex": regex, "$options": "i"}}
                ]
            })

            for doc in docs:
                title = doc.get("title", "제목 없음")
                url = doc.get("url", "")
                key = f"{title}_{url}"

                if key not in unique_results:
                    unique_results[key] = True
                    all_docs.append({
                        "title": title,
                        "url": url,
                        "major": doc.get("major", ""),
                        "position": doc.get("position", ""),
                        "section": doc.get("section", ""),
                        "body": doc.get("body", ""),
                        "content": doc.get("content", "")
                    })

        MAX_DOCS = 10
        selected_docs = all_docs[:MAX_DOCS]

        if selected_docs:
            context = "\n".join(
                f"- 제목: {doc['title']}\n"
                f"  전공: {doc['major']}\n"
                f"  직책: {doc['position']}\n"
                f"  소속: {doc['section']}\n"
                f"  링크: {doc['url']}\n"
                f"  내용: {doc['body'] or doc['content'] or '본문 없음'}"
                for doc in selected_docs
            )

            answer = generate_answer(req.question, context)
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