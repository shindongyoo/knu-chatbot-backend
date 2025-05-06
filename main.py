import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
import certifi
from fastapi.responses import PlainTextResponse
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
        f"위 내용을 참고하여 질문에 대해 요약된 답변을 작성해주세요.\n"
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

@app.post("/ask", response_class=PlainTextResponse)
async def ask(req: QuestionRequest):
    try:
        keyword = req.question.strip()
        all_results = []

        chatbot_db = mongo_client.chatbot_database
        collections = chatbot_db.list_collection_names()

        unique_results = {}

        for coll_name in collections:
            coll = chatbot_db[coll_name]
            regex = "|".join(keyword.split())  # "졸업요건 알려줘" → "졸업요건|알려줘"
            docs = coll.find({
                "$or": [
                    {"title": {"$regex": regex, "$options": "i"}},
                    {"content": {"$regex": regex, "$options": "i"}}
                ]
            })
            for doc in docs:
                title = doc.get("title", "제목 없음")
                url = doc.get("url", "")
                
                if title not in unique_results:
                    key = f"{title}__{url}"
                    if key not in unique_results:
                        unique_results[key] = (title, url)

        MAX_DOCS = 10     
               
        all_results = list(unique_results.values())[:MAX_DOCS]  # ✅ 중복 제거
        
        context = "\n".join(
            f"- {title}: {url}" if url else f"- {title}"
            for (title, url) in all_results
        )




        if all_results:
            context = "\n".join(
                f"- [{title}]({url})" if url else f"- {title}"
                for (title, url) in all_results
            )
            answer = generate_answer(req.question, context)
            return answer  # ✅ 이제 AI 요약 결과를 실제로 반환함!
        else:
            return f"'{keyword}' 관련된 문서를 찾지 못했습니다.'"



    except Exception as e:
        print("❗ 예외 발생:", e)
        return f"Error: {str(e)}"
    
#프론트 연결
    