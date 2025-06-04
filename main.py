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
from fastapi import UploadFile, File, Form
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from datetime import datetime

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
            f"답변을 할 때는 반드시 자연스러운 한국어 띄어쓰기를 모두 적용해서 출력하세요. 붙여쓰기가 있는 부분은 전부 띄어쓰기를 바로잡아 주세요.\n"

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
        
        print("DEBUG 답변: ", repr(answer))

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
        # 여기서 한글자씩 yield하지 않고, 누적만 하다가 마지막에 yield!
        buffer = ""
        for chunk in response:
            delta = getattr(chunk.choices[0].delta, "content", "") or ""
            buffer += delta
            # 예시) 문장 끝(마침표 등) 기준으로 적당히 flush
            if buffer.endswith(("다.", "\n")) or len(buffer) > 5:
                yield f"data: {buffer}\n\n"
                buffer = ""
        if buffer:
            yield f"data: {buffer}\n\n"

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

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 폴더 없으면 자동 생성

@app.post("/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    print(f"[UPLOAD] 업로드 요청: session_id={session_id}, filename={file.filename}", flush=True)
    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 1. 파일 저장 전후로 print
    print(f"[UPLOAD] 업로드 시도: {filename}", flush=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    print(f"[UPLOAD] 파일 저장 완료: {file_path}", flush=True)
    
    extracted_text = ""
    try:
        if filename.lower().endswith(".pdf"):
            print(f"[UPLOAD] PDF 파일 처리 시작: {file_path}", flush=True)
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    extracted_text += txt + "\n"
            print(f"[UPLOAD] PDF 텍스트 추출 완료", flush=True)
        elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"[UPLOAD] 이미지 파일 처리 시작: {file_path}", flush=True)
            try:
                image = Image.open(file_path)
                print("[UPLOAD] 이미지 열기 성공", flush=True)
                extracted_text = pytesseract.image_to_string(image, lang="kor+eng")
                print("[UPLOAD] OCR 추출 성공", flush=True)
            except Exception as e:
                print("[UPLOAD][IMAGE ERROR]", e, flush=True)
                import traceback
                traceback.print_exc()
                return JSONResponse(content={"error": f"이미지 처리 중 오류: {e}"}, status_code=400)
        else:
            print("[UPLOAD][ERROR] 지원하지 않는 파일 형식", flush=True)
            return JSONResponse(content={"error": "지원하지 않는 파일 형식입니다."}, status_code=400)
    except Exception as e:
        print("[UPLOAD][ERROR]", e, flush=True)
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": f"파일 처리 중 오류: {e}"}, status_code=400)

    print(f"[UPLOAD] 최종 extracted_text(앞 200글자): {extracted_text[:200]}", flush=True)

    chatbot_db.uploaded_files.update_one(
        {"session_id": session_id},
        {"$set": {
            "text": extracted_text,
            "filename": filename,
            "uploaded_at": datetime.utcnow()
        }},
        upsert=True
    )
    print("[UPLOAD] MongoDB 저장 완료", flush=True)
    return {"msg": "MongoDB 저장 성공", "text_length": len(extracted_text)}

# TTL 인덱스 최초 1회 생성 코드(운영 환경에서는 한 번만 실행!)
chatbot_db.uploaded_files.create_index(
    [("uploaded_at", 1)],
    expireAfterSeconds=3 * 24 * 60 * 60  # 3일
)

@app.post("/ask", response_class=JSONResponse)
async def ask(req: QuestionRequest):
    try:
        # 파일에서 추출된 텍스트 가져오기
        file_doc = chatbot_db.uploaded_files.find_one({"session_id": req.session_id})
        file_context = file_doc["text"] if file_doc and "text" in file_doc else ""
        
        print(f"[ASK] file_context(앞 500글자):\n{file_context[:500]}")

        # ... 기존 get_context_and_fields 코드 ...
        context, field_names = get_context_and_fields(req.question)
        full_context = (file_context + "\n" if file_context else "") + context

        # ... prompt와 GPT 호출에서 context → full_context 사용 ...
        prompt = (
            f"파일에서 추출된 내용:\n{file_context}\n\n" if file_context else ""
        ) + (
            f"아래는 관련 문서 정보입니다:\n{context}\n\n"
        ) + (
            f"사용자 질문: {req.question}"
        )

        print(f"[ASK] prompt(앞 800글자):\n{prompt[:800]}")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 친절한 경북대 전기과 졸업요건 안내 챗봇이야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        answer = response.choices[0].message.content.strip()
        print("DEBUG 답변: ", repr(answer))
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        import traceback
        print("[UPLOAD][ERROR]", e)
        traceback.print_exc()
        return JSONResponse(content={"error": f"파일 처리 중 오류: {e}"}, status_code=400)
