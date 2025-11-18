# app/main.py
import os
import json
import time
import openai
import pytesseract
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from PIL import Image
from dotenv import load_dotenv
import traceback # 오류 로깅

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from app.database import chatbot_db, r

# --- [핵심 수정: 함수들을 '도구'로 import] ---
from app.search_engine import search_similar_documents, get_graduation_info
tools = [search_similar_documents, get_graduation_info]
# ----------------------------------------

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

def get_recent_history(session_id: str, n=5) -> list[dict]: # n=3 -> n=5 (조절 가능)
    """
    Redis에서 최근 N개의 대화 기록을 'messages' API 형식으로 불러옵니다.
    (예: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}])
    """
    if not r: return [] # 빈 리스트 반환
    try:
        key = f"chat:{session_id}"
        logs = r.lrange(key, -n * 2, -1) # 최근 N*2 (Q, A) 항목
        
        messages = []
        for item in logs:
            parsed = json.loads(item)
            q = parsed.get("question")
            a = parsed.get("answer")
            
            # 저장된 순서(Q, A)를 보장하기 위해 확인 후 추가
            if q: messages.append({"role": "user", "content": q})
            if a: messages.append({"role": "assistant", "content": a})
        
        return messages # list[dict] 반환
    
    except Exception as e:
        print(f"Redis 히스토리 조회 오류: {e}")
        return [] # 오류 시 빈 리스트 반환

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

AGENT_SYSTEM_PROMPT = """당신은 경북대학교 전기공학과 학생들을 돕는 '자율 AI 비서'입니다.
당신은 스스로 '생각'하고, '계획'을 세우며, '도구'를 사용하여 답변을 찾습니다.

[당신이 사용 가능한 도구]
1. `search_similar_documents`: "수강신청", "장학생", "교수님" 등 '졸업 요건'을 제외한 모든 일반 정보를 검색합니다.
2. `get_graduation_info`: '졸업 요건'에 대한 상세 정보를 검색합니다.

[행동 지침]
1.  **[맥락 파악]** '이전 대화 기록'을 확인하여 후속 질문(예: "그럼 이메일은?")인지 파악합니다.
2.  **[사고 및 계획]** 사용자의 질문 의도를 파악하고, 어떤 도구를 사용해야 할지 결정합니다.
3.  **[졸업요건 특별 규칙]**
    * 만약 사용자가 "졸업 요건"(예: "졸업하려면?", "필수 과목")을 물어봤는데, `student_id_prefix`나 `abeek_bool` 정보를 모른다면, **절대 도구를 사용하지 마세요.**
    * 대신, **당신의 지식으로** 사용자에게 "졸업 요건을 확인하기 위해, 학번(예: 18, 19, 20...)과 ABEEK 이수 여부(O/X)를 **'18/O'** 형식으로 입력해주세요."라고 **반드시 되물어야 합니다.**
    * 사용자가 "18/O"라고 답하면, 그제서야 `abeek_bool=True`, `student_id_prefix="18"`로 `get_graduation_info` 도구를 호출하세요.
4.  **[일반 질문 규칙]**
    * "장학생", "수강신청", "한세경 교수" 등 다른 모든 질문은 `search_similar_documents` 도구를 사용하세요.
    * 도구 검색 결과가 엉뚱하면(예: "장학생" 질문에 "선거일" 응답), "죄송합니다, 관련 정보를 찾지 못했습니다."라고 답변하세요.
5.  **[잡담 규칙]**
    * "안녕?" 같은 단순 대화는 도구 없이 당신의 지식으로 친절하게 대답하세요.
"""

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"), # 대화 기록이 들어올 자리
    ("user", "{input}"), # 사용자의 새 질문
    MessagesPlaceholder(variable_name="agent_scratchpad"), # AI의 '생각'이 들어올 자리
])

# --- [에이전트 실행기 생성] ---
agent = create_openai_functions_agent(client, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, # AI의 '생각' 과정을 터미널에 출력 (매우 중요!)
    handle_parsing_errors=True # AI가 실수해도 멈추지 않게 함
)


@app.post("/stream")
def stream_answer(req: QuestionRequest):
    question = req.question
    session_id = req.session_id
    user_id = req.user_id

    def event_generator():
        try:
            # --- [수정] 모든 if문, 상태 관리 로직 삭제 ---
            print(f"[Agent Handling] '{question}'")
            
            # 1. 이전 대화 기록 불러오기 (API 형식)
            history_messages = get_recent_history(session_id, n=5)
            
            # 2. 에이전트 스트리밍 실행
            response_stream = agent_executor.stream({
                "input": question,
                "chat_history": history_messages
            })
            
            collected_answer = ""
            for chunk in response_stream:
                # agent.stream()은 'output' 키로 최종 답변 조각을 줍니다.
                if "output" in chunk:
                    delta = chunk["output"]
                    collected_answer += delta
                    yield f"data: {json.dumps({'text': delta})}\n\n"
                elif "steps" in chunk:
                    # AI가 어떤 '도구'를 '왜' 쓰는지 로그로 확인
                    print(f"[Agent Step] {chunk['steps']}") 

            # 3. 최종 답변 저장
            save_chat_history(user_id, session_id, question, collected_answer)

        except Exception as e:
            print(f"!!!!!!!!!!!!!! Agent Stream Error !!!!!!!!!!!!!!")
            traceback.print_exc()
            error_message = json.dumps({"error": "답변 생성 중 오류가 발생했습니다."})
            yield f"data: {error_message}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# app/main.py의 @app.post("/ask") 함수 전체를 이걸로 교체하세요.

@app.post("/ask")
async def ask(req: QuestionRequest):
    try:
        history_messages = get_recent_history(req.session_id, n=5)
        question = req.question
        user_id = req.user_id
        session_id = req.session_id

        print(f"[Agent Handling] '/ask' route for: '{question}'")

        # 2. 에이전트 비동기 실행 (invoke 대신 ainvoke 사용)
        response = await agent_executor.ainvoke({
            "input": question,
            "chat_history": history_messages
        })
        
        answer = response.get("output", "답변 생성에 실패했습니다.")
        
        save_chat_history(user_id, session_id, question, answer)
        return JSONResponse(content={"answer": answer})

    except Exception as e:
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
        
# main.py 파일에 추가

@app.delete("/delete_session/{session_id}")
async def delete_session(session_id: str, user_id: str):
    """
    특정 세션의 대화 기록을 삭제합니다.
    소유권 확인을 위해 user_id가 필요합니다.
    """
    if not r:
        return JSONResponse(status_code=503, content={"error": "Redis service is unavailable"})

    try:
        # 1. 소유권 확인
        session_list_key = f"user:{user_id}:sessions_sorted"
        if r.zscore(session_list_key, session_id) is None:
            return JSONResponse(status_code=403, content={"error": "Forbidden"})

        # 2. 세션 목록에서 해당 session_id 제거
        r.zrem(session_list_key, session_id)

        # 3. 실제 대화 내용 데이터 삭제
        chat_key = f"chat:{session_id}"
        r.delete(chat_key)

        print(f"✅ 세션 삭제 성공: user='{user_id}', session='{session_id}'")
        return JSONResponse(status_code=200, content={"message": "Session deleted successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# app/main.py 파일에 추가

@app.get("/sessions/latest/{user_id}")
async def get_latest_session_history(user_id: str):
    """
    특정 사용자의 가장 최근 대화 세션 기록 전체를 반환합니다.
    """
    if not r:
        return JSONResponse(status_code=503, content={"error": "Redis service is unavailable"})

    try:
        # 1. 해당 사용자의 가장 최근 session_id를 1개만 가져옵니다.
        session_list_key = f"user:{user_id}:sessions_sorted"
        latest_session_ids = r.zrevrange(session_list_key, 0, 0)
        
        if not latest_session_ids:
            # 이 사용자의 대화 기록이 아예 없는 경우
            return JSONResponse(content={"session_id": None, "messages": []})

        latest_session_id = latest_session_ids[0]

        # 2. 해당 session_id의 전체 대화 기록을 가져옵니다.
        chat_key = f"chat:{latest_session_id}"
        logs_raw = r.lrange(chat_key, 0, -1)

        messages = []
        for item_raw in logs_raw:
            item = json.loads(item_raw)
            question, answer = item.get("question"), item.get("answer")
            if question: messages.append({"role": "user", "text": question})
            if answer: messages.append({"role": "assistant", "text": answer})

        # 3. 프론트엔드가 대화를 이어갈 수 있도록 session_id와 메시지 목록을 함께 반환합니다.
        return JSONResponse(content={
            "session_id": latest_session_id,
            "messages": messages
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    


@app.get("/debug-db-files")
def debug_db_files():
    """
    서버에 배포된 vector_store 폴더의 실제 파일 목록과 크기를 확인하는 진단용 엔드포인트.
    """
    base_path = os.path.join(os.path.dirname(__file__), '..', 'vector_store')
    file_info = {}
    
    if not os.path.exists(base_path):
        return {"error": f"vector_store 폴더를 찾을 수 없습니다: {base_path}"}
        
    for root, dirs, files in os.walk(base_path):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                # 파일 경로에서 /app/ 부분을 기준으로 상대 경로를 만듭니다.
                relative_path = os.path.relpath(file_path, start=os.path.join(os.path.dirname(__file__), '..'))
                file_size_bytes = os.path.getsize(file_path)
                file_info[relative_path] = f"{file_size_bytes} bytes"
            except Exception as e:
                file_info[file_path] = f"크기 확인 오류: {e}"
                
    return file_info