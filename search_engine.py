# search_engine.py

import os
import time
import openai
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# FAISS 인덱스 & 메타데이터 로드
print("📚 FAISS 인덱스 로드 중...")
index = faiss.read_index("notices.index")
with open("notices_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# 임베딩 생성 함수
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    time.sleep(0.5)
    return response.data[0].embedding

# 공지사항 검색 함수
def search_notices(query, top_k=3):
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    results = [metadata[idx] for idx in indices[0]]
    return distances, indices, results

# FastAPI용 문맥 추출 함수
def search_similar_documents(query, top_k=10):
    _, _, results = search_notices(query, top_k=top_k)
    context = ""
    field_names = set()
    for doc in results:
        context += "- 문서 정보:\n"
        for key, value in doc.items():
            if value:
                context += f"  {key}: {value}\n"
                field_names.add(key)
        context += "|\n"
    return context, field_names

# 요약 함수 (선택적 사용)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_llm_summary(query, results):
    """검색 결과를 LLM이 맥락을 고려해 요약하고 재정렬합니다."""
    
    # 최대 길이 제한 설정 (characters 기준, 원하는 대로 조정 가능)
    MAX_CONTENT_LEN = 1000
    MAX_TABLE_LEN = 1500

    # results_text 생성 (본문과 표 별도 길이 제한 적용)
    results_text = "\n\n".join([
        f"[{i+1}] 제목: {r['title']}\n"
        f"날짜: {r['date']}\n"
        f"내용: "
        + (
            # 본문 content인 경우
            (r.get('content', '')[:MAX_CONTENT_LEN] + ' ...(생략됨)' if len(r.get('content', '')) > MAX_CONTENT_LEN else r.get('content', ''))
            if r['type'] == 'content'
            # 표 table_content인 경우
            else (r.get('table_content', '')[:MAX_TABLE_LEN] + ' ...(생략됨)' if len(r.get('table_content', '')) > MAX_TABLE_LEN else r.get('table_content', ''))
        )
        for i, r in enumerate(results)
    ])

    # 프롬프트 구성
    prompt = f"""다음은 사용자의 질문과 관련된 공지사항 검색 결과입니다.
        사용자의 질문에 가장 관련성 높은 순서대로 결과를 재정렬하고, 각 결과가 왜 관련있는지 간단히 설명해주세요.

        사용자 질문: {query}

        검색 결과:
        {results_text}

        다음 형식으로 응답해주세요:
        1. [재정렬된 순위] 제목: (제목)
        - 관련성 설명: (이 결과가 왜 관련있는지 설명)
        2. [재정렬된 순위] 제목: (제목)
        - 관련성 설명: (이 결과가 왜 관련있는지 설명)
        ...
        """

    # LLM 호출
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 필요한 경우 "gpt-4" 로 변경 가능
            messages=[{"role": "user", "content": prompt}]
        )
        time.sleep(1)  # Rate Limit 방지
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ LLM 요약 실패: {str(e)}")
        raise