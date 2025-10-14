# test_search.py

import os
from dotenv import load_dotenv

# 중요: .env 파일을 먼저 로드해서 API 키를 준비합니다.
# search_engine.py가 import될 때 OpenAIEmbeddings를 초기화하기 때문입니다.
print("[INFO] 테스트를 위해 .env 파일을 로드합니다...")
load_dotenv()

# 이제 우리가 테스트하려는 검색 함수를 가져옵니다.
from app.search_engine import search_similar_documents

def run_test():
    """하나의 질문으로 검색 기능을 테스트하는 함수"""

    # --- ▼▼▼ 테스트하고 싶은 질문을 여기에 입력하세요 ▼▼▼ ---
    test_query = "한세경 교수님 연구실 어디에요?"
    # ----------------------------------------------------

    print("-" * 50)
    print(f"테스트 질문: '{test_query}'")
    print("-" * 50)

    try:
        # 검색 함수를 직접 호출합니다.
        context, field_names = search_similar_documents(test_query)

        print("\n✅ 검색 성공! 아래는 검색된 내용입니다.\n")
        
        print("--- [검색 결과 (Context)] ---")
        print(context)
        print("--------------------------\n")

        print("--- [참고한 메타데이터 필드 종류] ---")
        print(field_names)
        print("------------------------------------")

    except Exception as e:
        print("\n❌ 테스트 중 오류가 발생했습니다.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()