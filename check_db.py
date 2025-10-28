import pickle
import os

# 확인할 파일 경로 (방금 올리신 'docs.pkl' 파일입니다)
FILE_TO_CHECK = "vector_store/jobs/jobs_openai_docs.pkl"

print(f"--- '{FILE_TO_CHECK}' 파일 내용 확인 시작 ---")

# 1. 파일이 존재하는지 확인
if not os.path.exists(FILE_TO_CHECK):
    print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인하세요.")
else:
    try:
        # 2. pickle 파일 열기
        with open(FILE_TO_CHECK, "rb") as f:
            # 3. 파일 내용 로드 (이것이 원본 문서 리스트입니다)
            loaded_documents = pickle.load(f)

        print(f"✅ 총 {len(loaded_documents)}개의 문서를 찾았습니다.")
        
        # 4. 상위 5개 문서의 내용과 출처(metadata)를 출력
        for i, doc in enumerate(loaded_documents[:5]):
            print("\n---------------------------------")
            print(f"📄 문서 #{i+1}")
            print("---------------------------------")
            
            # doc 객체의 page_content (본문)와 metadata (출처) 속성 출력
            if hasattr(doc, 'page_content'):
                print(f"   [내용]: {doc.page_content[:300]}...") # 너무 길 수 있으니 300자만 출력
            
            if hasattr(doc, 'metadata'):
                print(f"   [출처]: {doc.metadata}")

        print("\n--- 확인 완료 ---")
        if len(loaded_documents) > 5:
            print(f" (전체 {len(loaded_documents)}개 중 5개만 표시했습니다)")

    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        print("이 오류는 보통 LangChain 관련 라이브러리가 설치되지 않았을 때 발생합니다.")
        print("터미널에서 '.\venv\Scripts\activate' 실행 후 다시 시도해 보세요.")