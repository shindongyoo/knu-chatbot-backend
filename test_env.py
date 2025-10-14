# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ 성공: .env 파일에서 API 키를 찾았습니다.")
    print(f"   내 키는 '{api_key[:5]}...' 로 시작합니다.")
else:
    print("❌ 실패: .env 파일에서 API 키를 찾을 수 없습니다!")
    print(f"   현재 작업 폴더: {os.getcwd()}")