import os
import certifi
import redis
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# --- MongoDB 설정 ---
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
chatbot_db = mongo_client.chatbot_database
print("✅ MongoDB 클라이언트 초기화 완료.")


# --- Redis 설정 ---
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True
    )
    r.ping()
    print("✅ Redis 연결 성공.")
except Exception as e:
    print(f"❌ Redis 연결 실패: {e}")
    r = None