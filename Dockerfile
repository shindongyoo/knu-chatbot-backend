FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-kor \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt를 먼저 복사하여 pip install을 실행
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# vector_store 폴더를 복사
COPY ./vector_store /app/vector_store

# app 폴더를 /app 폴더의 하위 폴더로 복사, 컨테이너 내부에 /app/app/main.py 구조
COPY ./app /app/app

# uvicorn 실행 명령어를 원래대로 되돌려 /app 폴더에서 app.main 모듈 find
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]