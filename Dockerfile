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

# requirements.txt를 먼저 복사하여 pip install을 실행 (캐시 활용)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# vector_store 폴더를 복사
COPY ./vector_store /app/vector_store

# 소스 코드가 담긴 app 폴더를 복사
COPY ./app /app

# CMD ["uvicorn", "main:app", ...] (X)
# uvicorn이 실행할 파일의 위치가 app.main:app으로 변경됨
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]