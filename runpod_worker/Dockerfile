FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY runpod_worker/requirements.txt /app/runpod_worker/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /app/runpod_worker/requirements.txt

COPY src /app/src
COPY external/VideoCAD/model /app/external/VideoCAD/model
COPY external/VideoCAD/model_configs /app/external/VideoCAD/model_configs
COPY runpod_worker /app/runpod_worker

ENV PYTHONPATH=/app/src:/app/external/VideoCAD
CMD ["python", "-u", "/app/runpod_worker/handler.py"]

