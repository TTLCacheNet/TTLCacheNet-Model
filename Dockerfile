# ---- 1. Python 기반 이미지 선택 ----
FROM python:3.12-slim

# ---- 2. 작업 디렉토리 설정 ----
WORKDIR /app

# ---- 3. uv 설치 ----
RUN pip install --no-cache-dir uv

# ---- 4. 의존성 파일 복사 ----
COPY uv.lock pyproject.toml ./

# ---- 5. 의존성 설치 ----
RUN uv sync --frozen

# ---------- Copy Model Server ----------
COPY app ./app
COPY artifacts ./artifacts

# ---- 7. 포트 설정 및 실행 명령 ----
EXPOSE 8001
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]