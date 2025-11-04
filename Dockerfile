FROM python:3.12-slim
WORKDIR /app
COPY uv.lock .
RUN pip install uv
RUN uv run
COPY . .
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]