FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

# 拷贝 backend 文件夹整个目录结构
COPY backend /app/backend
COPY backend/requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
