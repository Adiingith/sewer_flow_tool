from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
import random
from backend.schemas.refresh_product_view import start_scheduler
from backend.core.config import get_settings
from sqlalchemy.exc import SQLAlchemyError
from backend.core.db_connect import get_session
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from backend.api.v1 import functionApi
from backend.api.v1 import checkApi
from backend.api.v1 import monitorApi
from backend.api.v1 import responsibilityApi

settings = get_settings()

UPLOAD_DIR = settings.UPLOAD_DIR
app = FastAPI(title=settings.PROJECT_NAME)

# CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(functionApi.router, prefix=settings.API_V1_STR, tags=["functionApi"])
app.include_router(checkApi.router, prefix=settings.API_V1_STR, tags=["checkApi"])
app.include_router(monitorApi.router, prefix=settings.API_V1_STR, tags=["monitorApi"])
app.include_router(responsibilityApi.router, prefix=settings.API_V1_STR, tags=["responsibilityApi"])

# 上传目录
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 数据库连接健康检查接口
@app.get("/health/db")
async def test_db_connection(session: AsyncSession = Depends(get_session)):
    try:
        await session.execute(text("SELECT 1"))
        return {"status": "ok"}
    except SQLAlchemyError as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

# 启动时加载任务调度器
@app.on_event("startup")
def startup_event():
    start_scheduler()

# 上传接口
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        filename = file.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            return {"error": f"Unsupported file type: {ext}"}
        path = os.path.join(UPLOAD_DIR, filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(filename)
    return {"message": "Files uploaded successfully", "files": saved_files}

# 删除接口
@app.post("/delete")
async def delete_file(filename: str = Form(...)):
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return {"message": f"{filename} deleted"}
    return {"error": f"{filename} not found"}

# 替换接口
@app.post("/replace")
async def replace_file(
    old_filename: str = Form(...),
    new_file: UploadFile = File(...)
):
    old_path = os.path.join(UPLOAD_DIR, old_filename)
    if os.path.exists(old_path):
        os.remove(old_path)

    filename = new_file.filename or "uploaded_file"
    new_path = os.path.join(UPLOAD_DIR, filename)
    with open(new_path, "wb") as f:
        shutil.copyfileobj(new_file.file, f)

    return {"message": "File replaced", "new_file": filename}

# 打分接口
@app.get("/evaluate")
async def evaluate_files():
    result = []
    for fname in os.listdir(UPLOAD_DIR):
        if fname.endswith(".csv") or fname.endswith(".xlsx"):
            score = round(random.uniform(0, 1), 3)
            result.append({"file": fname, "score": score})
    return {"evaluations": result}
