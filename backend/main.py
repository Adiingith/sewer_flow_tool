from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import get_settings
from backend.api.v1 import functionApi
from backend.api.v1 import checkApi
from backend.api.v1 import monitorApi
from backend.api.v1 import responsibilityApi
from backend.api.v1 import aiPredictApi
from backend.api.v1 import monitorAnalysisApi

settings = get_settings()

UPLOAD_DIR = settings.UPLOAD_DIR
app = FastAPI(title=settings.PROJECT_NAME)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the source
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(functionApi.router, prefix=settings.API_V1_STR, tags=["functionApi"])
app.include_router(checkApi.router, prefix=settings.API_V1_STR, tags=["checkApi"])
app.include_router(monitorApi.router, prefix=settings.API_V1_STR, tags=["monitorApi"])
app.include_router(responsibilityApi.router, prefix=settings.API_V1_STR, tags=["responsibilityApi"])
app.include_router(aiPredictApi.router, prefix=settings.API_V1_STR, tags=["aiPredictApi"])
app.include_router(monitorAnalysisApi.router, prefix=settings.API_V1_STR, tags=["monitorAnalysisApi"])

