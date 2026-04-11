"""
Nepali Hate Content Detection API
FastAPI backend — loads model once, serves predictions, explainability, history.

Run from major_project/ root:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.app.routers import predict, explain, history, batch, status, analyze
from backend.app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    await ModelService.initialize()
    print("Model ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Nepali Hate Content Detection API",
    description="XLM-RoBERTa-large fine-tuned for Nepali hate speech classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Routers
app.include_router(predict.router,  prefix="/api", tags=["predict"])
app.include_router(explain.router,  prefix="/api", tags=["explain"])
app.include_router(history.router,  prefix="/api", tags=["history"])
app.include_router(batch.router,    prefix="/api", tags=["batch"])
app.include_router(status.router,   prefix="/api", tags=["status"])
app.include_router(analyze.router,  prefix="/api", tags=["analyze"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": ModelService.is_ready(),
        "device": ModelService.get_device(),
    }