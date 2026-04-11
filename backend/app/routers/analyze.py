"""
POST /api/analyze
Runs script detection and emoji analysis on raw text WITHOUT running the model.
Used by the frontend preprocessing details panel — matches the Streamlit app's
get_script_info() and get_emoji_info() calls shown in the preprocessing expander.
"""

import asyncio
from functools import partial
from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import AnalyzeRequest, AnalyzeResponse
from backend.app.services.model_service import ModelService

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Script detection and emoji analysis (no model inference)",
)
async def analyze_text(req: AnalyzeRequest):
    """
    Lightweight endpoint: runs only the preprocessing pipeline to return
    script_info and emoji_info for a given text.
    Does not run the transformer model — fast even before model loads.
    """
    if not ModelService.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Retry in a few seconds.",
        )

    loop = asyncio.get_running_loop()

    script_info = await loop.run_in_executor(
        None, partial(ModelService.get_script_info, req.text)
    )
    emoji_info = await loop.run_in_executor(
        None, partial(ModelService.get_emoji_info, req.text)
    )

    return {
        "script_info": script_info,
        "emoji_info": emoji_info,
    }
