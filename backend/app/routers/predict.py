"""
Single text prediction endpoint.

POST /api/predict
  - Preprocesses input text (script detection, transliteration, emoji mapping)
  - Runs inference with XLM-RoBERTa-large
  - Optionally saves result to history (background task, non-blocking)
  - Returns prediction, confidence, per-class probabilities, and preprocessing details
"""

import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.app.models.schemas import PredictRequest, PredictResponse
from backend.app.services.model_service import ModelService
from backend.app.utils.history import append_history

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single Nepali text",
    response_description="Prediction label, confidence, per-class probabilities, and preprocessing info",
)
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """
    Accepts Devanagari, Romanized Nepali, English, or mixed-script text.
    The preprocessing pipeline (script detection → transliteration → translation
    → emoji mapping → text cleaning) runs before inference so the model always
    receives clean, unified Devanagari input.
    """
    if not ModelService.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Retry in a few seconds.",
        )

    if not req.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Text must not be empty or whitespace only.",
        )

    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(None, ModelService.predict, req.text)
    except MemoryError:
        raise HTTPException(
            status_code=503,
            detail="Out of memory during inference. Try a shorter text.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.save_to_history:
        background_tasks.add_task(append_history, result)

    return result
