"""
Batch prediction endpoint.

POST /api/batch
  Accepts a list of texts (max 200) and streams results back as
  newline-delimited JSON (NDJSON) so the frontend can update a live
  progress bar without waiting for the entire batch to complete.

Wire format — one JSON object per line:
  Progress line:  {"index": 0, "total": 5, "result": { ... }}
  Final line:     {"done": true, "total": 5}
"""

import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.models.schemas import BatchRequest
from backend.app.services.model_service import ModelService

router = APIRouter()

_MAX_BATCH = 200


@router.post(
    "/batch",
    summary="Batch classify multiple texts (streaming)",
    response_description="NDJSON stream of per-text predictions with a final completion line",
)
async def batch_predict(req: BatchRequest):
    """
    Streams predictions one at a time so the frontend receives results
    incrementally. Each text is preprocessed and inferred independently —
    an error on one text does not abort the rest.
    """
    if not ModelService.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Retry in a few seconds.",
        )

    texts = req.texts
    if len(texts) > _MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(texts)} exceeds the maximum of {_MAX_BATCH}.",
        )

    total = len(texts)

    async def _stream():
        loop = asyncio.get_running_loop()

        for i, text in enumerate(texts):
            try:
                result = await loop.run_in_executor(
                    None, ModelService.predict, text
                )
                payload = {
                    "index": i,
                    "total": total,
                    "result": {
                        "text": text[:80] + ("..." if len(text) > 80 else ""),
                        "full_text": text,
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "preprocessed_text": result["preprocessed_text"],
                    },
                }
            except Exception as e:
                payload = {
                    "index": i,
                    "total": total,
                    "result": {
                        "text": text[:80] + ("..." if len(text) > 80 else ""),
                        "full_text": text,
                        "prediction": "Error",
                        "confidence": 0.0,
                        "preprocessed_text": "",
                        "error": str(e),
                    },
                }

            yield json.dumps(payload, ensure_ascii=False) + "\n"

        yield json.dumps({"done": True, "total": total}) + "\n"

    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
