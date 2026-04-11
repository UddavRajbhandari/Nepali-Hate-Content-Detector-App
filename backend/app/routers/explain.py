"""
Explainability endpoints — LIME, SHAP, Captum Integrated Gradients.

Each method has its own route so the frontend can call them independently
and show per-method loading states without blocking other requests.

All three methods operate on the *preprocessed* text (unified Devanagari)
rather than the raw input. This is the critical fix for the consistency bug
in the original Streamlit app where LIME was perturbing the original mixed-
script input while the model only ever saw the preprocessed version, causing
token labels to misalign with the actual model input.
"""

import asyncio
from functools import partial
from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import ExplainRequest, ExplainResponse
from backend.app.services.model_service import ModelService

router = APIRouter()


def _check_ready():
    if not ModelService.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Retry in a few seconds.",
        )


async def _run_in_thread(fn, *args):
    """
    Offload a CPU-bound / blocking call to a thread pool so we never block
    the uvicorn event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

@router.post(
    "/explain/lime",
    response_model=ExplainResponse,
    summary="LIME token-level explanation",
    response_description="Word importance scores computed via local linear approximation",
)
async def explain_lime(req: ExplainRequest):
    _check_ready()
    try:
        result = await _run_in_thread(
            ModelService.explain_lime, req.text, req.num_samples
        )
        return result
    except ImportError:
        raise HTTPException(
            status_code=422,
            detail="LIME is not installed. Add 'lime' to requirements.txt and rebuild.",
        )
    except MemoryError:
        raise HTTPException(
            status_code=503,
            detail="Out of memory running LIME. Reduce num_samples or text length.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

@router.post(
    "/explain/shap",
    response_model=ExplainResponse,
    summary="SHAP token-level explanation",
    response_description="Shapley value attributions for each token",
)
async def explain_shap(req: ExplainRequest):
    _check_ready()
    try:
        result = await _run_in_thread(ModelService.explain_shap, req.text)
        return result
    except ImportError:
        raise HTTPException(
            status_code=422,
            detail="SHAP is not installed. Add 'shap' to requirements.txt and rebuild.",
        )
    except MemoryError:
        raise HTTPException(
            status_code=503,
            detail="Out of memory running SHAP. Try a shorter text.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Captum Integrated Gradients
# ---------------------------------------------------------------------------

@router.post(
    "/explain/captum",
    response_model=ExplainResponse,
    summary="Captum Integrated Gradients explanation",
    response_description="Subword token attributions via gradient integration",
)
async def explain_captum(req: ExplainRequest):
    _check_ready()
    try:
        result = await _run_in_thread(
            ModelService.explain_captum, req.text, req.n_steps
        )
        return result
    except ImportError:
        raise HTTPException(
            status_code=422,
            detail="Captum is not installed. Add 'captum' to requirements.txt and rebuild.",
        )
    except MemoryError:
        raise HTTPException(
            status_code=422,
            detail=(
                "Out of memory running Captum Integrated Gradients. "
                "Use LIME or SHAP instead on cloud deployments with limited memory."
            ),
        )
    except RuntimeError as e:
        err = str(e).lower()
        if "memory" in err or "cuda out of memory" in err:
            raise HTTPException(
                status_code=422,
                detail="GPU/CPU memory exhausted. Reduce n_steps or switch to LIME/SHAP.",
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
