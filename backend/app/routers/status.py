"""
GET /api/status
Returns which optional XAI packages are installed and whether the model is ready.
Mirrors check_explainability() and check_captum_availability() from the Streamlit app.
The React frontend calls this once on load to know which explain buttons to show.
"""

from fastapi import APIRouter
from backend.app.models.schemas import StatusResponse
from backend.app.services.model_service import ModelService

router = APIRouter()


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Model and XAI capability status",
)
async def get_status():
    """
    Returns availability of LIME, SHAP, Captum, and the preprocessing pipeline.
    Safe to call before the model finishes loading.
    """
    return ModelService.get_capabilities()
