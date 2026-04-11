"""
Pydantic v2 schemas — Python 3.8 compatible.
Uses Dict/List/Union from typing, Annotated from typing_extensions.
"""

from typing import Dict, List, Optional, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field, field_validator, model_validator

Label = str
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class ScriptInfo(BaseModel):
    """Script detection result — mirrors get_script_info() output."""
    script_type: str
    confidence: float


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    save_to_history: bool = Field(default=True)

    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text must not be empty or whitespace only.")
        return v


class PredictResponse(BaseModel):
    prediction: Label
    confidence: Confidence
    probabilities: Dict[str, Annotated[float, Field(ge=0.0, le=1.0)]]
    original_text: str
    preprocessed_text: str
    emoji_features: Dict[str, Union[int, float, bool]]
    script_info: Optional[ScriptInfo] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Analyze (script + emoji info without running full prediction)
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text must not be empty or whitespace only.")
        return v


class EmojiInfo(BaseModel):
    """Matches actual get_emoji_info() return dict from transformer_data_preprocessing.py"""
    emojis_found: List[str]
    total_count: int
    known_emojis: List[str]
    known_count: int
    unknown_emojis: List[str]
    unknown_count: int
    coverage: float


class AnalyzeResponse(BaseModel):
    """Preprocessing analysis without running the model — for the details panel."""
    script_info: ScriptInfo
    emoji_info: EmojiInfo


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to classify. Max 200 per request.")

    @model_validator(mode="after")
    def check_batch_size(self) -> "BatchRequest":
        if len(self.texts) == 0:
            raise ValueError("texts list must contain at least one item.")
        if len(self.texts) > 200:
            raise ValueError(f"Batch size {len(self.texts)} exceeds the maximum of 200.")
        self.texts = [t.strip() for t in self.texts if t.strip()]
        if not self.texts:
            raise ValueError("All texts were empty after stripping whitespace.")
        return self


class BatchResultItem(BaseModel):
    text: str
    full_text: str
    prediction: Label
    confidence: Confidence
    preprocessed_text: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    num_samples: int = Field(default=200, ge=50, le=1000)  # default 200 matches Streamlit
    n_steps: int = Field(default=50, ge=10, le=200)        # default 50 matches Streamlit slider

    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text must not be empty or whitespace only.")
        return v


class WordScore(BaseModel):
    word: str
    score: float


class ExplainResponse(BaseModel):
    method: str
    prediction: Label
    confidence: Confidence
    word_scores: List[WordScore]
    preprocessed_text: str
    convergence_delta: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Status / capabilities
# ---------------------------------------------------------------------------

class StatusResponse(BaseModel):
    """Returned by GET /api/status — tells frontend which XAI methods are available."""
    model_loaded: bool
    device: str
    preprocessor: bool
    lime: bool
    shap: bool
    captum: bool


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class HistoryItem(BaseModel):
    timestamp: str
    text: str
    prediction: Label
    confidence: Confidence
    probabilities: Dict[str, float]
    preprocessed_text: str
    emoji_features: Dict[str, Union[int, float, bool]]


class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int
    limit: int
    offset: int


class HistoryStatsResponse(BaseModel):
    total: int
    avg_confidence: Optional[float]
    class_counts: Dict[str, int]
    most_common_class: Optional[str]