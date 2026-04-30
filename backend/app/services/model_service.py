"""
ModelService — singleton that holds model, tokenizer, preprocessor.

UPDATED FOR PRODUCTION (Railway-safe):
- Adds timeout protection during model load
- Adds HF cache control for faster cold starts
- Adds safe fallback model handling (env-driven)
- Adds initialization guard (prevents double load)
- Adds better logging for debugging deployment issues
- Preserves ALL existing functionality (no regressions)
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder


class ModelService:
    _model = None
    _tokenizer = None
    _label_encoder: Optional[LabelEncoder] = None
    _preprocessor = None
    _device: Optional[torch.device] = None
    _ready = False

    # Capability flags
    _lime_available: bool = False
    _shap_available: bool = False
    _captum_available: bool = False

    _emoji_map: Dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    @classmethod
    async def initialize(cls):
        """
        Async initialization with timeout protection.
        Prevents container hanging indefinitely during model load.
        """
        import asyncio

        if cls._ready:
            return

        loop = asyncio.get_event_loop()

        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, cls._load_sync),
                timeout=int(os.getenv("MODEL_LOAD_TIMEOUT", "300")),  # 5 min default
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Model loading timed out.")

    @classmethod
    def _load_sync(cls):
        """Synchronous model loading (runs in executor thread)."""

        if cls._ready:
            return

        print("[ModelService] Starting model initialization...")

        # ✅ HF cache control (IMPORTANT for Railway performance)
        os.environ["HF_HOME"] = os.getenv("HF_HOME", "/app/cache")
        os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/app/cache")

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import joblib

        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Environment-driven configuration
        local_path = os.getenv(
            "MODEL_PATH",
            "models/saved_models/xlm_roberta_results/large_final",
        )

        hf_id = os.getenv(
            "HF_MODEL_ID",
            "UDHOV/xlm-roberta-base-nepali-hate-classification",  # safer default
        )

        print(f"[ModelService] Loading model from: {hf_id}")

        # Label encoder default
        le = LabelEncoder()
        le.fit(["NO", "OO", "OR", "OS"])

        # Load model
        if os.path.exists(local_path):
            print("[ModelService] Loading local model...")
            cls._tokenizer = AutoTokenizer.from_pretrained(local_path)
            cls._model = AutoModelForSequenceClassification.from_pretrained(local_path)

            le_path = os.path.join(local_path, "label_encoder.pkl")
            if os.path.exists(le_path):
                le = joblib.load(le_path)
        else:
            print("[ModelService] Loading model from HuggingFace...")
            cls._tokenizer = AutoTokenizer.from_pretrained(hf_id)
            cls._model = AutoModelForSequenceClassification.from_pretrained(hf_id)

            try:
                from huggingface_hub import hf_hub_download
                le_file = hf_hub_download(repo_id=hf_id, filename="label_encoder.pkl")
                le = joblib.load(le_file)
            except Exception as e:
                print(f"[ModelService] Label encoder not found on hub: {e}")

        cls._model.to(cls._device).eval()
        cls._label_encoder = le

        # Preprocessor
        try:
            from scripts.transformer_data_preprocessing import (
                HateSpeechPreprocessor,
                EMOJI_TO_NEPALI,
            )

            cls._preprocessor = HateSpeechPreprocessor(
                model_type="xlmr",
                translate_english=True,
                cache_size=2000,
            )
            cls._emoji_map = EMOJI_TO_NEPALI

        except ImportError as e:
            print(f"[WARNING] Preprocessor not loaded: {e}")
            cls._preprocessor = None
            cls._emoji_map = {}

        # Capability detection
        cls._lime_available = cls._check_import("lime")
        cls._shap_available = cls._check_import("shap")
        cls._captum_available = cls._check_import("captum")

        cls._ready = True
        print("[ModelService] Model initialized successfully.")

    @staticmethod
    def _check_import(pkg: str) -> bool:
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    @classmethod
    def is_ready(cls) -> bool:
        return cls._ready

    @classmethod
    def get_device(cls) -> str:
        return str(cls._device) if cls._device else "unknown"

    @classmethod
    def get_capabilities(cls) -> Dict[str, Any]:
        return {
            "model_loaded": cls._ready,
            "device": cls.get_device(),
            "preprocessor": cls._preprocessor is not None,
            "lime": cls._lime_available,
            "shap": cls._shap_available,
            "captum": cls._captum_available,
        }

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    @classmethod
    def predict(cls, text: str, max_length: int = 256) -> Dict[str, Any]:
        """
        Main prediction method (unchanged behavior).
        """

        if not cls._ready:
            raise RuntimeError("Model not loaded yet.")

        if cls._device is None:
            cls._device = torch.device("cpu")

        if cls._preprocessor:
            preprocessed, emoji_features = cls._preprocessor.preprocess(text, verbose=False)
        else:
            preprocessed = text
            emoji_features = {}

        if not preprocessed.strip():
            return cls._empty_result(text, emoji_features)

        inputs = cls._tokenizer(
            preprocessed,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(cls._device)
        attention_mask = inputs["attention_mask"].to(cls._device)

        with torch.no_grad():
            logits = cls._model(input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = cls._label_encoder.classes_[pred_idx]

        return {
            "prediction": pred_label,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                cls._label_encoder.classes_[i]: float(probs[i])
                for i in range(len(cls._label_encoder.classes_))
            },
            "original_text": text,
            "preprocessed_text": preprocessed,
            "emoji_features": emoji_features,
            "script_info": cls.get_script_info(text),
        }

    # -------------------------------------------------------------------------
    # Script / Emoji helpers
    # -------------------------------------------------------------------------

    @classmethod
    def get_script_info(cls, text: str) -> Dict[str, Any]:
        if cls._preprocessor is None:
            return {"script_type": "unknown", "confidence": 0.0}
        try:
            from scripts.transformer_data_preprocessing import get_script_info
            info = get_script_info(text)
            info["confidence"] = min(float(info.get("confidence", 0.0)), 1.0)
            return info
        except Exception:
            return {"script_type": "unknown", "confidence": 0.0}

    @classmethod
    def get_emoji_info(cls, text: str) -> Dict[str, Any]:
        if cls._preprocessor is None:
            return {"emojis_found": [], "count": 0}
        try:
            from scripts.transformer_data_preprocessing import get_emoji_info
            return get_emoji_info(text)
        except Exception:
            return {"emojis_found": [], "count": 0}

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------

    @classmethod
    def _empty_result(cls, original: str, emoji_features: Dict) -> Dict[str, Any]:
        labels = list(cls._label_encoder.classes_)
        return {
            "prediction": "NO",
            "confidence": 0.0,
            "probabilities": {l: 0.0 for l in labels},
            "original_text": original,
            "preprocessed_text": "",
            "emoji_features": emoji_features,
            "script_info": {"script_type": "unknown", "confidence": 0.0},
            "error": "Empty text after preprocessing",
        }