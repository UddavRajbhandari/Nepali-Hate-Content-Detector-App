"""
ModelService — singleton that holds model, tokenizer, preprocessor.
Loaded once at startup via lifespan, shared across all requests.

Matches all features from the Streamlit app:
  - max_length=256 (same as Streamlit predict_text)
  - script_info included in predict response
  - get_script_info() and get_emoji_info() exposed as class methods
  - explainability availability check
  - Captum uses EMOJI_TO_NEPALI map when available
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

    # Cached availability flags — checked once at startup
    _lime_available: bool = False
    _shap_available: bool = False
    _captum_available: bool = False

    # EMOJI_TO_NEPALI from preprocessing script — used by Captum
    _emoji_map: Dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    @classmethod
    async def initialize(cls):
        """Load everything. Called once at startup."""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, cls._load_sync)

    @classmethod
    def _load_sync(cls):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import joblib

        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        local_path = os.getenv(
            "MODEL_PATH",
            "models/saved_models/xlm_roberta_results/large_final",
        )
        hf_id = os.getenv(
            "HF_MODEL_ID",
            "UDHOV/xlm-roberta-large-nepali-hate-classification",
        )

        # Label encoder — default labels, overridden if pkl exists
        le = LabelEncoder()
        le.fit(["NO", "OO", "OR", "OS"])

        # Try local first, fallback to HF Hub
        if os.path.exists(local_path):
            cls._tokenizer = AutoTokenizer.from_pretrained(local_path)
            cls._model = AutoModelForSequenceClassification.from_pretrained(local_path)
            le_path = os.path.join(local_path, "label_encoder.pkl")
            if os.path.exists(le_path):
                le = joblib.load(le_path)
        else:
            cls._tokenizer = AutoTokenizer.from_pretrained(hf_id)
            cls._model = AutoModelForSequenceClassification.from_pretrained(hf_id)
            try:
                from huggingface_hub import hf_hub_download
                le_file = hf_hub_download(repo_id=hf_id, filename="label_encoder.pkl")
                le = joblib.load(le_file)
            except Exception:
                pass

        cls._model.to(cls._device).eval()
        cls._label_encoder = le

        # Preprocessor + emoji map
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
        except ImportError:
            cls._preprocessor = None
            cls._emoji_map = {}

        # Check optional XAI package availability once
        try:
            import lime  # noqa: F401
            cls._lime_available = True
        except ImportError:
            cls._lime_available = False

        try:
            import shap  # noqa: F401
            cls._shap_available = True
        except ImportError:
            cls._shap_available = False

        try:
            import captum  # noqa: F401
            cls._captum_available = True
        except ImportError:
            cls._captum_available = False

        cls._ready = True

    # -------------------------------------------------------------------------
    # Status / capability helpers
    # -------------------------------------------------------------------------

    @classmethod
    def is_ready(cls) -> bool:
        return cls._ready

    @classmethod
    def get_device(cls) -> str:
        return str(cls._device) if cls._device else "unknown"

    @classmethod
    def get_capabilities(cls) -> Dict[str, Any]:
        """Return availability of optional XAI packages and preprocessor."""
        return {
            "model_loaded": cls._ready,
            "device": cls.get_device(),
            "preprocessor": cls._preprocessor is not None,
            "lime": cls._lime_available,
            "shap": cls._shap_available,
            "captum": cls._captum_available,
        }

    # -------------------------------------------------------------------------
    # Preprocessing info helpers  (mirrors Streamlit get_script_info / get_emoji_info)
    # -------------------------------------------------------------------------

    @classmethod
    def get_script_info(cls, text: str) -> Dict[str, Any]:
        """
        Return script detection info for a raw text string.
        Mirrors the Streamlit app's get_script_info() call.
        Returns a safe default if the preprocessor is not loaded.
        """
        if cls._preprocessor is None:
            return {"script_type": "unknown", "confidence": 0.0}
        try:
            from scripts.transformer_data_preprocessing import get_script_info
            info = get_script_info(text)
            # Cap confidence at 1.0 — Streamlit does min(..., 100%) / 100
            info["confidence"] = min(float(info.get("confidence", 0.0)), 1.0)
            return info
        except Exception:
            return {"script_type": "unknown", "confidence": 0.0}

    @classmethod
    def get_emoji_info(cls, text: str) -> Dict[str, Any]:
        """
        Return emoji analysis for a raw text string.
        Mirrors the Streamlit app's get_emoji_info() call.
        """
        if cls._preprocessor is None:
            return {"emojis_found": [], "count": 0}
        try:
            from scripts.transformer_data_preprocessing import get_emoji_info
            return get_emoji_info(text)
        except Exception:
            return {"emojis_found": [], "count": 0}

    # -------------------------------------------------------------------------
    # Core prediction  (max_length=256 matches Streamlit predict_text)
    # -------------------------------------------------------------------------

    @classmethod
    def predict(cls, text: str, max_length: int = 256) -> Dict[str, Any]:
        """
        Preprocess → tokenize → infer → return structured result.
        Includes script_info so the frontend can show the preprocessing details tab.
        max_length=256 matches the Streamlit app.
        """
        if not cls._ready:
            raise RuntimeError("Model not loaded yet.")

        # Preprocessing
        if cls._preprocessor:
            preprocessed, emoji_features = cls._preprocessor.preprocess(text, verbose=False)
        else:
            preprocessed = text
            emoji_features = {}

        if not preprocessed.strip():
            return cls._empty_result(text, emoji_features)

        # Script info (for preprocessing details panel in frontend)
        script_info = cls.get_script_info(text)

        # Tokenize
        inputs = cls._tokenizer(
            preprocessed,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(cls._device)
        attention_mask = inputs["attention_mask"].to(cls._device)

        # Inference
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
            "script_info": script_info,
        }

    # -------------------------------------------------------------------------
    # Explainability
    # -------------------------------------------------------------------------

    @classmethod
    def _get_preprocessed(cls, text: str):
        """Helper: preprocess text, return (preprocessed, emoji_features)."""
        if cls._preprocessor:
            return cls._preprocessor.preprocess(text)
        return text, {}

    @classmethod
    def _predict_fn_for_xai(cls, texts, max_length: int = 128):
        """
        Shared predict function for LIME / SHAP perturbations.
        Returns numpy array of shape (n, num_classes).
        """
        results = []
        for t in texts:
            inputs = cls._tokenizer(
                t, return_tensors="pt",
                max_length=max_length, padding="max_length", truncation=True,
            )
            iids = inputs["input_ids"].to(cls._device)
            mask = inputs["attention_mask"].to(cls._device)
            with torch.no_grad():
                logits = cls._model(iids, attention_mask=mask).logits
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            results.append(p)
        return np.array(results)


    @classmethod
    def explain_lime(cls, text: str, num_samples: int = 200) -> Dict[str, Any]:
        """
        LIME explanation matching explainability.py LIMEExplainer.explain().

        Key differences from naive approach:
        - predict_proba() receives a list of strings (LIME perturbs preprocessed text)
        - ModelExplainerWrapper.predict_proba() handles ndarray→list conversion
        - We replicate that ndarray handling here directly
        - No split_expression override: default W+ works because LIME receives
          the preprocessed Devanagari string and we handle the token mapping ourselves
        """
        from lime.lime_text import LimeTextExplainer
        if not cls._ready:
            raise RuntimeError("Model not loaded.")

        preprocessed, _ = cls._get_preprocessed(text)
        labels = list(cls._label_encoder.classes_)
        pred_result = cls.predict(text)
        pred_idx = labels.index(pred_result["prediction"])

        def _predict_proba(texts):
            """
            Mirrors ModelExplainerWrapper.predict_proba().
            CRITICAL: must return exactly len(texts) rows — LIME checks this strictly.
            Empty strings (all-masked perturbations) must NOT be filtered out;
            instead we return uniform probabilities for them in-place.
            """
            if isinstance(texts, np.ndarray):
                texts = texts.tolist() if texts.ndim > 0 else [str(texts)]

            n = len(texts)
            uniform = np.ones(len(labels)) / len(labels)
            results = np.zeros((n, len(labels)))

            # Identify non-empty texts and their original indices
            non_empty = [(i, str(t).strip()) for i, t in enumerate(texts) if str(t).strip()]

            if not non_empty:
                # All texts are empty — return uniform for all
                for i in range(n):
                    results[i] = uniform
                return results

            indices, valid_texts = zip(*non_empty)

            enc = cls._tokenizer(
                list(valid_texts),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(cls._device)
            attention_mask = enc["attention_mask"].to(cls._device)
            with torch.no_grad():
                probs = torch.softmax(
                    cls._model(input_ids=input_ids, attention_mask=attention_mask).logits,
                    dim=-1,
                ).cpu().numpy()

            # Place predictions at correct indices; fill empty slots with uniform
            for i in range(n):
                results[i] = uniform
            for out_idx, orig_idx in enumerate(indices):
                results[orig_idx] = probs[out_idx]

            return results

        # split_expression=r'\s+' keeps full Devanagari words intact
        # (default \\W+ splits on every non-ASCII char, fragmenting Devanagari)
        #
        # num_features must not exceed the actual word count — LIME's internal
        # ridge regression will get mismatched sample counts if num_features > n_words,
        # causing "inconsistent numbers of samples" ValueError.
        n_words = len(preprocessed.split())
        num_features = min(15, n_words) if n_words > 0 else 1

        # num_samples must be strictly greater than num_features for ridge to work.
        # Ensure at least num_features * 10 samples, capped at the caller's request.
        safe_num_samples = max(num_samples, num_features * 10 + 1)

        explainer = LimeTextExplainer(
            class_names=labels,
            split_expression=r"\s+",
            random_state=42,
        )
        exp = explainer.explain_instance(
            preprocessed,
            _predict_proba,
            num_features=num_features,
            num_samples=safe_num_samples,
            labels=[pred_idx],
        )

        word_scores = exp.as_list(label=pred_idx)
        return {
            "method": "LIME",
            "prediction": pred_result["prediction"],
            "confidence": pred_result["confidence"],
            "word_scores": [{"word": w, "score": float(s)} for w, s in word_scores],
            "preprocessed_text": preprocessed,
            "convergence_delta": None,
            "error": None,
        }

    @classmethod
    def explain_shap(cls, text: str) -> Dict[str, Any]:
        """
        SHAP explanation matching explainability.py SHAPExplainer.explain().

        Fallback matches _gradient_based_attribution() (occlusion / leave-one-out),
        NOT KernelExplainer. This is what the Streamlit app actually does.
        predict_masked handles both str and ndarray inputs as in the original.
        """
        import shap
        if not cls._ready:
            raise RuntimeError("Model not loaded.")

        preprocessed, _ = cls._get_preprocessed(text)
        labels = list(cls._label_encoder.classes_)
        pred_result = cls.predict(text)
        pred_idx = labels.index(pred_result["prediction"])
        tokens = preprocessed.split()

        if not tokens:
            return {
                "method": "SHAP",
                "word_scores": [],
                "prediction": pred_result["prediction"],
                "confidence": pred_result["confidence"],
                "preprocessed_text": preprocessed,
                "convergence_delta": None,
                "error": None,
            }

        def _predict_masked(masked_texts):
            """
            Mirrors SHAPExplainer.predict_masked():
            Handles str, list, and ndarray inputs identically to the original.
            """
            if isinstance(masked_texts, np.ndarray):
                if masked_texts.ndim == 1:
                    texts = [" ".join(str(t) for t in masked_texts if str(t).strip())]
                else:
                    texts = [" ".join(str(t) for t in row if str(t).strip())
                             for row in masked_texts]
            elif isinstance(masked_texts, str):
                texts = [masked_texts]
            elif isinstance(masked_texts, list):
                texts = masked_texts
            else:
                texts = [str(masked_texts)]

            texts = [t for t in texts if t.strip()]
            if not texts:
                return np.ones((1, len(labels))) / len(labels)

            enc = cls._tokenizer(
                texts, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            )
            iids = enc["input_ids"].to(cls._device)
            mask = enc["attention_mask"].to(cls._device)
            with torch.no_grad():
                probs = torch.softmax(
                    cls._model(input_ids=iids, attention_mask=mask).logits, dim=-1
                )
            return probs.cpu().numpy()

        method_used = "primary"
        try:
            # Primary: SHAP text masker
            explainer = shap.Explainer(_predict_masked, shap.maskers.Text(preprocessed))
            sv = explainer([preprocessed])[0]
            shap_tokens = list(sv.data)
            values_array = np.array(sv.values)

            if len(shap_tokens) == 0 or values_array.size == 0:
                raise ValueError("SHAP returned empty results")

            # Extract values for predicted class
            if values_array.ndim == 1:
                token_values = values_array
            elif values_array.ndim == 2:
                token_values = values_array[:, pred_idx]
            elif values_array.ndim == 3:
                token_values = values_array[0, :, pred_idx]
            else:
                token_values = values_array.flatten()[:len(shap_tokens)]

            word_scores = [
                {"word": str(w), "score": float(v)}
                for w, v in zip(shap_tokens, token_values)
            ]

        except Exception:
            # Fallback: occlusion / leave-one-out — matches _gradient_based_attribution()
            method_used = "gradient_fallback"
            base_probs = _predict_masked([preprocessed])[0]
            base_pred_score = float(base_probs[pred_idx])

            attributions = []
            for i in range(len(tokens)):
                masked_words = tokens[:i] + tokens[i + 1:]
                masked_text = " ".join(masked_words)
                if not masked_text.strip():
                    attributions.append(base_pred_score)
                    continue
                masked_probs = _predict_masked([masked_text])[0]
                # Attribution = score drop when word is removed
                attributions.append(base_pred_score - float(masked_probs[pred_idx]))

            word_scores = [
                {"word": tokens[i], "score": attributions[i]}
                for i in range(len(tokens))
            ]

        return {
            "method": "SHAP",
            "prediction": pred_result["prediction"],
            "confidence": pred_result["confidence"],
            "word_scores": sorted(word_scores, key=lambda x: abs(x["score"]), reverse=True),
            "preprocessed_text": preprocessed,
            "convergence_delta": None,
            "error": None if method_used == "primary" else f"Used {method_used}",
        }

    @classmethod
    def explain_captum(cls, text: str, n_steps: int = 50) -> Dict[str, Any]:
        """
        Captum explanation matching captum_explainer.py CaptumExplainer.explain().

        Key differences from naive approach:
        - Uses LayerIntegratedGradients on word_embeddings layer (not IntegratedGradients on embeddings)
        - Baseline is pad_token_id (not zeros)
        - forward_func takes (input_ids, attention_mask) directly
        - Aggregates subword attributions by summing abs values per word
        - Returns signed and abs scores per word, matching word_attributions tuples
        """
        try:
            from captum.attr import LayerIntegratedGradients
        except ImportError:
            raise RuntimeError("Captum not installed. Run: pip install captum")

        if not cls._ready:
            raise RuntimeError("Model not loaded.")

        preprocessed, _ = cls._get_preprocessed(text)
        labels = list(cls._label_encoder.classes_)
        pred_result = cls.predict(text)
        pred_idx = labels.index(pred_result["prediction"])

        encoding = cls._tokenizer(
            preprocessed,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        input_ids = encoding["input_ids"].to(cls._device)
        attention_mask = encoding["attention_mask"].to(cls._device)

        # Get embedding layer — matches CaptumExplainer._get_embedding_layer()
        if hasattr(cls._model, "roberta"):
            embedding_layer = cls._model.roberta.embeddings.word_embeddings
        elif hasattr(cls._model, "bert"):
            embedding_layer = cls._model.bert.embeddings.word_embeddings
        else:
            raise RuntimeError("Unsupported model architecture for Captum.")

        def forward_func(input_ids_arg, attention_mask_arg):
            """Matches CaptumExplainer.forward_func — takes raw input_ids."""
            return cls._model(
                input_ids=input_ids_arg,
                attention_mask=attention_mask_arg,
            ).logits[:, pred_idx]

        lig = LayerIntegratedGradients(forward_func, embedding_layer)

        # Baseline: pad_token_id (matches CaptumExplainer — NOT zeros)
        baseline_ids = torch.full_like(input_ids, cls._tokenizer.pad_token_id)

        attributions, delta = lig.attribute(
            input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True,
            n_steps=n_steps,
        )

        # Sum across embedding dim → scalar per token
        attributions_sum = attributions.sum(dim=-1).squeeze(0)

        tokens = cls._tokenizer.convert_ids_to_tokens(
            input_ids[0].cpu().tolist(),
            skip_special_tokens=False,
        )

        # Aggregate subword tokens → word level, matching _aggregate_word_attributions()
        word_attributions = []   # (word, abs_score, signed_score)
        current_indices = []

        special = {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>", "[PAD]"}

        for i, tok in enumerate(tokens):
            if tok in special:
                continue
            if tok.startswith("▁"):
                if current_indices:
                    grp = attributions_sum[current_indices].detach().cpu().numpy()
                    word = "".join(tokens[j].replace("▁", "") for j in current_indices)
                    word_attributions.append((word, float(np.sum(np.abs(grp))), float(np.sum(grp))))
                current_indices = [i]
            else:
                current_indices.append(i)

        if current_indices:
            grp = attributions_sum[current_indices].detach().cpu().numpy()
            word = "".join(tokens[j].replace("▁", "") for j in current_indices)
            word_attributions.append((word, float(np.sum(np.abs(grp))), float(np.sum(grp))))

        # word_scores uses signed score (matches what frontend chart will plot)
        word_scores = [
            {"word": w, "score": signed}
            for w, abs_s, signed in word_attributions
        ]

        return {
            "method": "Captum-IG",
            "prediction": pred_result["prediction"],
            "confidence": pred_result["confidence"],
            "word_scores": word_scores,
            "preprocessed_text": preprocessed,
            "convergence_delta": float(delta.sum().cpu().numpy()),
            "error": None,
        }

    # -------------------------------------------------------------------------
    # Private helpers
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