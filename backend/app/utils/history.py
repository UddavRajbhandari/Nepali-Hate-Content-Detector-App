"""
Prediction history store — thread-safe, append-only JSON Lines file.
"""

import json
import os
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List

HISTORY_FILE = os.getenv("HISTORY_FILE", "data/prediction_history.jsonl")
_lock = Lock()

def _ensure_dir() -> None:
    directory = os.path.dirname(HISTORY_FILE)
    if directory:
        os.makedirs(directory, exist_ok=True)

def _build_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "text": result.get("original_text", ""),
        "prediction": result.get("prediction", ""),
        "confidence": result.get("confidence", 0.0),
        "probabilities": result.get("probabilities", {}),
        "preprocessed_text": result.get("preprocessed_text", ""),
        "emoji_features": result.get("emoji_features", {}),
    }

def load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records

def append_history(result: Dict[str, Any]) -> None:
    entry = _build_entry(result)
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:
        _ensure_dir()
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def clear_history() -> None:
    with _lock:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)