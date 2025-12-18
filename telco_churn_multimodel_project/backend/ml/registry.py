from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import joblib

from backend.config import MODELS_DIR, REPORTS_DIR

@dataclass(frozen=True)
class ModelInfo:
    key: str
    path: Path

def available_models() -> List[ModelInfo]:
    MODELS_DIR.mkdir(exist_ok=True)
    models = []
    for p in MODELS_DIR.glob("*.joblib"):
        key = p.stem
        models.append(ModelInfo(key=key, path=p))
    models.sort(key=lambda m: m.key)
    return models

def load_model(model_key: str):
    path = MODELS_DIR / f"{model_key}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_key}. Train first.")
    return joblib.load(path)

def load_metrics() -> dict:
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_shap_summary() -> dict:
    shap_path = REPORTS_DIR / "shap_summary.json"
    if not shap_path.exists():
        return {}
    return json.loads(shap_path.read_text(encoding="utf-8"))

def best_model_key(metrics: dict, split: str = "validation", metric: str = "recall") -> Optional[str]:
    # Choose best by specified metric on specified split
    best_k, best_v = None, None
    for model_key, m in metrics.get("models", {}).items():
        v = m.get(split, {}).get(metric, None)
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v = v
            best_k = model_key
    return best_k
