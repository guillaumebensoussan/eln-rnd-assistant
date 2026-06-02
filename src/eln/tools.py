"""Deterministic tools shared by the agents and the Flow.

Each function can be used (a) directly by the Flow for a reproducible pipeline,
and (b) exposed as a CrewAI tool (`@tool`) to the agents.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd
from crewai.tools import tool

from .rag import RAG
from .sanitizer import Sanitizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model.pkl"

_SOLVENTS = ["THF", "DCM", "toluene", "methanol", "DMF", "water", "acetonitrile"]


def parse_experiment(text: str) -> dict:
    """Structure a natural-language experiment entry -> record."""
    def grab(pattern, default=None, cast=float):
        m = re.search(pattern, text, re.IGNORECASE)
        return cast(m.group(1)) if m else default

    solvent = next((s for s in _SOLVENTS if s.lower() in text.lower()), "THF")
    return {
        "temperature_c": grab(r"(\d+(?:\.\d+)?)\s*°?\s*c\b", 25.0),
        "time_h": grab(r"(\d+(?:\.\d+)?)\s*(?:h|hours?|hrs?)\b", 8.0),
        "catalyst_loading_molpct": grab(r"(\d+(?:\.\d+)?)\s*mol\s*%", 2.0),
        "pressure_bar": grab(r"(\d+(?:\.\d+)?)\s*bar", 1.0),
        "purity_in_pct": grab(r"purity\s*(\d+(?:\.\d+)?)", 95.0),
        "solvent": solvent,
        "raw_text": text,
    }


def predict_outcome(record: dict) -> dict:
    """Predict yield + QC pass using the trained scikit-learn model."""
    if not MODEL_PATH.exists():
        return {"error": "model not trained — run ml/train.py first"}
    bundle = joblib.load(MODEL_PATH)
    feats = bundle["num_features"] + bundle["cat_features"]
    X = pd.DataFrame([{k: record.get(k) for k in feats}])
    yield_pred = float(bundle["regressor"].predict(X)[0])
    qc_pred = bool(bundle["classifier"].predict(X)[0])
    return {"predicted_yield_pct": round(yield_pred, 2), "predicted_qc_pass": qc_pred}


# --- CrewAI tool versions (for the agents) ---

@tool("parse_experiment")
def parse_experiment_tool(text: str) -> dict:
    """Parse a free-text experiment description into a structured record."""
    return parse_experiment(text)


@tool("predict_outcome")
def predict_outcome_tool(record: dict) -> dict:
    """Predict reaction yield and QC pass from a structured experiment record."""
    return predict_outcome(record)


@tool("search_history")
def search_history_tool(query: str) -> str:
    """Semantic search over local experiment history, papers and patents."""
    return RAG().context(query, k=4)


@tool("sanitize_text")
def sanitize_text_tool(text: str) -> dict:
    """Redact IP-sensitive reagent names before any cloud call."""
    clean, redacted = Sanitizer().scrub(text)
    return {"clean": clean, "redacted": redacted}
