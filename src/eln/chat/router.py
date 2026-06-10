"""Deterministic chat routing — pure, offline, fully testable.

Intent classification by keywords and reagent extraction by inventory matching.
IMPORTANT: extraction runs on RAW user text, locally, BEFORE sanitisation —
after scrubbing, aliases would no longer match inventory names.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INVENTORY_CSV = ROOT / "data" / "inventory.csv"

# Priority order: first intent with a keyword hit wins (safety outranks all).
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "safety": ("safe", "safety", "hazard", "danger", "mix", "incompatib",
               "toxic", "explos", "react"),
    "predict": ("yield", "predict", "outcome", "expect", "qc"),
    "history": ("previous", "history", "past", "before", "last time",
                "similar", "paper", "patent", "literature"),
    "inventory": ("stock", "inventory", "quantity", "expir", "how much",
                  "remaining", "left"),
    "log_experiment": ("log this", "record this", "i ran", "i performed",
                       "completed"),
}


def classify_intent(text: str) -> str:
    low = text.lower()
    for intent, words in _KEYWORDS.items():
        if any(w in low for w in words):
            return intent
    return "unknown"


def extract_reagents(text: str, inventory_csv: Path | None = None) -> list[str]:
    """Canonical inventory names found in raw text, longest-first, deduped."""
    inv = pd.read_csv(inventory_csv or INVENTORY_CSV)
    names = sorted(inv["name"].astype(str), key=len, reverse=True)
    if not names:
        return []
    pattern = re.compile("|".join(re.escape(n) for n in names), re.IGNORECASE)
    canonical = {n.lower(): n for n in names}
    found: list[str] = []
    for m in pattern.finditer(text):
        name = canonical[m.group(0).lower()]
        if name not in found:
            found.append(name)
    return found
