"""Chat assistant tests: sanitizer round-trip, router, safe tools, engine, llm config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eln.sanitizer import Sanitizer


def test_sanitizer_unscrub_is_inverse_of_scrub():
    s = Sanitizer()
    sensitive = next(iter(s.alias_map))  # a real ip_sensitive name from inventory
    original = f"Mix {sensitive} with toluene at 80C."
    clean, redacted = s.scrub(original)
    assert sensitive not in clean
    assert redacted == [sensitive]
    assert s.unscrub(clean) == original


def test_sanitizer_unscrub_handles_text_without_aliases():
    s = Sanitizer()
    assert s.unscrub("plain text, no alias") == "plain text, no alias"


from eln import llm


def test_default_model_requires_anthropic_key(monkeypatch):
    monkeypatch.delenv("ELN_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.has_cloud() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert llm.has_cloud() is True


def test_gemini_model_requires_gemini_key(monkeypatch):
    monkeypatch.setenv("ELN_MODEL", "gemini/gemini-2.5-flash")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # wrong provider's key
    assert llm.has_cloud() is False
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    assert llm.has_cloud() is True


def test_unknown_provider_falls_back_to_prefix_convention(monkeypatch):
    monkeypatch.setenv("ELN_MODEL", "mistral/mistral-large")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert llm.has_cloud() is False


from eln.chat.router import classify_intent, extract_reagents


def test_intent_safety():
    assert classify_intent("Is it safe to mix bleach with ammonia?") == "safety"


def test_intent_predict():
    assert classify_intent("What yield should I expect at 110C?") == "predict"


def test_intent_history():
    assert classify_intent("Did we run a similar coupling before?") == "history"


def test_intent_inventory():
    assert classify_intent("How much toluene is left in stock?") == "inventory"


def test_intent_unknown():
    assert classify_intent("Hello there!") == "unknown"


def test_intent_safety_wins_over_predict_on_tie():
    # one keyword each -> priority order puts safety first
    assert classify_intent("Is this mix safe and what yield?") == "safety"


def test_extract_reagents_multiword_and_case():
    found = extract_reagents("mix HYDROCHLORIC ACID with sodium hydroxide slowly")
    assert found == ["Hydrochloric acid", "Sodium hydroxide"]


def test_extract_reagents_ip_sensitive_and_dedup():
    found = extract_reagents("IntermediateX-171 then more IntermediateX-171 in Toluene")
    assert found == ["IntermediateX-171", "Toluene"]


def test_extract_reagents_none():
    assert extract_reagents("no chemicals here") == []
