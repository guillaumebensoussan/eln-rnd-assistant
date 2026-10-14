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


def test_ollama_model_needs_no_api_key(monkeypatch):
    monkeypatch.setenv("ELN_MODEL", "ollama/gemma4:latest")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_ollama_up", lambda: True)
    assert llm.has_cloud() is True


def test_ollama_down_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("ELN_MODEL", "ollama/gemma4:latest")
    monkeypatch.setattr(llm, "_ollama_up", lambda: False)
    assert llm.has_cloud() is False


def test_ollama_is_flagged_as_local(monkeypatch):
    monkeypatch.setenv("ELN_MODEL", "ollama/gemma4:latest")
    assert llm.is_local() is True
    monkeypatch.setenv("ELN_MODEL", "anthropic/claude-sonnet-4-5")
    assert llm.is_local() is False


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


from eln.chat import safe_tools


def test_scrub_obj_removes_sensitive_names_recursively():
    s = Sanitizer()
    sensitive = next(iter(s.alias_map))
    payload = {"names": [sensitive, "Toluene"], "nested": {"note": f"used {sensitive}"}}
    clean = safe_tools.scrub_obj(payload, s)
    flat = repr(clean)
    assert sensitive not in flat
    assert "Toluene" in flat  # non-sensitive names pass through


def test_unalias_translates_back_for_local_execution():
    s = Sanitizer()
    sensitive, alias = next(iter(s.alias_map.items()))
    assert safe_tools.unalias(f"check {alias} please", s) == f"check {sensitive} please"


def test_safe_check_safety_accepts_aliases_and_returns_no_real_names(tmp_path, monkeypatch):
    import eln.sanitizer as sanitizer_mod
    monkeypatch.setattr(sanitizer_mod, "AUDIT_LOG", tmp_path / "audit.jsonl")
    s = Sanitizer()
    sensitive, alias = next(iter(s.alias_map.items()))
    result = safe_tools.run_check_safety([alias, "Toluene"])
    assert sensitive not in repr(result)  # outbound payload is scrubbed
    # the underlying check really resolved the real reagent (it is in inventory,
    # its CAS is internal so it lands in unknowns — as the alias, not the name)
    assert any(alias in u for u in result["unknowns"]) or result["hazards"]


from eln.chat.engine import ChatEngine, ChatReply


def _local_engine(monkeypatch, tmp_path):
    import eln.sanitizer as sanitizer_mod
    from eln import llm as llm_mod
    monkeypatch.setattr(sanitizer_mod, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(llm_mod, "has_cloud", lambda: False)
    return ChatEngine()


def test_local_safety_reply_is_critical_and_briefed(monkeypatch, tmp_path):
    engine = _local_engine(monkeypatch, tmp_path)
    reply = engine.reply([], "Is it safe to mix Hydrochloric acid with Sodium hydroxide?")
    assert isinstance(reply, ChatReply)
    assert reply.mode == "local"
    assert reply.critical is True
    assert "SAFETY ALERT" in reply.text


def test_local_safety_reply_without_reagents_asks_for_them(monkeypatch, tmp_path):
    engine = _local_engine(monkeypatch, tmp_path)
    reply = engine.reply([], "Is this mix dangerous?")
    assert reply.critical is False
    assert "reagent" in reply.text.lower()


def test_local_unknown_intent_lists_capabilities(monkeypatch, tmp_path):
    engine = _local_engine(monkeypatch, tmp_path)
    reply = engine.reply([], "Hello!")
    low = reply.text.lower()
    assert "safety" in low and "predict" in low and "history" in low and "inventory" in low


def test_local_inventory_reply_quotes_stock(monkeypatch, tmp_path):
    engine = _local_engine(monkeypatch, tmp_path)
    reply = engine.reply([], "How much Toluene is left in stock?")
    assert "Toluene" in reply.text


def test_cloud_failure_falls_back_to_local(monkeypatch, tmp_path):
    import eln.sanitizer as sanitizer_mod
    from eln import llm as llm_mod
    monkeypatch.setattr(sanitizer_mod, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(llm_mod, "has_cloud", lambda: True)
    engine = ChatEngine()
    monkeypatch.setattr(engine, "_cloud_reply",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("quota")))
    reply = engine.reply([], "Is it safe to mix Hydrochloric acid with Sodium hydroxide?")
    assert reply.mode == "local_fallback"
    assert "answered locally" in reply.text
    assert reply.critical is True


def test_chat_turn_is_audited(monkeypatch, tmp_path):
    import json as _json
    import eln.sanitizer as sanitizer_mod
    from eln import llm as llm_mod
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(sanitizer_mod, "AUDIT_LOG", log)
    monkeypatch.setattr(llm_mod, "has_cloud", lambda: False)
    ChatEngine().reply([], "How much Toluene is left?")
    entries = [_json.loads(line) for line in log.read_text().splitlines()]
    assert any(e["operation"] == "chat_turn" for e in entries)
    turn = next(e for e in entries if e["operation"] == "chat_turn")
    assert turn["chars_sent_to_cloud"] == 0  # local path: nothing left the machine


# Fix 1: word-boundary tests
def test_extract_reagents_word_boundaries():
    assert extract_reagents("the wastewater tank is underwater") == []
    assert extract_reagents("add Water to the flask") == ["Water"]


# Fix 2: token-based intent tests
def test_intent_reactor_is_not_safety():
    assert classify_intent("I put it in the reactor overnight") == "unknown"


def test_intent_turn_left_is_not_inventory():
    assert classify_intent("turn left then right at the bench") == "unknown"


def test_intent_reaction_is_safety():
    assert classify_intent("planning a reaction with bleach tomorrow") == "safety"


# Fix 3: history window scrub test
def test_history_window_is_scrubbed_before_cloud(monkeypatch, tmp_path):
    engine = _local_engine(monkeypatch, tmp_path)
    s = engine.sanitizer
    sensitive, alias = next(iter(s.alias_map.items()))
    history = [{"role": "assistant", "content": f"Stock of {sensitive}: 20 g"}] * 15
    transcript = engine._sanitized_transcript(history)
    assert sensitive not in transcript
    assert alias in transcript
    assert transcript.count("assistant:") == 12  # HISTORY_WINDOW


def test_local_llm_turn_audits_zero_chars_to_cloud(monkeypatch, tmp_path):
    import json as _json
    import eln.sanitizer as sanitizer_mod
    from eln import llm as llm_mod
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(sanitizer_mod, "AUDIT_LOG", log)
    monkeypatch.setattr(llm_mod, "has_cloud", lambda: True)
    monkeypatch.setattr(llm_mod, "is_local", lambda: True)
    engine = ChatEngine()
    monkeypatch.setattr(engine, "_cloud_reply",
                        lambda *a, **k: ChatReply(text="ok", mode="cloud"))
    engine.reply([], "How much Toluene is left in stock?")
    entries = [_json.loads(line) for line in log.read_text().splitlines()]
    turn = next(e for e in entries if e["operation"] == "chat_turn")
    assert turn["chars_sent_to_cloud"] == 0  # Ollama inference stays on-device
    assert any("Ollama" in s for s in turn["stayed_local"])
