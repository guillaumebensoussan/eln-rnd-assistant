# ELN Chat Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A multi-turn "Assistant" chat tab in the Streamlit app that answers free-text questions using all ELN capabilities (safety, prediction, RAG history, inventory), with a hybrid cloud/local engine and a structurally enforced IP boundary.

**Architecture:** Hybrid per the approved spec (`docs/superpowers/specs/2026-06-10-eln-chat-assistant-design.md`): a `ChatEngine` with two reply paths — CrewAI orchestrator + EHS agent with a sanitised tool belt when a cloud key is configured, deterministic keyword router + existing template writers otherwise. Reagent extraction runs on raw text locally *before* sanitisation; the cloud LLM only ever sees aliases; replies are de-sanitised locally for display.

**Tech Stack:** Python 3.13, uv, CrewAI (LiteLLM), ChromaDB, pandas, Streamlit, pytest.

**Conventions (apply to every task):**
- Repo root: all commands and paths are relative to the repository root.
- Everything in English: code, comments, docstrings, UI strings. Zero French.
- Test command (Chroma store must be purged first or stale state breaks RAG tests):
  `rm -rf artifacts/chroma && PYTHONPATH=src uv run pytest -q`
- Lint: `uv run ruff check .` must stay clean.
- After running tests, `artifacts/ip_audit_log.jsonl` may have new runtime lines: run `git restore artifacts/ip_audit_log.jsonl` before committing.
- Test files set `sys.path` before imports (E402 is allowed in `tests/*.py` via pyproject).

---

### Task 1: `Sanitizer.unscrub()` — inverse alias mapping

The chat displays answers with real reagent names; the cloud only sees aliases. We need the inverse translation.

**Files:**
- Modify: `src/eln/sanitizer.py` (add one method to `Sanitizer`)
- Test: `tests/test_chat.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_chat.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_chat.py -q`
Expected: FAIL with `AttributeError: 'Sanitizer' object has no attribute 'unscrub'`

- [ ] **Step 3: Implement `unscrub`**

In `src/eln/sanitizer.py`, add to the `Sanitizer` class, after `scrub`:

```python
    def unscrub(self, text: str) -> str:
        """Inverse mapping (alias -> real name) for local display only.

        The result must NEVER be sent to the cloud.
        """
        for name, alias in self.alias_map.items():
            text = text.replace(alias, name)
        return text
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_chat.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/eln/sanitizer.py tests/test_chat.py
git commit -m "feat(chat): Sanitizer.unscrub inverse mapping for local display"
```

---

### Task 2: Configurable LLM provider (`ELN_MODEL`)

**Files:**
- Modify: `src/eln/llm.py` (full rewrite below)
- Modify: `.env.example`
- Test: `tests/test_chat.py` (append)

- [ ] **Step 1: Check for other usages of `CLOUD_MODEL`**

Run: `grep -rn "CLOUD_MODEL" src/ app/ tests/`
Expected: only `src/eln/llm.py`. If anything else shows up, update it to use `get_model()` in step 3.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_chat.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_chat.py -q`
Expected: the two new tests FAIL (`has_cloud` ignores `ELN_MODEL` today)

- [ ] **Step 4: Rewrite `src/eln/llm.py`**

```python
"""Shared LLM config. Cloud LLM if the right key is available, otherwise graceful degradation.

The provider is configurable via ELN_MODEL (any LiteLLM model string).
Examples: anthropic/claude-sonnet-4-5 (default), gemini/gemini-2.5-flash,
openai/gpt-4o-mini.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"

_KEY_FOR_PREFIX = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def get_model() -> str:
    return os.environ.get("ELN_MODEL", DEFAULT_MODEL)


def required_key_name() -> str:
    """Env var holding the API key for the configured provider."""
    prefix = get_model().split("/", 1)[0]
    return _KEY_FOR_PREFIX.get(prefix, f"{prefix.upper()}_API_KEY")


def has_cloud() -> bool:
    return bool(os.environ.get(required_key_name()))


def get_llm():
    """Return a CrewAI LLM for the configured provider, or None if no key is set."""
    if not has_cloud():
        return None
    from crewai import LLM

    return LLM(model=get_model(), temperature=0.2)
```

- [ ] **Step 5: Update `.env.example`**

```bash
# Optional cloud LLM for wording (everything else runs locally).
# Pick ONE provider: set ELN_MODEL (LiteLLM format) and the matching key.
# Default model (if ELN_MODEL unset): anthropic/claude-sonnet-4-5
ELN_MODEL=
ANTHROPIC_API_KEY=
# GEMINI_API_KEY=        # for ELN_MODEL=gemini/gemini-2.5-flash
# OPENAI_API_KEY=        # for ELN_MODEL=openai/gpt-4o-mini
```

- [ ] **Step 6: Run the full suite (module-level `_llm = get_llm()` in crew files must still import fine)**

Run: `rm -rf artifacts/chroma && PYTHONPATH=src uv run pytest -q`
Expected: all pass (29 at this point)

- [ ] **Step 7: Commit**

```bash
git restore artifacts/ip_audit_log.jsonl
git add src/eln/llm.py .env.example tests/test_chat.py
git commit -m "feat(llm): configurable provider via ELN_MODEL (Claude default, Gemini/OpenAI supported)"
```

---

### Task 3: `chat/router.py` — local intent classification + reagent extraction

**Files:**
- Create: `src/eln/chat/__init__.py`
- Create: `src/eln/chat/router.py`
- Test: `tests/test_chat.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_chat.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'eln.chat'`

- [ ] **Step 3: Create the module**

`src/eln/chat/__init__.py`:

```python
"""Conversational assistant — local router, sanitised tool belt, hybrid engine."""
```

`src/eln/chat/router.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_chat.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/eln/chat/ tests/test_chat.py
git commit -m "feat(chat): deterministic intent router and inventory-based reagent extraction"
```

---

### Task 4: `chat/safe_tools.py` — sanitised tool belt

**Files:**
- Create: `src/eln/chat/safe_tools.py`
- Test: `tests/test_chat.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_chat.py -q`
Expected: FAIL with `ImportError` (no `safe_tools`)

- [ ] **Step 3: Implement `src/eln/chat/safe_tools.py`**

```python
"""Sanitised tool belt for the chat assistant — the structural IP guarantee.

The cloud-side agent only ever sees aliases (REAGENT_XX):
- inbound: tool arguments are translated alias -> real name locally;
- outbound: every tool result is scrubbed (real -> alias) before returning;
- every call writes an audit entry.

Pure helpers (scrub_obj, unalias, run_*) are module functions so they are
testable without CrewAI; the @tool wrappers below expose them to the agent.
"""

from __future__ import annotations

import json

from crewai.tools import tool

from ..rag import RAG
from ..sanitizer import Sanitizer, audit
from ..tools import check_safety as _check_safety
from ..tools import parse_experiment as _parse_experiment
from ..tools import predict_outcome as _predict_outcome

_sanitizer = Sanitizer()


def unalias(text: str, sanitizer: Sanitizer | None = None) -> str:
    """Alias -> real name, locally, so underlying tools get real inventory names."""
    return (sanitizer or _sanitizer).unscrub(text)


def scrub_obj(obj, sanitizer: Sanitizer | None = None):
    """Recursively sanitise every string in a JSON-like structure."""
    s = sanitizer or _sanitizer
    if isinstance(obj, str):
        return s.scrub(obj)[0]
    if isinstance(obj, list):
        return [scrub_obj(x, s) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_obj(v, s) for k, v in obj.items()}
    return obj


def _audited(name: str, result):
    clean = scrub_obj(result)
    audit(
        operation=f"chat_tool:{name}",
        sent_to_cloud=json.dumps(clean, ensure_ascii=False, default=str),
        redacted=[],
        stayed_local=["raw tool arguments and results"],
    )
    return clean


def run_check_safety(reagents: list[str]) -> dict:
    real = [unalias(r) for r in reagents]
    return _audited("check_safety", _check_safety(real))


def run_search_history(query: str) -> str:
    return _audited("search_history", RAG().context(unalias(query), k=4))


def run_predict_outcome(record: dict) -> dict:
    return _audited("predict_outcome", _predict_outcome(record))


def run_parse_experiment(text: str) -> dict:
    return _audited("parse_experiment", _parse_experiment(unalias(text)))


@tool("check_safety")
def safe_check_safety_tool(reagents: list[str]) -> dict:
    """Check GHS hazards, incompatibilities and mixing order for reagents (aliases OK)."""
    return run_check_safety(reagents)


@tool("search_history")
def safe_search_history_tool(query: str) -> str:
    """Semantic search over local experiment history, papers and patents."""
    return run_search_history(query)


@tool("predict_outcome")
def safe_predict_outcome_tool(record: dict) -> dict:
    """Predict reaction yield and QC pass from a structured experiment record."""
    return run_predict_outcome(record)


@tool("parse_experiment")
def safe_parse_experiment_tool(text: str) -> dict:
    """Parse a free-text experiment description into a structured record."""
    return run_parse_experiment(text)


SAFE_TOOLS = [
    safe_check_safety_tool,
    safe_search_history_tool,
    safe_predict_outcome_tool,
    safe_parse_experiment_tool,
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_chat.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git restore artifacts/ip_audit_log.jsonl
git add src/eln/chat/safe_tools.py tests/test_chat.py
git commit -m "feat(chat): sanitised tool belt — aliases in, scrubbed results out, audited"
```

---

### Task 5: `chat/engine.py` — local path

**Files:**
- Create: `src/eln/chat/engine.py`
- Modify: `src/eln/chat/__init__.py`
- Test: `tests/test_chat.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_chat.py -q`
Expected: FAIL with `ImportError` (no `engine`)

- [ ] **Step 3: Implement `src/eln/chat/engine.py`**

Note: `has_cloud` is referenced as `llm.has_cloud()` (module attribute) on purpose — it makes monkeypatching reliable.

```python
"""ChatEngine — hybrid conversational engine, the only API the UI knows.

Cloud path (key configured): CrewAI orchestrator + EHS agent with the
sanitised tool belt. Local path (no key, or cloud failure): deterministic
router + existing template writers. Both paths share one audited IP boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import llm
from ..crew_safety import write_safety_briefing
from ..inventory import Inventory
from ..rag import RAG
from ..safety import check_safety
from ..sanitizer import Sanitizer, audit
from ..tools import parse_experiment, predict_outcome
from .router import classify_intent, extract_reagents

HISTORY_WINDOW = 12

_LOCAL_CAPABILITIES = (
    "I am running in local mode (no cloud LLM). I can answer questions about:\n"
    "- **safety** — e.g. \"Is it safe to mix bleach with ammonia?\"\n"
    "- **prediction** — e.g. \"What yield should I expect at 110C in toluene?\"\n"
    "- **history** — e.g. \"Did we run a similar coupling before?\"\n"
    "- **inventory** — e.g. \"How much toluene is left in stock?\""
)


@dataclass
class ChatReply:
    text: str
    critical: bool = False
    mode: str = "local"  # "cloud" | "local" | "local_fallback"


class ChatEngine:
    def __init__(self):
        self.sanitizer = Sanitizer()

    def reply(self, history: list[dict], message: str) -> ChatReply:
        """history: [{"role": "user"|"assistant", "content": str}, ...]"""
        reagents = extract_reagents(message)  # raw text, local, BEFORE scrub
        clean_msg, redacted = self.sanitizer.scrub(message)

        if llm.has_cloud():
            audit(operation="chat_turn", sent_to_cloud=clean_msg,
                  redacted=redacted, stayed_local=["raw message", "reagent names"])
            try:
                return self._cloud_reply(history, clean_msg, reagents)
            except Exception:
                local = self._local_reply(message, reagents)
                return ChatReply(
                    text=local.text + "\n\n_answered locally (cloud unavailable)_",
                    critical=local.critical, mode="local_fallback")

        audit(operation="chat_turn", sent_to_cloud="", redacted=redacted,
              stayed_local=["raw message", "reagent names", "full local answer"])
        return self._local_reply(message, reagents)

    # ---- local path (deterministic) ----

    def _local_reply(self, message: str, reagents: list[str]) -> ChatReply:
        intent = classify_intent(message)
        if intent == "safety":
            if not reagents:
                return ChatReply(text="Which reagents do you plan to combine? "
                                      "Name them and I will run the safety check.")
            report = check_safety(reagents)
            return ChatReply(text=write_safety_briefing(report),
                             critical=report.severity == "critical")
        if intent == "predict":
            record = parse_experiment(message)
            pred = predict_outcome(record)
            if "error" in pred:
                return ChatReply(text=f"Prediction unavailable: {pred['error']}")
            return ChatReply(text=(
                f"Predicted yield: **{pred['predicted_yield_pct']}%** · "
                f"predicted QC pass: **{pred['predicted_qc_pass']}** "
                f"(solvent={record['solvent']}, T={record['temperature_c']}C, "
                f"t={record['time_h']}h)"))
        if intent == "history":
            context = RAG().context(message, k=4)
            return ChatReply(text="From the local knowledge base:\n\n" + context)
        if intent == "inventory":
            inv = Inventory()
            df = inv.df[inv.df["name"].isin(reagents)] if reagents else inv.df
            lines = [f"- {r['name']}: {r['quantity_g']} g · expires {r['expiry']} "
                     f"· {r['location']}" for _, r in df.head(10).iterrows()]
            return ChatReply(text="Current stock:\n" + "\n".join(lines))
        if intent == "log_experiment":
            record = parse_experiment(message)
            return ChatReply(text="Structured record (use the *Log experiment* tab "
                                  f"to run the full flow):\n```json\n{record}\n```")
        return ChatReply(text=_LOCAL_CAPABILITIES)

    # ---- cloud path (CrewAI orchestrator + EHS agent, sanitised tools) ----

    def _cloud_reply(self, history: list[dict], clean_msg: str,
                     reagents: list[str]) -> ChatReply:
        from crewai import Agent, Crew, Task

        from ..crew_safety import ehs_agent
        from .safe_tools import SAFE_TOOLS

        window = history[-HISTORY_WINDOW:]
        transcript = "\n".join(
            f"{m['role']}: {self.sanitizer.scrub(m['content'])[0]}" for m in window)
        aliases = [self.sanitizer.scrub(r)[0] for r in reagents]

        assistant = Agent(
            role="Lab Assistant",
            goal="Answer the chemist using the lab's tools; ground every claim "
                 "in tool results, never invent hazard data.",
            backstory="A meticulous R&D assistant. Reagent names like REAGENT_A1 "
                      "are anonymised aliases: use them verbatim, never guess "
                      "what they might be.",
            tools=SAFE_TOOLS,
            llm=llm.get_llm(),
            verbose=False,
        )
        task = Task(
            description=(
                f"Conversation so far:\n{transcript}\n\n"
                f"New message from the chemist:\n{clean_msg}\n\n"
                f"Reagents detected in the message (aliases): {aliases}\n"
                "Answer the new message. If the question involves combining "
                "reagents, you MUST call check_safety and report its findings "
                "faithfully, including severity."
            ),
            expected_output="A concise, factual answer in Markdown.",
            agent=assistant,
        )
        crew = Crew(agents=[assistant, ehs_agent()], tasks=[task], verbose=False)
        answer = str(crew.kickoff())

        critical = False
        if reagents:
            critical = check_safety(reagents).severity == "critical"
        return ChatReply(text=self.sanitizer.unscrub(answer),
                         critical=critical, mode="cloud")
```

Then update `src/eln/chat/__init__.py`:

```python
"""Conversational assistant — local router, sanitised tool belt, hybrid engine."""

from .engine import ChatEngine, ChatReply

__all__ = ["ChatEngine", "ChatReply"]
```

- [ ] **Step 4: Run the full suite**

Run: `rm -rf artifacts/chroma && PYTHONPATH=src uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Run lint**

Run: `uv run ruff check .`
Expected: All checks passed!

- [ ] **Step 6: Commit**

```bash
git restore artifacts/ip_audit_log.jsonl
git add src/eln/chat/ tests/test_chat.py
git commit -m "feat(chat): hybrid ChatEngine — CrewAI orchestrator with cloud key, deterministic router without"
```

---

### Task 6: Streamlit "Assistant" tab

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Add the tab**

In `app/streamlit_app.py`, change the imports block to add:

```python
from eln.chat import ChatEngine
from eln.llm import get_model
```

Change the tabs line to put Assistant first:

```python
tab_chat, tab_log, tab_inv, tab_safety, tab_audit, tab_hist = st.tabs(
    ["Assistant", "Log experiment", "Inventory", "Safety check", "IP Audit log", "History"]
)
```

Add the tab body (before `with tab_log:`):

```python
with tab_chat:
    st.subheader("Lab assistant")
    st.caption(
        f"Cloud LLM: {get_model()}" if has_cloud()
        else "Local mode — deterministic answers (no API key set)"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_pending_ack" not in st.session_state:
        st.session_state.chat_pending_ack = False

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.chat_pending_ack:
        st.error("⛔ SAFETY ALERT — critical incompatibility. "
                 "Read the warnings above before continuing.")
        if st.checkbox("I have read these warnings and take responsibility.",
                       key="chat_ack"):
            from eln.sanitizer import audit
            audit(operation="safety_acknowledgement", sent_to_cloud="",
                  redacted=[], stayed_local=["chat critical-alert acknowledgement"])
            st.session_state.chat_pending_ack = False
            st.rerun()

    prompt = st.chat_input("Describe your next reaction or ask about the lab",
                           disabled=st.session_state.chat_pending_ack)
    if prompt:
        with st.spinner("Thinking…"):
            reply = ChatEngine().reply(st.session_state.chat_history, prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": reply.text})
        if reply.critical:
            st.session_state.chat_pending_ack = True
        st.rerun()
```

- [ ] **Step 2: Manual smoke test (local mode)**

Run: `uv run streamlit run app/streamlit_app.py` (no API key in env)
Check, then Ctrl-C:
- Assistant tab is first and shows "Local mode".
- "Is it safe to mix Hydrochloric acid with Sodium hydroxide?" → ⛔ briefing, chat input disabled until the acknowledgement box is ticked.
- "How much Toluene is left in stock?" → stock line.
- "Hello!" → capability list.
- IP Audit log tab now shows `chat_turn` entries.

- [ ] **Step 3: Run the full suite + lint (UI changes must not break imports)**

Run: `rm -rf artifacts/chroma && PYTHONPATH=src uv run pytest -q && uv run ruff check .`
Expected: all pass, lint clean

- [ ] **Step 4: Commit**

```bash
git restore artifacts/ip_audit_log.jsonl
git add app/streamlit_app.py
git commit -m "feat(chat): Assistant tab — multi-turn chat with critical-alert acknowledgement"
```

---

### Task 7: README + final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the assistant in `README.md`**

Add a section after the safety-module section (match the file's existing tone; key points):

```markdown
## Chat assistant

The **Assistant** tab is a multi-turn chat over all ELN capabilities
(safety, prediction, history, inventory).

- **With a cloud key**: a CrewAI orchestrator (plus the EHS agent) answers,
  using a sanitised tool belt — the cloud LLM only ever sees aliases
  (`REAGENT_A1`), tool calls run locally on real names, results are scrubbed
  before returning, and every turn lands in the IP audit log.
- **Without a key**: a deterministic keyword router answers from local
  engines and templates. Nothing leaves the machine.

### Choosing the LLM provider

Set `ELN_MODEL` in `.env` (LiteLLM format) with the matching key:

| Provider | ELN_MODEL | Key |
|---|---|---|
| Anthropic (default) | `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
```

- [ ] **Step 2: Full verification**

Run, in order, all from repo root:
1. `rm -rf artifacts/chroma && PYTHONPATH=src uv run pytest -q` — expected: all pass (≈45)
2. `uv run ruff check .` — expected: All checks passed!
3. Anti-French scan: `grep -rnE "\b(le|la|les|des|une|avec|pour|réactif|sécurité|mélange)\b" src/ app/ tests/ README.md --include="*.py" --include="*.md" | grep -v Binary` — expected: no hits in code/docs (manually review any false positives)

- [ ] **Step 3: Commit**

```bash
git restore artifacts/ip_audit_log.jsonl
git add README.md
git commit -m "docs: document the chat assistant and LLM provider configuration"
```

---

## Self-review notes

- Spec coverage: components (Tasks 3-6), IP boundary incl. de-sanitised display (Tasks 1, 4, 5), degraded mode + cloud-failure fallback (Task 5), provider config (Task 2), UI + critical ack (Task 6), audit of chat turns and tool calls (Tasks 4, 5), docs (Task 7). History window of 12 in `engine.py` per spec.
- The cloud path is not covered by automated tests (no key in CI) — same policy as the existing codebase; the fallback path *is* tested.
- `ChatReply.critical` on the cloud path re-runs `check_safety` locally on extracted reagents rather than trusting the LLM — deterministic decision, LLM wording only.
