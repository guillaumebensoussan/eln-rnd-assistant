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
