"""Factory for the two CrewAI LLM objects: Claude manager, Ollama workers."""

from __future__ import annotations

from crewai import LLM

from lab_team.config import MANAGER_MODEL, OLLAMA_BASE_URL, WORKER_MODEL


# Pass num_ctx via extra_body — OpenAI client forwards it verbatim in the JSON body,
# Ollama's /v1 endpoint reads options.num_ctx from there.
_OLLAMA_EXTRA = {"extra_body": {"options": {"num_ctx": 16384}}}


def manager_llm() -> LLM:
    """Manager LLM — currently Ollama fallback; switch MANAGER_MODEL back to anthropic/* for Claude."""
    if MANAGER_MODEL.startswith("ollama/"):
        return LLM(model=MANAGER_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2,
                   additional_params=_OLLAMA_EXTRA)
    return LLM(model=MANAGER_MODEL, temperature=0.2)


def worker_llm() -> LLM:
    """Local Ollama worker model. Temperature 0 for deterministic tool calls."""
    return LLM(model=WORKER_MODEL, base_url=OLLAMA_BASE_URL, temperature=0,
               additional_params=_OLLAMA_EXTRA)
