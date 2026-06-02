"""Shared LLM config. Claude in the cloud if a key is available, otherwise graceful degradation."""

from __future__ import annotations

import os

CLOUD_MODEL = "anthropic/claude-sonnet-4-5"


def has_cloud() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_llm():
    """Return a CrewAI LLM configured for Claude, or None if no key is set."""
    if not has_cloud():
        return None
    from crewai import LLM

    return LLM(model=CLOUD_MODEL, temperature=0.2)
