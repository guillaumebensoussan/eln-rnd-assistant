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
