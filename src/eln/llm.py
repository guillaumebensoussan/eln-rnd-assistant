"""Shared LLM config. LLM if the right key (or local server) is available, otherwise graceful degradation.

The provider is configurable via ELN_MODEL (any LiteLLM model string).
Examples: anthropic/claude-sonnet-4-5 (default), gemini/gemini-2.5-flash,
openai/gpt-4o-mini, ollama/gemma4:latest (local, no API key needed).
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"
OLLAMA_BASE_URL = "http://localhost:11434"

_KEY_FOR_PREFIX = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Providers that run on this machine: no API key, nothing leaves the workstation.
_LOCAL_PREFIXES = ("ollama",)


def get_model() -> str:
    return os.environ.get("ELN_MODEL", DEFAULT_MODEL)


def _prefix() -> str:
    return get_model().split("/", 1)[0]


def is_local() -> bool:
    """True when the configured LLM runs on this machine (e.g. Ollama)."""
    return _prefix() in _LOCAL_PREFIXES


def required_key_name() -> str:
    """Env var holding the API key for the configured provider."""
    return _KEY_FOR_PREFIX.get(_prefix(), f"{_prefix().upper()}_API_KEY")


def has_cloud() -> bool:
    """True when an LLM is usable: local server configured, or provider key present."""
    if is_local():
        return True
    return bool(os.environ.get(required_key_name()))


def get_llm():
    """Return a CrewAI LLM for the configured provider, or None if no key is set."""
    if not has_cloud():
        return None
    from crewai import LLM

    if is_local():
        return LLM(model=get_model(), base_url=OLLAMA_BASE_URL, temperature=0.2)
    return LLM(model=get_model(), temperature=0.2)
