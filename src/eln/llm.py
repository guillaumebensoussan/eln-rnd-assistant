"""Config LLM partagée. Claude en cloud si clé dispo, sinon dégradation gracieuse."""

from __future__ import annotations

import os

CLOUD_MODEL = "anthropic/claude-sonnet-4-5"


def has_cloud() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_llm():
    """Retourne un LLM CrewAI configuré sur Claude, ou None si pas de clé."""
    if not has_cloud():
        return None
    from crewai import LLM

    return LLM(model=CLOUD_MODEL, temperature=0.2)
