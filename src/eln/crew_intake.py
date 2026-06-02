"""Crew 1 — Lab Intake & Knowledge (≈ "Data Analyst Crew" du brief HUJI).

Trois agents : Intake/Structuring, Inventory, Retrieval (RAG).
Les agents sont définis avec rôles/objectifs/outils. Ils utilisent Claude si une clé
est présente ; sinon le Flow exploite directement les outils déterministes sous-jacents.
"""

from __future__ import annotations

from crewai import Agent

from .llm import get_llm
from .tools import parse_experiment_tool, predict_outcome_tool, search_history_tool

_llm = get_llm()


def intake_agent() -> Agent:
    return Agent(
        role="Lab Intake Specialist",
        goal="Convert spoken or typed experiment notes into a clean structured record.",
        backstory="A meticulous R&D technician who never loses a data point.",
        tools=[parse_experiment_tool],
        llm=_llm,
        verbose=False,
    )


def inventory_agent() -> Agent:
    return Agent(
        role="Inventory Manager",
        goal="Track reagent stock, deduct on each experiment, flag low stock and expiries.",
        backstory="Keeps the lab supplied and never lets a critical reagent run out.",
        llm=_llm,
        verbose=False,
    )


def retrieval_agent() -> Agent:
    return Agent(
        role="Knowledge Retrieval Agent",
        goal="Answer questions over past experiments, papers and patents using local search.",
        backstory="A librarian for the lab's entire knowledge base, fully on-premise.",
        tools=[search_history_tool],
        llm=_llm,
        verbose=False,
    )


CREW1_AGENTS = ["intake", "inventory", "retrieval"]
