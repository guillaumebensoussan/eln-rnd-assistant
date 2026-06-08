"""Crew member: EHS / Safety Officer.

Phrases a precomputed SafetyReport into a clear briefing. The LLM is used only
to word the report (sanitised, IP-safe); it never decides hazards. Without a
Claude key it falls back to a deterministic template.
"""

from __future__ import annotations

from crewai import Agent

from .llm import get_llm
from .safety.checker import SafetyReport
from .tools import check_safety_tool

_llm = get_llm()


def ehs_agent() -> Agent:
    return Agent(
        role="EHS / Safety Officer",
        goal="Flag hazards, incompatibilities and the correct mixing order before any bench work.",
        backstory="A lab safety officer who grounds every warning in verified GHS data and never improvises.",
        tools=[check_safety_tool],
        llm=_llm,
        verbose=False,
    )


def write_safety_briefing(report: SafetyReport) -> str:
    """Deterministic, grounded briefing from a SafetyReport (template fallback)."""
    lines: list[str] = []
    if report.severity == "critical":
        lines.append("## ⛔ SAFETY ALERT — critical incompatibility detected")
    elif report.severity == "warning":
        lines.append("## ⚠️ Safety warning")
    else:
        lines.append("## Safety check")

    if report.incompatibilities:
        lines.append("\n### Incompatibilities")
        for inc in report.incompatibilities:
            lines.append(f"- **[{inc.severity.upper()}] {inc.a} + {inc.b}** — {inc.reason}")

    if report.procedural_warnings:
        lines.append("\n### Mixing order")
        for w in report.procedural_warnings:
            lines.append(f"- {w}")
    if report.addition_order:
        lines.append(f"\nSuggested addition order: {' -> '.join(report.addition_order)}")

    if report.hazards:
        lines.append("\n### Per-reagent hazards")
        for h in report.hazards:
            codes = ", ".join(h.h_codes) if h.h_codes else "no public GHS data"
            lines.append(f"- {h.name} ({h.source}): {codes}")

    if report.unknowns:
        lines.append("\n### No public SDS — check the internal safety sheet")
        for u in report.unknowns:
            lines.append(f"- {u}")

    return "\n".join(lines)
