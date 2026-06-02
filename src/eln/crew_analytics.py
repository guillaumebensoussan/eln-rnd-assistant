"""Crew 2 — Analytics & Reporting (≈ the "Data Scientist Crew" in the HUJI brief).

Three agents: Prediction (scikit-learn), Synthesis/Report (Claude), IP-Audit.
"""

from __future__ import annotations

from crewai import Agent, Crew, Task

from .llm import get_llm, has_cloud
from .tools import predict_outcome_tool, sanitize_text_tool

_llm = get_llm()


def prediction_agent() -> Agent:
    return Agent(
        role="Predictive Modelling Agent",
        goal="Estimate expected yield and QC outcome from the experiment parameters.",
        backstory="A data scientist who turns lab parameters into reliable forecasts.",
        tools=[predict_outcome_tool],
        llm=_llm,
        verbose=False,
    )


def report_agent() -> Agent:
    return Agent(
        role="Scientific Report Writer",
        goal="Draft a concise Methods section and synthesis from sanitised experiment context.",
        backstory="A scientific writer who only ever sees IP-safe, sanitised data.",
        llm=_llm,
        verbose=False,
    )


def audit_agent() -> Agent:
    return Agent(
        role="IP Compliance Auditor",
        goal="Ensure no IP-sensitive data leaves the premises; sanitise and log every cloud call.",
        backstory="A compliance officer obsessed with keeping trade secrets local.",
        tools=[sanitize_text_tool],
        llm=_llm,
        verbose=False,
    )


def write_report(sanitised_context: str, prediction: dict) -> str:
    """Write the report. Claude if available, otherwise a deterministic template (graceful degradation)."""
    if has_cloud():
        agent = report_agent()
        task = Task(
            description=(
                "Write a 'Methods' section and a one-paragraph synthesis for an R&D experiment. "
                f"Sanitised context:\n{sanitised_context}\n\n"
                f"Model prediction: {prediction}\n"
                "Be factual and concise."
            ),
            expected_output="A Methods section followed by a short synthesis paragraph.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        return str(crew.kickoff())
    # fallback template
    return (
        "## Methods (auto-draft, offline)\n"
        f"{sanitised_context}\n\n"
        "## Synthesis\n"
        f"Predicted yield: {prediction.get('predicted_yield_pct', 'n/a')}% · "
        f"predicted QC pass: {prediction.get('predicted_qc_pass', 'n/a')}. "
        "Report generated locally without cloud LLM (no API key set)."
    )


CREW2_AGENTS = ["prediction", "report", "audit"]
