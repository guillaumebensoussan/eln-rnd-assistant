"""Build the hierarchical Crew: Claude manager delegating to 3 Ollama workers."""

from __future__ import annotations

from crewai import Crew, Process, Task

from lab_team import paths  # noqa: F401  (extends sys.path) — must be first
import db

from lab_team.agents.eln_database import build_eln_agent
from lab_team.agents.inventory_safety import build_inventory_agent
from lab_team.agents.manager import build_manager_agent
from lab_team.agents.scientist import build_scientist_agent
from lab_team.llm_factory import manager_llm, worker_llm


def build_crew() -> Crew:
    """Provision the DB and assemble the crew. Call once; reuse for every turn."""
    db.init_db()  # inventory + reservations + ELN schema (idempotent)

    manager = build_manager_agent(manager_llm())
    wl = worker_llm()
    inventory = build_inventory_agent(wl)
    eln = build_eln_agent(wl)
    scientist = build_scientist_agent(wl)

    task = Task(
        description=(
            "Handle this lab request from the user:\n\n{user_request}\n\n"
            "Rules:\n"
            "1. Delegate ONE sub-task at a time — wait for each result before "
            "delegating the next.\n"
            "2. For CREATE requests: first create the parent record, then "
            "immediately create all child records (experiments, materials) in "
            "separate follow-up delegations — do NOT skip any step.\n"
            "3. Return one consolidated answer including: record IDs created, "
            "absolute file paths generated, and every safety alert relayed.\n"
            "4. Use exact names/codes from the user request — never substitute "
            "or paraphrase them.\n"
            "Specialists: 'Inventory & Safety Officer' (stock, expiry, SDS), "
            "'ELN Database Steward' (projects, experiments, materials), "
            "'Research Scientist & Report Author' (Word/Excel reports, charts)."
        ),
        expected_output=(
            "A clear, consolidated answer to the user's request, including created "
            "record identifiers, generated file paths, and any safety alerts."
        ),
        # Do NOT set agent=manager here: CrewAI uses task.agent to build the
        # delegation tool roster, and if set it only includes [task.agent] instead
        # of self.agents (the 3 workers). Leave it None so workers are visible.
    )

    return Crew(
        agents=[inventory, eln, scientist],  # workers only; manager is separate
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )


def ask(crew: Crew, user_request: str) -> str:
    """Run one turn through the crew and return the manager's final answer."""
    result = crew.kickoff(inputs={"user_request": user_request})
    return str(result)
