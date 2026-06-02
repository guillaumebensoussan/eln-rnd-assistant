"""Flow CrewAI — orchestration ELN de bout en bout.

Handoff Crew 1 (Intake & Knowledge) -> Crew 2 (Analytics & Reporting), avec :
- validation du dataset_contract et des features requises avant modélisation,
- étapes déterministes + logging clair,
- échec gracieux (le Flow s'arrête proprement et journalise au lieu de crasher).

Usage : `python -m eln.flow --demo`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

from .crew_analytics import write_report
from .inventory import Inventory
from .rag import RAG
from .sanitizer import Sanitizer, audit
from .tools import parse_experiment, predict_outcome

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "data" / "dataset_contract.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("eln.flow")


class ELNState(BaseModel):
    text: str = ""
    question: str = ""
    record: dict = {}
    inventory_result: dict = {}
    rag_context: str = ""
    prediction: dict = {}
    report: str = ""
    errors: list[str] = []


class ELNFlow(Flow[ELNState]):
    """Crew 1 -> validation -> Crew 2."""

    # ---- Crew 1 : Intake & Knowledge ----
    @start()
    def begin(self):
        return "begin"

    @listen(begin)
    def intake(self, _):
        log.info("[Crew1:Intake] structuring experiment entry")
        self.state.record = parse_experiment(self.state.text)
        return "intake_done"

    @listen(intake)
    def inventory(self, _):
        log.info("[Crew1:Inventory] deducting stock")
        inv = Inventory()
        rid = inv.df["reagent_id"].iloc[0]
        self.state.inventory_result = inv.deduct(rid, 5.0)
        return "inventory_done"

    @listen(inventory)
    def retrieval(self, _):
        log.info("[Crew1:Retrieval] RAG over local knowledge base")
        q = self.state.question or "expected yield for this experiment"
        self.state.rag_context = RAG().context(q, k=4)
        return "retrieval_done"

    # ---- Validation gate (handoff Crew1 -> Crew2) ----
    @listen(retrieval)
    def validate_handoff(self, _):
        log.info("[Gate] validating dataset contract + required features")
        try:
            contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.state.errors.append("dataset_contract.json missing")
            return "fail"
        required = contract["required_features_for_ml"]
        missing = [f for f in required if self.state.record.get(f) is None]
        if missing:
            self.state.errors.append(f"missing required features: {missing}")
            log.error("[Gate] handoff REJECTED: %s", missing)
            return "fail"
        log.info("[Gate] contract OK, handoff accepted")
        return "ok"

    # ---- Crew 2 : Analytics & Reporting ----
    @listen(validate_handoff)
    def analytics(self, gate):
        if gate == "fail":
            log.error("[Crew2] skipped — failing gracefully. errors=%s", self.state.errors)
            self.state.report = "FLOW HALTED: " + "; ".join(self.state.errors)
            return self.state

        log.info("[Crew2:Prediction] running scikit-learn model")
        self.state.prediction = predict_outcome(self.state.record)

        log.info("[Crew2:IP-Audit] sanitising context before any cloud call")
        clean, redacted = Sanitizer().scrub(self.state.rag_context)
        audit(
            operation="report_generation",
            sent_to_cloud=clean,
            redacted=redacted,
            stayed_local=["raw record", "embeddings", "inventory"],
        )

        log.info("[Crew2:Report] drafting report (%d names redacted)", len(redacted))
        self.state.report = write_report(clean, self.state.prediction)
        return self.state


def run(text: str, question: str = "") -> ELNState:
    flow = ELNFlow()
    flow.kickoff(inputs={"text": text, "question": question})
    s = flow.state
    # matérialise un snapshot détaché du StateProxy (lecture fiable hors du Flow)
    return ELNState(
        text=s.text,
        question=s.question,
        record=dict(s.record),
        inventory_result=dict(s.inventory_result),
        rag_context=s.rag_context,
        prediction=dict(s.prediction),
        report=s.report,
        errors=list(s.errors),
    )
