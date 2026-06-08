"""Orchestrator: turn a list of reagents into a deterministic SafetyReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd

from . import reactivity as rx
from .pubchem import DEFAULT_CACHE, PubChemClient

ROOT = Path(__file__).resolve().parents[3]
INVENTORY_CSV = ROOT / "data" / "inventory.csv"


@dataclass
class ReagentHazard:
    reagent_id: str
    name: str
    cas: str | None
    cid: int | None
    ghs_pictograms: list[str] = field(default_factory=list)
    h_codes: list[str] = field(default_factory=list)
    signal_word: str | None = None
    reactivity_groups: list[str] = field(default_factory=list)
    source: str = "unknown"


@dataclass
class Incompatibility:
    a: str
    b: str
    groups: tuple[str, str]
    severity: str
    reason: str


@dataclass
class SafetyReport:
    hazards: list[ReagentHazard] = field(default_factory=list)
    incompatibilities: list[Incompatibility] = field(default_factory=list)
    addition_order: list[str] = field(default_factory=list)
    procedural_warnings: list[str] = field(default_factory=list)
    severity: str = "info"
    unknowns: list[str] = field(default_factory=list)


def _resolve(name_or_id_or_cas: str, inv: pd.DataFrame) -> dict | None:
    key = str(name_or_id_or_cas).strip().lower()
    for col in ("name", "reagent_id", "cas"):
        match = inv[inv[col].astype(str).str.lower() == key]
        if not match.empty:
            row = match.iloc[0]
            return {"reagent_id": row["reagent_id"], "name": row["name"], "cas": str(row["cas"])}
    return None


def check_safety(reagents: list[str], inventory_csv: Path | None = None,
                 client: PubChemClient | None = None) -> SafetyReport:
    inv = pd.read_csv(inventory_csv or INVENTORY_CSV)
    client = client or PubChemClient(cache_path=DEFAULT_CACHE, allow_network=False)

    report = SafetyReport()
    groups_by_reagent: dict[str, list[str]] = {}

    for raw in reagents:
        resolved = _resolve(raw, inv)
        if resolved is None:
            report.unknowns.append(str(raw))
            continue
        haz_data = client.get_hazard(resolved["cas"])
        groups = rx.groups_for(resolved["cas"], haz_data.get("h_codes", []))
        hazard = ReagentHazard(
            reagent_id=resolved["reagent_id"], name=resolved["name"], cas=resolved["cas"],
            cid=haz_data.get("cid"), ghs_pictograms=haz_data.get("ghs_pictograms", []),
            h_codes=haz_data.get("h_codes", []), signal_word=haz_data.get("signal_word"),
            reactivity_groups=groups, source=haz_data.get("source", "unknown"),
        )
        report.hazards.append(hazard)
        groups_by_reagent[resolved["name"]] = groups
        if hazard.source == "unknown":
            report.unknowns.append(resolved["name"])

    severities: list[str] = []
    for h1, h2 in combinations(report.hazards, 2):
        pair = rx.evaluate_pair(h1.reactivity_groups, h2.reactivity_groups)
        if pair is not None:
            report.incompatibilities.append(Incompatibility(
                a=h1.name, b=h2.name, groups=(pair["a"], pair["b"]),
                severity=pair["severity"], reason=pair["reason"]))
            severities.append(pair["severity"])

    report.procedural_warnings = rx.mixing_warnings(groups_by_reagent)
    report.addition_order = _addition_order(report.hazards)
    report.severity = rx.max_severity(severities)
    return report


def _addition_order(hazards: list[ReagentHazard]) -> list[str]:
    """Water/solvent first, oxidizers/acids/bases last (added slowly)."""
    def rank(h: ReagentHazard) -> int:
        g = h.reactivity_groups
        if "water" in g:
            return 0
        if any(x in g for x in ("oxidizer", "acid", "base", "water_reactive")):
            return 2
        return 1
    return [h.name for h in sorted(hazards, key=rank)]
