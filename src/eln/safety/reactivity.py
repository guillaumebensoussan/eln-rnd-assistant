"""Pure, deterministic reactivity rule engine.

Loads curated rules from data/reactivity_rules.json and exposes:
- groups_for(cas, h_codes): reactivity groups of a reagent,
- evaluate_pair(groups_a, groups_b): incompatibility for a pair (or None),
- mixing_warnings(groups_by_reagent): procedural order warnings.

No network, no I/O beyond reading the rules file once.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RULES_PATH = ROOT / "data" / "reactivity_rules.json"

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@lru_cache(maxsize=1)
def _rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def groups_for(cas: str | None, h_codes: list[str]) -> list[str]:
    """Reactivity groups from H-codes (GHS-derived) + curated CAS overrides."""
    rules = _rules()
    groups: list[str] = []
    for h in h_codes:
        for g in rules["hcode_to_groups"].get(h, []):
            if g not in groups:
                groups.append(g)
    for g in rules["cas_overrides"].get(cas or "", []):
        if g not in groups:
            groups.append(g)
    return groups


def evaluate_pair(groups_a: list[str], groups_b: list[str]) -> dict | None:
    """Most severe incompatibility between two reagents' groups, or None."""
    rules = _rules()
    best: dict | None = None
    for pair in rules["pairs"]:
        a, b = pair["a"], pair["b"]
        hit = (a in groups_a and b in groups_b) or (a in groups_b and b in groups_a)
        if hit and (best is None or _SEVERITY_ORDER[pair["severity"]] > _SEVERITY_ORDER[best["severity"]]):
            best = pair
    return best


def mixing_warnings(groups_by_reagent: dict[str, list[str]]) -> list[str]:
    """Procedural order instructions given which groups are present."""
    rules = _rules()
    present: set[str] = set()
    for groups in groups_by_reagent.values():
        present.update(groups)
    out: list[str] = []
    for rule in rules["order_rules"]:
        if rule["group"] in present:
            with_g = rule["with"]
            if with_g == "any" or with_g in present:
                if rule["instruction"] not in out:
                    out.append(rule["instruction"])
    return out


def max_severity(severities: list[str]) -> str:
    if not severities:
        return "info"
    return max(severities, key=lambda s: _SEVERITY_ORDER[s])
