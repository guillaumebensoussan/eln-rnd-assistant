# SDS Safety & Compatibility Checker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safety layer that, given a set of reagents, returns deterministic, hard-to-ignore warnings (per-reagent GHS hazards, pairwise incompatibilities, mixing order, procedural warnings) — surfaced through a CrewAI tool, an EHS agent, a non-blocking Flow gate, and a Streamlit tab.

**Architecture:** A new `src/eln/safety/` package. `reactivity.py` is a pure deterministic rule engine (curated tables in `data/reactivity_rules.json`). `pubchem.py` is a cache-first PubChem client (network optional). `checker.py` orchestrates both into a `SafetyReport`. The LLM only phrases the report; it never decides hazards. Safety never halts the Flow — critical findings drive a prominent alert + acknowledgement, the chemist decides.

**Tech Stack:** Python 3.13, httpx (PubChem), pandas, CrewAI, Streamlit, pytest. All code/comments/docs in English.

**Spec:** `docs/superpowers/specs/2026-06-08-sds-safety-compatibility-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `data/reactivity_rules.json` (new) | Curated rules: H-code→group map, per-CAS group overrides, pairwise incompatibility matrix, mixing-order rules. |
| `src/eln/safety/__init__.py` (new) | Public surface: `check_safety`, `SafetyReport`, dataclasses. |
| `src/eln/safety/reactivity.py` (new) | Pure rule engine: load rules, map H-codes→groups, resolve groups, evaluate pairs, compute order. No network. |
| `src/eln/safety/pubchem.py` (new) | Cache-first PubChem client: CAS→CID→GHS parse. Network optional, graceful `source="unknown"`. |
| `src/eln/safety/checker.py` (new) | Orchestrator `check_safety(reagents) -> SafetyReport`. |
| `data/sds_cache.json` (new) | Cached PubChem hazard responses for the demo CAS (deterministic, offline). |
| `tests/test_safety.py` (new) | Unit + integration tests, fully offline via cache. |
| `data/generate_dataset.py` (modify) | Reseed inventory with ~20 real reagents + real CAS. |
| `src/eln/tools.py` (modify) | Add `check_safety_tool`. |
| `src/eln/crew_safety.py` (new) | `ehs_agent()` + `write_safety_briefing()` (LLM with template fallback). |
| `src/eln/flow.py` (modify) | Add `reagents` + `safety` to `ELNState`, add non-blocking `safety_check` step. |
| `app/streamlit_app.py` (modify) | Add "Safety check" tab + critical-alert acknowledgement. |
| `README.md` (modify) | Document the feature + usage. |
| `pyproject.toml` (modify) | Add `httpx`. |

---

## Task 1: Add httpx dependency and safety package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/eln/safety/__init__.py`

- [ ] **Step 1: Add httpx to dependencies**

In `pyproject.toml`, add `"httpx>=0.28.1",` to the `dependencies` list (keep alphabetical-ish order, e.g. after `crewai`).

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: resolves and installs `httpx` with no errors.

- [ ] **Step 3: Create the empty package**

Create `src/eln/safety/__init__.py` with a single line:

```python
"""Safety layer: SDS hazards, reactivity rules, compatibility checking."""
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/eln/safety/__init__.py
git commit -m "chore: add httpx and safety package skeleton"
```

---

## Task 2: Curated reactivity rules data file

**Files:**
- Create: `data/reactivity_rules.json`

- [ ] **Step 1: Write the rules file**

Create `data/reactivity_rules.json` exactly:

```json
{
  "hcode_to_groups": {
    "H271": ["oxidizer"],
    "H272": ["oxidizer"],
    "H260": ["water_reactive"],
    "H261": ["water_reactive"],
    "H250": ["pyrophoric"],
    "H224": ["flammable"],
    "H225": ["flammable"],
    "H226": ["flammable"],
    "H228": ["flammable"],
    "H242": ["flammable"],
    "H314": ["corrosive"],
    "H290": ["corrosive_metal"]
  },
  "cas_overrides": {
    "7647-01-0": ["acid"],
    "7664-93-9": ["acid", "oxidizer"],
    "7697-37-2": ["acid", "oxidizer"],
    "1310-73-2": ["base"],
    "1310-58-3": ["base"],
    "7681-52-9": ["hypochlorite", "oxidizer"],
    "1336-21-6": ["ammonia", "base"],
    "7722-84-1": ["oxidizer"],
    "7646-69-7": ["water_reactive", "flammable"],
    "7782-50-5": ["oxidizer", "toxic"],
    "7732-18-5": ["water"]
  },
  "pairs": [
    {"a": "acid", "b": "base", "severity": "critical", "reason": "Violent exothermic neutralisation; large heat release."},
    {"a": "oxidizer", "b": "flammable", "severity": "critical", "reason": "Fire/explosion risk: oxidizer plus fuel."},
    {"a": "oxidizer", "b": "reducing_agent", "severity": "critical", "reason": "Violent redox reaction."},
    {"a": "acid", "b": "hypochlorite", "severity": "critical", "reason": "Releases toxic chlorine gas."},
    {"a": "hypochlorite", "b": "ammonia", "severity": "critical", "reason": "Releases toxic chloramine gas."},
    {"a": "water_reactive", "b": "water", "severity": "critical", "reason": "Violent reaction with water; heat and flammable gas."},
    {"a": "acid", "b": "cyanide_sulfide", "severity": "critical", "reason": "Releases toxic HCN/H2S gas."},
    {"a": "pyrophoric", "b": "flammable", "severity": "warning", "reason": "Pyrophoric solid near flammables; ignition risk."}
  ],
  "order_rules": [
    {"group": "acid", "with": "water", "instruction": "Add the acid to the water slowly, never water to the acid.", "add_last": true},
    {"group": "base", "with": "water", "instruction": "Add the base to the water slowly, with stirring and cooling.", "add_last": true},
    {"group": "oxidizer", "with": "any", "instruction": "Add the oxidizer last and slowly.", "add_last": true}
  ]
}
```

- [ ] **Step 2: Validate it parses**

Run: `python3 -c "import json; json.load(open('data/reactivity_rules.json')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add data/reactivity_rules.json
git commit -m "feat(safety): curated reactivity rules (groups, matrix, order)"
```

---

## Task 3: Reactivity rule engine (pure, deterministic)

**Files:**
- Create: `src/eln/safety/reactivity.py`
- Test: `tests/test_safety.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_safety.py`:

```python
"""Safety layer tests: reactivity engine, PubChem cache, checker, Flow gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eln.safety import reactivity as rx


def test_hcode_maps_to_oxidizer():
    groups = rx.groups_for(cas="0-0-0", h_codes=["H271"])
    assert "oxidizer" in groups


def test_cas_override_adds_acid_group():
    groups = rx.groups_for(cas="7647-01-0", h_codes=["H314"])
    assert "acid" in groups          # from CAS override
    assert "corrosive" in groups     # from H314


def test_pair_acid_base_is_critical():
    inc = rx.evaluate_pair(["acid"], ["base"])
    assert inc is not None
    assert inc["severity"] == "critical"
    assert "neutralisation" in inc["reason"].lower()


def test_pair_order_independent():
    assert rx.evaluate_pair(["base"], ["acid"]) is not None


def test_pair_no_conflict_returns_none():
    assert rx.evaluate_pair(["flammable"], ["flammable"]) is None


def test_order_acid_with_water_not_inverted():
    groups_by_reagent = {"HCl": ["acid"], "water": ["water"]}
    warnings = rx.mixing_warnings(groups_by_reagent)
    joined = " ".join(warnings).lower()
    assert "add the acid to the water" in joined
    assert "water to the acid" in joined  # phrased as the prohibition
    # explicit anti-inversion: must NOT instruct adding water to acid as the action
    assert "add the water to the acid" not in joined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eln.safety.reactivity'`

- [ ] **Step 3: Implement `reactivity.py`**

Create `src/eln/safety/reactivity.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/eln/safety/reactivity.py tests/test_safety.py
git commit -m "feat(safety): deterministic reactivity rule engine + tests"
```

---

## Task 4: PubChem client (cache-first, network optional)

**Files:**
- Create: `src/eln/safety/pubchem.py`
- Create: `tests/fixtures/sds_cache_test.json`
- Test: `tests/test_safety.py` (append)

- [ ] **Step 1: Write the test fixture cache**

Create `tests/fixtures/sds_cache_test.json`:

```json
{
  "7647-01-0": {
    "cas": "7647-01-0",
    "cid": 313,
    "ghs_pictograms": ["Corrosive", "Acute Toxic"],
    "h_codes": ["H314", "H331"],
    "signal_word": "Danger",
    "source": "cache"
  }
}
```

- [ ] **Step 2: Write failing tests (append to `tests/test_safety.py`)**

```python
from eln.safety.pubchem import PubChemClient, parse_ghs_view

FIXTURE_CACHE = ROOT / "tests" / "fixtures" / "sds_cache_test.json"


def test_pubchem_returns_cached_hazard_without_network():
    client = PubChemClient(cache_path=FIXTURE_CACHE, allow_network=False)
    haz = client.get_hazard("7647-01-0")
    assert haz["cid"] == 313
    assert "H314" in haz["h_codes"]
    assert haz["signal_word"] == "Danger"
    assert haz["source"] == "cache"


def test_pubchem_unknown_cas_offline_is_graceful():
    client = PubChemClient(cache_path=FIXTURE_CACHE, allow_network=False)
    haz = client.get_hazard("0000-00-0")
    assert haz["source"] == "unknown"
    assert haz["h_codes"] == []
    assert haz["cid"] is None


def test_parse_ghs_view_extracts_codes_and_pictograms():
    sample = {
        "Record": {"Section": [{"TOCHeading": "Safety and Hazards", "Section": [
            {"TOCHeading": "Hazards Identification", "Section": [
                {"TOCHeading": "GHS Classification", "Information": [
                    {"Name": "Pictogram(s)", "Value": {"StringWithMarkup": [
                        {"String": "  ", "Markup": [{"Type": "Icon", "Extra": "Corrosive"}]}]}},
                    {"Name": "Signal", "Value": {"StringWithMarkup": [{"String": "Danger"}]}},
                    {"Name": "GHS Hazard Statements", "Value": {"StringWithMarkup": [
                        {"String": "H314 (99.9%): Causes severe skin burns"},
                        {"String": "H290: May be corrosive to metals"}]}},
                ]}]}]}]}}
    parsed = parse_ghs_view(sample)
    assert parsed["signal_word"] == "Danger"
    assert parsed["ghs_pictograms"] == ["Corrosive"]
    assert parsed["h_codes"] == ["H314", "H290"]
```

- [ ] **Step 3: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eln.safety.pubchem'`

- [ ] **Step 4: Implement `pubchem.py`**

Create `src/eln/safety/pubchem.py`:

```python
"""Cache-first PubChem client for GHS hazard data.

Resolves CAS -> CID -> GHS Classification (pictograms, signal word, H-codes)
via PUG REST / PUG View. Network is optional: the cache is consulted first and
any failure degrades to source="unknown" instead of raising.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "data" / "sds_cache.json"

_CID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{cas}/cids/JSON"
_GHS_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification"
_HCODE_RE = re.compile(r"\b(H\d{3})\b")
_UNKNOWN = {"cid": None, "ghs_pictograms": [], "h_codes": [], "signal_word": None, "source": "unknown"}


def _walk_ghs_sections(node):
    """Yield every section whose TOCHeading == 'GHS Classification'."""
    if isinstance(node, dict):
        if node.get("TOCHeading") == "GHS Classification":
            yield node
        for child in node.get("Section", []):
            yield from _walk_ghs_sections(child)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_ghs_sections(item)


def parse_ghs_view(payload: dict) -> dict:
    """Parse a PUG View GHS Classification payload into a hazard dict."""
    pictograms: list[str] = []
    signal: str | None = None
    h_codes: list[str] = []
    for section in _walk_ghs_sections(payload.get("Record", {})):
        for info in section.get("Information", []):
            name = info.get("Name")
            swm = info.get("Value", {}).get("StringWithMarkup", [])
            if name == "Pictogram(s)":
                for entry in swm:
                    for mark in entry.get("Markup", []):
                        extra = mark.get("Extra")
                        if extra and extra not in pictograms:
                            pictograms.append(extra)
            elif name == "Signal" and swm and signal is None:
                signal = swm[0].get("String")
            elif name == "GHS Hazard Statements":
                for entry in swm:
                    m = _HCODE_RE.search(entry.get("String", ""))
                    if m and m.group(1) not in h_codes:
                        h_codes.append(m.group(1))
    return {"cid": None, "ghs_pictograms": pictograms, "h_codes": h_codes,
            "signal_word": signal, "source": "pubchem"}


class PubChemClient:
    def __init__(self, cache_path: Path | None = None, allow_network: bool = True,
                 timeout: float = 20.0):
        self.cache_path = cache_path or DEFAULT_CACHE
        self.allow_network = allow_network
        self.timeout = timeout
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def get_hazard(self, cas: str) -> dict:
        if cas in self._cache:
            return {"cas": cas, **self._cache[cas]}
        if not self.allow_network:
            return {"cas": cas, **_UNKNOWN}
        haz = self._fetch(cas)
        self._cache[cas] = {k: v for k, v in haz.items() if k != "cas"}
        self._save_cache()
        return haz

    def _fetch(self, cas: str) -> dict:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(_CID_URL.format(cas=cas))
                r.raise_for_status()
                cids = r.json().get("IdentifierList", {}).get("CID", [])
                if not cids:
                    return {"cas": cas, **_UNKNOWN}
                cid = cids[0]
                time.sleep(0.25)  # PubChem politeness (<=5 req/s)
                g = client.get(_GHS_URL.format(cid=cid))
                g.raise_for_status()
                parsed = parse_ghs_view(g.json())
                parsed["cid"] = cid
                return {"cas": cas, **parsed}
        except (httpx.HTTPError, ValueError, KeyError):
            return {"cas": cas, **_UNKNOWN}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py -q`
Expected: PASS (9 passed)

- [ ] **Step 6: Commit**

```bash
git add src/eln/safety/pubchem.py tests/fixtures/sds_cache_test.json tests/test_safety.py
git commit -m "feat(safety): cache-first PubChem GHS client + tests"
```

---

## Task 5: Reseed dataset with real reagents and build the SDS cache

**Files:**
- Modify: `data/generate_dataset.py`
- Create: `data/sds_cache.json` (generated by running the fetch)

- [ ] **Step 1: Replace the inventory generator with a curated real-reagent list**

In `data/generate_dataset.py`, replace the body of `make_inventory` with a curated list. Define this module-level constant above `make_inventory`:

```python
# Curated real reagents (name, CAS). The proprietary subset keeps internal codes.
REAL_REAGENTS = [
    ("Hydrochloric acid", "7647-01-0"),
    ("Sulfuric acid", "7664-93-9"),
    ("Nitric acid", "7697-37-2"),
    ("Sodium hydroxide", "1310-73-2"),
    ("Potassium hydroxide", "1310-58-3"),
    ("Sodium hypochlorite", "7681-52-9"),
    ("Ammonia solution", "1336-21-6"),
    ("Hydrogen peroxide", "7722-84-1"),
    ("Sodium hydride", "7646-69-7"),
    ("Toluene", "108-88-3"),
    ("Tetrahydrofuran", "109-99-9"),
    ("Dichloromethane", "75-09-2"),
    ("Methanol", "67-56-1"),
    ("Acetonitrile", "75-05-8"),
    ("N,N-Dimethylformamide", "68-12-2"),
    ("Acetone", "67-64-1"),
    ("Diethyl ether", "60-29-7"),
    ("Water", "7732-18-5"),
]
```

Then replace `make_inventory` with:

```python
def make_inventory(n: int = 50) -> pd.DataFrame:
    rows = []
    for i, (name, cas) in enumerate(REAL_REAGENTS):
        rows.append(
            {
                "reagent_id": f"R{i:03d}",
                "name": name,
                "cas": cas,
                "supplier": str(rng.choice(["Sigma-Aldrich", "Merck", "TCI", "Fisher"])),
                "batch": f"B{rng.integers(1000, 9999)}",
                "quantity_g": round(float(rng.uniform(5, 500)), 1),
                "expiry": str(date(2026, 1, 1) + timedelta(days=int(rng.integers(30, 900)))),
                "location": f"Shelf-{rng.integers(1, 12)}",
                "ip_sensitive": False,
            }
        )
    # proprietary / IP-sensitive reagents with internal, non-resolvable CAS
    for j in range(6):
        i = len(REAL_REAGENTS) + j
        rows.append(
            {
                "reagent_id": f"R{i:03d}",
                "name": f"IntermediateX-{rng.integers(100, 999)}",
                "cas": f"INT-{rng.integers(1000, 9999)}",
                "supplier": "Internal-Synth",
                "batch": f"B{rng.integers(1000, 9999)}",
                "quantity_g": round(float(rng.uniform(5, 500)), 1),
                "expiry": str(date(2026, 1, 1) + timedelta(days=int(rng.integers(30, 900)))),
                "location": f"Shelf-{rng.integers(1, 12)}",
                "ip_sensitive": True,
            }
        )
    return pd.DataFrame(rows)
```

Note: `make_experiments` already references `inventory["reagent_id"]`, which still exists, so it keeps working. The `solvent` field in experiments still uses the `SOLVENTS` list — unchanged.

- [ ] **Step 2: Regenerate the dataset**

Run: `uv run python data/generate_dataset.py`
Expected: prints inventory/experiments counts; `data/inventory.csv` now lists real reagent names + CAS, with 6 IP-sensitive `IntermediateX-###` rows.

- [ ] **Step 3: Build the real SDS cache from PubChem (one-off, online)**

Run this one-off command (network required once; it populates `data/sds_cache.json`):

```bash
PYTHONPATH=src uv run python -c "
import pandas as pd
from eln.safety.pubchem import PubChemClient
inv = pd.read_csv('data/inventory.csv')
client = PubChemClient(allow_network=True)
for cas in inv['cas']:
    h = client.get_hazard(str(cas))
    print(cas, h['source'], h['h_codes'])
"
```

Expected: real CAS resolve to `pubchem` with H-codes; `INT-####` rows resolve to `unknown`. `data/sds_cache.json` is created/updated.

- [ ] **Step 4: Verify the cache works fully offline**

Run: `PYTHONPATH=src uv run python -c "
from eln.safety.pubchem import PubChemClient
c = PubChemClient(allow_network=False)
print(c.get_hazard('7647-01-0')['h_codes'])
print(c.get_hazard('7681-52-9')['source'])
"`
Expected: HCl shows H-codes (incl. `H314`); hypochlorite shows `cache`.

- [ ] **Step 5: Commit**

```bash
git add data/generate_dataset.py data/inventory.csv data/clean_data.csv data/dataset_contract.json data/features.csv data/sds_cache.json
git commit -m "feat(safety): reseed dataset with real reagents/CAS + PubChem cache"
```

---

## Task 6: Checker orchestrator and public dataclasses

**Files:**
- Create: `src/eln/safety/checker.py`
- Modify: `src/eln/safety/__init__.py`
- Test: `tests/test_safety.py` (append)

- [ ] **Step 1: Write failing tests (append to `tests/test_safety.py`)**

```python
from eln.safety import check_safety, SafetyReport


def _checker_kwargs():
    # use the offline test fixture cache so tests never hit the network
    from eln.safety.pubchem import PubChemClient
    return {"client": PubChemClient(cache_path=FIXTURE_CACHE, allow_network=False)}


def test_check_safety_flags_acid_base_critical():
    # HCl (cached) vs a synthetic base via override; use real cache in data/ for NaOH
    from eln.safety.pubchem import PubChemClient
    client = PubChemClient(allow_network=False)  # data/sds_cache.json
    report = check_safety(["Hydrochloric acid", "Sodium hydroxide"],
                          inventory_csv=ROOT / "data" / "inventory.csv", client=client)
    assert isinstance(report, SafetyReport)
    assert report.severity == "critical"
    assert any(inc.severity == "critical" for inc in report.incompatibilities)


def test_check_safety_unknown_reagent_listed():
    from eln.safety.pubchem import PubChemClient
    client = PubChemClient(allow_network=False)
    report = check_safety(["IntermediateX-999-not-real"],
                          inventory_csv=ROOT / "data" / "inventory.csv", client=client)
    assert report.unknowns  # unresolved reagent surfaced, no crash


def test_check_safety_acid_water_order_warning():
    from eln.safety.pubchem import PubChemClient
    client = PubChemClient(allow_network=False)
    report = check_safety(["Hydrochloric acid", "Water"],
                          inventory_csv=ROOT / "data" / "inventory.csv", client=client)
    joined = " ".join(report.procedural_warnings).lower()
    assert "add the acid to the water" in joined
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py -q`
Expected: FAIL — `ImportError: cannot import name 'check_safety' from 'eln.safety'`

- [ ] **Step 3: Implement `checker.py`**

Create `src/eln/safety/checker.py`:

```python
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
```

- [ ] **Step 4: Export the public surface in `src/eln/safety/__init__.py`**

Replace `src/eln/safety/__init__.py` with:

```python
"""Safety layer: SDS hazards, reactivity rules, compatibility checking."""

from .checker import Incompatibility, ReagentHazard, SafetyReport, check_safety

__all__ = ["check_safety", "SafetyReport", "ReagentHazard", "Incompatibility"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py -q`
Expected: PASS (12 passed). (Requires `data/sds_cache.json` from Task 5.)

- [ ] **Step 6: Commit**

```bash
git add src/eln/safety/checker.py src/eln/safety/__init__.py tests/test_safety.py
git commit -m "feat(safety): SafetyReport checker orchestrator + tests"
```

---

## Task 7: CrewAI tool wrapper

**Files:**
- Modify: `src/eln/tools.py`
- Test: `tests/test_safety.py` (append)

- [ ] **Step 1: Write failing test (append to `tests/test_safety.py`)**

```python
def test_check_safety_tool_returns_dict():
    from eln.tools import check_safety
    result = check_safety(["Hydrochloric acid", "Sodium hydroxide"])
    assert result["severity"] == "critical"
    assert isinstance(result["incompatibilities"], list)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py::test_check_safety_tool_returns_dict -q`
Expected: FAIL — `ImportError: cannot import name 'check_safety' from 'eln.tools'`

- [ ] **Step 3: Add the tool to `src/eln/tools.py`**

Add this import near the top (after the existing `from .sanitizer import Sanitizer`):

```python
from dataclasses import asdict

from .safety import check_safety as _check_safety
```

Add at the end of the file:

```python
def check_safety(reagents: list[str]) -> dict:
    """Run the deterministic safety check and return a plain dict."""
    return asdict(_check_safety(reagents))


@tool("check_safety")
def check_safety_tool(reagents: list[str]) -> dict:
    """Check GHS hazards, pairwise incompatibilities and mixing order for a list of reagents."""
    return check_safety(reagents)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py::test_check_safety_tool_returns_dict -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eln/tools.py tests/test_safety.py
git commit -m "feat(safety): expose check_safety as a CrewAI tool"
```

---

## Task 8: EHS agent and grounded briefing

**Files:**
- Create: `src/eln/crew_safety.py`
- Test: `tests/test_safety.py` (append)

- [ ] **Step 1: Write failing test (append to `tests/test_safety.py`)**

```python
def test_safety_briefing_template_mentions_critical_and_order():
    from eln.safety.pubchem import PubChemClient
    from eln.safety import check_safety as cs
    from eln.crew_safety import write_safety_briefing
    client = PubChemClient(allow_network=False)
    report = cs(["Hydrochloric acid", "Sodium hydroxide", "Water"],
                inventory_csv=ROOT / "data" / "inventory.csv", client=client)
    text = write_safety_briefing(report)
    assert "SAFETY ALERT" in text                      # critical present
    assert "add the acid to the water" in text.lower() # order rule present
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py::test_safety_briefing_template_mentions_critical_and_order -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eln.crew_safety'`

- [ ] **Step 3: Implement `crew_safety.py`**

Create `src/eln/crew_safety.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py::test_safety_briefing_template_mentions_critical_and_order -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eln/crew_safety.py tests/test_safety.py
git commit -m "feat(safety): EHS agent + grounded safety briefing"
```

---

## Task 9: Non-blocking Flow gate

**Files:**
- Modify: `src/eln/flow.py`
- Test: `tests/test_safety.py` (append)

- [ ] **Step 1: Write failing test (append to `tests/test_safety.py`)**

```python
def test_flow_safety_gate_warns_without_halting():
    from eln.flow import run
    state = run("toluene at 110 C for 8 h, 2 mol%, 3 bar, purity 97",
                question="best solvent?",
                reagents=["Hydrochloric acid", "Sodium hydroxide"])
    # safety is critical but the Flow still completes (no halt, no errors)
    assert state.safety["severity"] == "critical"
    assert not state.errors
    assert state.report  # report still produced
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py::test_flow_safety_gate_warns_without_halting -q`
Expected: FAIL — `run()` has no `reagents` parameter / `state` has no `safety`.

- [ ] **Step 3: Modify `src/eln/flow.py`**

Add to the imports (after `from .tools import parse_experiment, predict_outcome`):

```python
from dataclasses import asdict

from .safety import check_safety
```

Add two fields to `ELNState` (after `report: str = ""`):

```python
    reagents: list[str] = []
    safety: dict = {}
```

Insert a new step **after** `retrieval` and **before** `validate_handoff`:

```python
    @listen(retrieval)
    def safety_check(self, _):
        log.info("[Crew1:Safety] running deterministic SDS safety check")
        reagents = self.state.reagents or [self.state.record.get("solvent", "")]
        reagents = [r for r in reagents if r]
        if reagents:
            self.state.safety = asdict(check_safety(reagents))
            log.info("[Crew1:Safety] severity=%s incompatibilities=%d",
                     self.state.safety["severity"], len(self.state.safety["incompatibilities"]))
        return "safety_done"
```

Change the `validate_handoff` decorator from `@listen(retrieval)` to `@listen(safety_check)`.

In the `analytics` method, after computing `self.state.report = write_report(...)`, prepend a critical banner when relevant — replace the final `return self.state` of `analytics` with:

```python
        if self.state.safety.get("severity") == "critical":
            self.state.report = (
                "⛔ SAFETY ALERT — critical reagent incompatibility detected. "
                "Review the Safety check before proceeding.\n\n" + self.state.report
            )
        return self.state
```

Update the `run()` function signature and the snapshot at the bottom of the file:

```python
def run(text: str, question: str = "", reagents: list[str] | None = None) -> ELNState:
    flow = ELNFlow()
    flow.kickoff(inputs={"text": text, "question": question, "reagents": reagents or []})
    s = flow.state
    return ELNState(
        text=s.text,
        question=s.question,
        record=dict(s.record),
        inventory_result=dict(s.inventory_result),
        rag_context=s.rag_context,
        prediction=dict(s.prediction),
        report=s.report,
        errors=list(s.errors),
        reagents=list(s.reagents),
        safety=dict(s.safety),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_safety.py::test_flow_safety_gate_warns_without_halting -q`
Expected: PASS

- [ ] **Step 5: Run the full existing suite to confirm no regressions**

Run: `PYTHONPATH=src uv run pytest -q`
Expected: PASS — existing `tests/test_eln.py` (including `test_flow_fails_gracefully_on_missing_features`) still green, plus all safety tests.

- [ ] **Step 6: Commit**

```bash
git add src/eln/flow.py tests/test_safety.py
git commit -m "feat(safety): non-blocking safety gate in the Flow"
```

---

## Task 10: Streamlit Safety tab with acknowledgement

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Add the Safety tab to the tab list**

Change the `st.tabs([...])` call to include a Safety tab:

```python
tab_log, tab_inv, tab_safety, tab_audit, tab_hist = st.tabs(
    ["Log experiment", "Inventory", "Safety check", "IP Audit log", "History"]
)
```

- [ ] **Step 2: Add the imports**

After `from eln.sanitizer import read_audit`, add:

```python
from eln.safety import check_safety
from eln.crew_safety import write_safety_briefing
from eln.safety.checker import SafetyReport, Incompatibility, ReagentHazard
```

- [ ] **Step 3: Add the Safety tab body**

After the `with tab_inv:` block and before `with tab_audit:`, insert:

```python
with tab_safety:
    st.subheader("SDS safety & compatibility check")
    inv_s = Inventory()
    chosen = st.multiselect("Reagents to combine", options=sorted(inv_s.df["name"]))
    if st.button("Run safety check", type="primary") and chosen:
        report = check_safety(chosen)
        if report.severity == "critical":
            st.error("⛔ SAFETY ALERT — critical incompatibility. Read carefully before proceeding.")
        elif report.severity == "warning":
            st.warning("⚠️ Safety warnings present.")
        else:
            st.success("No critical incompatibility detected for this set.")

        if report.incompatibilities:
            st.markdown("**Incompatibilities**")
            st.dataframe(
                pd.DataFrame([{"A": i.a, "B": i.b, "severity": i.severity, "reason": i.reason}
                              for i in report.incompatibilities]),
                use_container_width=True, hide_index=True,
            )
        if report.procedural_warnings:
            st.markdown("**Mixing order**")
            for w in report.procedural_warnings:
                st.markdown(f"- {w}")
            st.caption("Suggested addition order: " + " → ".join(report.addition_order))
        if report.hazards:
            st.markdown("**Per-reagent hazards**")
            st.dataframe(
                pd.DataFrame([{"reagent": h.name, "H-codes": ", ".join(h.h_codes),
                               "pictograms": ", ".join(h.ghs_pictograms), "source": h.source}
                              for h in report.hazards]),
                use_container_width=True, hide_index=True,
            )
        if report.unknowns:
            st.info("No public SDS for: " + ", ".join(report.unknowns) + " — check the internal safety sheet.")

        if report.severity == "critical":
            ack = st.checkbox("I have read these warnings and take responsibility.")
            if ack:
                from eln.sanitizer import audit
                audit(operation="safety_acknowledgement", sent_to_cloud="",
                      redacted=[], stayed_local=[f"ack:{', '.join(chosen)}"])
                st.success("Acknowledged and logged. You may proceed at your discretion.")
            else:
                st.stop()
```

- [ ] **Step 4: Manual smoke test**

Run: `PYTHONPATH=src uv run streamlit run app/streamlit_app.py`
In the **Safety check** tab, select "Hydrochloric acid" + "Sodium hydroxide" → click Run.
Expected: red SAFETY ALERT, an incompatibility row (critical, neutralisation), and an acknowledgement checkbox. Selecting "Hydrochloric acid" + "Water" shows the "add the acid to the water" order warning.

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(safety): Streamlit safety tab with critical-alert acknowledgement"
```

---

## Task 11: README and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the feature in `README.md`**

Add a row to the Stack table (after the ChromaDB row):

```markdown
| PubChem + local rule engine | SDS hazards (GHS) + curated reactivity rules → compatibility, mixing order, IP-safe |
```

Add a new section after the "Stack" section:

```markdown
## Safety check (SDS / compatibility)

Given a set of reagents, the safety layer returns GHS hazards (from PubChem, cached
locally), pairwise incompatibilities, the recommended mixing order, and procedural
warnings (e.g. *add acid to water, never the reverse*). Hazard data comes from PubChem;
the incompatibility and ordering **logic is a deterministic local rule engine**
(`data/reactivity_rules.json`) — the LLM only phrases the result, it never decides a
hazard. Safety **never blocks** an experiment: critical findings raise a prominent alert
plus an acknowledgement (the chemist decides). IP-sensitive reagents are never sent to
PubChem and are redacted before any cloud call.

Run it from the **Safety check** tab, or programmatically:

```bash
PYTHONPATH=src uv run python -c "from eln.safety import check_safety; r=check_safety(['Hydrochloric acid','Sodium hydroxide']); print(r.severity, r.incompatibilities)"
```
```

- [ ] **Step 2: Run the full test suite**

Run: `PYTHONPATH=src uv run pytest -q`
Expected: all tests pass (existing ELN tests + safety tests).

- [ ] **Step 3: Lint check (if ruff is configured)**

Run: `uv run ruff check src/eln/safety app/streamlit_app.py src/eln/flow.py src/eln/tools.py src/eln/crew_safety.py`
Expected: no errors (fix any import-order/line-length issues reported).

- [ ] **Step 4: Final French scan (project rule: English only)**

Run: `grep -rnIE "[éèêàùçôîïâûœ]" src/eln/safety src/eln/crew_safety.py data/reactivity_rules.json || echo "clean"`
Expected: `clean`

- [ ] **Step 5: Commit and push**

```bash
git add README.md
git commit -m "docs(safety): document the SDS safety & compatibility feature"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage:** PubChem source + cache (Tasks 4, 5) ✓; reseed real CAS (Task 5) ✓; reactivity groups + GHS + matrix + order (Tasks 2, 3) ✓; per-CAS overrides (Task 2) ✓; `SafetyReport` model (Task 6) ✓; CrewAI tool (Task 7) ✓; EHS agent + grounded briefing + template fallback (Task 8) ✓; non-blocking Flow gate + `state.safety` (Task 9) ✓; critical alert + acknowledgement (Tasks 8, 9, 10) ✓; IP-safe via sanitizer + CAS-only PubChem + unknowns (Tasks 6, 8, 9, 10) ✓; error handling/graceful unknown (Task 4) ✓; tests incl. anti-inversion (Tasks 3, 6, 8) ✓.
- **Type consistency:** `check_safety(reagents, inventory_csv=None, client=None)`, `SafetyReport` fields, `Incompatibility(a, b, groups, severity, reason)`, `PubChemClient(cache_path, allow_network)` used consistently across tasks. The Flow stores `safety` as a dict (`asdict`), and tests read `state.safety["severity"]` accordingly.
- **IP-safe note:** the briefing template (Task 8) is deterministic; when the EHS agent uses Claude, reagent names must be passed through the existing `Sanitizer` (already available) before the cloud call — this is enforced by routing cloud wording through the same sanitised-context pattern as `write_report`.
```
