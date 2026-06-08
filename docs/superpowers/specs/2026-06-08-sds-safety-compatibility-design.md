# Design — SDS Safety & Compatibility Checker

**Project:** Agentic ELN — IP-safe R&D Lab Notebook (AI Developers Cohort 8, HUJI)
**Date:** 2026-06-08
**Authors:** Guillaume Bensoussan · Eyal Ben Ari
**Status:** Approved design — pending implementation plan

---

## 1. Goal

Add a safety layer that, once the chat or the Flow is prompted with a set of reagents,
runs all the relevant checks intelligently and returns **personalised, hard-to-ignore
warnings**:

- intrinsic hazards of each reagent (GHS classification, H-codes, pictograms),
- pairwise incompatibilities ("do not mix X with Y"),
- recommended **mixing order**,
- procedural warnings (e.g. *"add acid to water, never the reverse"*).

The chemist always remains the final decision-maker: the system **warns**, it never
hard-blocks an experiment.

## 2. Guiding principle

> **PubChem = intrinsic hazard data. A local rule engine = the logic. The LLM = wording only.**

The LLM never *decides* a hazard. It phrases a `SafetyReport` that has already been
computed deterministically. This is what makes a safety-critical instruction such as
"add acid to water" trustworthy rather than hallucinated. (The canonical rule is *add
acid TO water, never water to acid* — a single inverted word would be dangerous, hence
the rule base is authoritative, not the model.)

## 3. Data sources & demo data

- **Hazard data:** PubChem PUG REST / PUG View, looked up by **public CAS number**.
  - CAS → CID, then GHS Classification (pictograms, signal word, H-codes).
  - **Cache-first**: responses cached in `data/sds_cache.json`. Network is optional;
    tests run fully offline from the cache.
  - Throttle to ≤ 5 requests/s (PubChem policy). Any network error / missing CID /
    missing GHS section → `source="unknown"`, never a crash.
- **Demo dataset:** `data/generate_dataset.py` is reseeded with ~20 **real reagents and
  real CAS numbers** (e.g. toluene `108-88-3`, HCl `7647-01-0`, H₂O₂ `7722-84-1`,
  NaOH `1310-73-2`, NaH `7646-69-7`…). The proprietary / IP-sensitive subset keeps
  internal codes (`IntermediateX-###`) with an internal, non-resolvable CAS.
  `data/sds_cache.json` is pre-populated for the real CAS so the demo and the tests are
  deterministic and reproducible offline.

## 4. Module structure — `src/eln/safety/`

Four single-responsibility units:

| File | Responsibility | Depends on |
|---|---|---|
| `pubchem.py` | PUG REST/PUG View client: CAS → CID → GHS classification (H-codes, pictograms, signal word). Cache-first (`data/sds_cache.json`), optional network, ≤5 req/s throttle, graceful `source="unknown"` on any failure. | `httpx` |
| `reactivity.py` | Reactivity groups (acids, bases, oxidizers, water-reactive, flammables, pyrophorics, toxic-gas formers…). Mapping **H-code → group(s)**; curated per-CAS overrides for well-known cases (hypochlorite, ammonia, peroxides). **Pairwise matrix** `(groupA, groupB) → {severity, reason}`. **Mixing-order rules**. Pure, deterministic. Curated table lives in `data/reactivity_rules.json`. | — |
| `checker.py` | Orchestrator `check_safety(reagents) -> SafetyReport`: per-reagent hazards, pairwise incompatibilities, recommended addition order, procedural warnings, overall severity, unknowns. | `pubchem`, `reactivity` |
| `__init__.py` | Public surface: `check_safety`, `SafetyReport`. | — |

### H-code → reactivity group mapping (initial set)

| H-code(s) | Group |
|---|---|
| H314, H290 | corrosive / acid-or-base |
| H271, H272 | oxidizer |
| H260, H261 | water-reactive (flammable gas on contact with water) |
| H250 | pyrophoric |
| H224–H228, H242 | flammable |
| H300-series + curated | toxic-gas former (e.g. cyanides, sulfides, hypochlorite) |

Per-CAS curated overrides handle famous reactions not derivable from class combinations
alone (e.g. hypochlorite + ammonia → chloramine gas).

### Pairwise matrix (initial set, non-exhaustive)

| Group A | Group B | Severity | Reason |
|---|---|---|---|
| acid | base | critical | violent exothermic neutralisation |
| oxidizer | flammable | critical | fire / explosion risk |
| oxidizer | reducing agent | critical | violent redox |
| acid | hypochlorite | critical | toxic chlorine gas |
| hypochlorite | ammonia | critical | toxic chloramine gas |
| water-reactive | aqueous/water | critical | violent reaction, gas/heat |
| acid | cyanide/sulfide | critical | toxic HCN / H₂S gas |

### Mixing-order rules (initial set)

- If a strong acid (or strong base) is combined with water → **add the acid/base to the
  water slowly, never water to the reagent** (`add acid to water` — explicit, verified).
- Add oxidizers last and slowly.
- Water-reactive reagents: handle under inert atmosphere, no aqueous contact.

These rules produce `addition_order` and `procedural_warnings`.

## 5. Data model (`SafetyReport`)

```python
ReagentHazard:
    reagent_id: str
    name: str
    cas: str | None
    cid: int | None
    ghs_pictograms: list[str]
    h_codes: list[str]              # e.g. ["H314", "H271"]
    signal_word: str | None        # "Danger" | "Warning"
    reactivity_groups: list[str]   # e.g. ["oxidizer", "acid"]
    source: str                    # "pubchem" | "cache" | "unknown"

Incompatibility:
    a: str                         # reagent_id
    b: str                         # reagent_id
    groups: tuple[str, str]
    severity: str                  # "critical" | "warning" | "info"
    reason: str

SafetyReport:
    hazards: list[ReagentHazard]
    incompatibilities: list[Incompatibility]
    addition_order: list[str]      # recommended order (reagent_ids / names)
    procedural_warnings: list[str] # e.g. ["Add acid to water, never the reverse"]
    severity: str                  # max severity across the report
    unknowns: list[str]            # reagents with no public hazard data
```

**Input = an explicit list of reagents** (names / `reagent_id` / CAS), resolved against
the inventory + PubChem. Rationale: `parse_experiment` extracts only a single solvent,
whereas a mixing-safety check needs the full set. The list is supplied by (a) the
Streamlit multiselect, or (b) the reagents the EHS agent extracts from the chat.

## 6. Integration (follows the existing pattern)

**a) `tools.py` — `check_safety_tool`**: a CrewAI `@tool` wrapping
`checker.check_safety()`. The deterministic function stays callable directly by the Flow
*and* exposed to the agent (same dual pattern as the existing tools).

**b) EHS agent** (`src/eln/crew_safety.py`): `ehs_agent()` — role *"EHS / Safety
Officer"*, goal *"flag hazards, incompatibilities and the correct mixing order before any
bench work"*, tool `check_safety_tool`. It **phrases** a briefing strictly from the
`SafetyReport` (never invents hazards), with a deterministic template fallback when no
Claude key is set (same mechanism as `write_report`).

**c) Flow gate** (`flow.py`): a new `safety_check` step inserted **before** `analytics`.
It always computes the `SafetyReport` into a new `state.safety` field. **Safety never
halts the Flow** — prediction and report still run. Severity drives presentation only.
`state.errors` remains reserved for contract validation.

**d) Streamlit** — new **"Safety check"** tab: a multiselect of inventory reagents → a
button → display of per-reagent pictograms/H-codes, an incompatibility table coloured by
severity, the recommended addition order, and procedural warnings.

### Severity behaviour — warn, never block

The chemist is the final decision-maker. Nothing is forbidden; critical findings are made
**impossible to miss** via an acknowledge-to-proceed pattern.

- **`critical`** → a full-width, non-collapsed **red banner** at the top of everything
  (report + UI), worded unambiguously (e.g. *"⛔ SAFETY ALERT — HCl + NaOH: violent
  exothermic neutralisation…"*). In Streamlit the action is considered **confirmed only
  after** an explicit checkbox *"I have read these warnings and take responsibility"*; the
  acknowledgement is timestamped into the audit log (also serves as a diligence record).
  In the CLI/demo it renders as a prominent boxed ALERT block.
- **`warning`** → a visible orange callout, no acknowledgement required.
- **`info`** → a discreet note.

## 7. IP-safe handling (synergy with the existing sanitizer)

- Proprietary reagent names are **redacted by the existing `Sanitizer`** before any Claude
  call (the EHS briefing), and the call is logged to `artifacts/ip_audit_log.jsonl`.
- PubChem is queried **only by public CAS**. IP-sensitive `IntermediateX-###` reagents
  (internal, non-resolvable CAS) never leave the machine: they surface as
  `source="unknown"` with a note *"no public SDS — check the internal safety sheet"*.

## 8. Error handling

- No network / unknown CAS / missing GHS section → `source="unknown"`, never a crash; the
  report lists the affected reagents under `unknowns`.
- An unknown reagent never silently downgrades severity: it is reported explicitly so the
  chemist knows data is missing rather than safe.

## 9. Testing (`tests/test_safety.py`, offline via cache)

- H-code → reactivity-group mapping.
- Pairwise matrix: `HCl + NaOH` → critical; `toluene + H₂O₂` → critical;
  `hypochlorite + ammonia` → critical (toxic gas).
- **Mixing order**: strong acid + water produces exactly *"add acid to water, never the
  reverse"* — an explicit anti-inversion test.
- Unknown reagent handled gracefully (appears in `unknowns`, no crash).
- Flow gate: an incompatible pair sets `state.safety.severity == "critical"` and the report
  leads with the `SAFETY ALERT` banner, while the Flow still completes (no halt).
- PubChem client tested against cached fixtures (deterministic, offline).

## 10. Dependencies & files

- **New dependency:** `httpx` (PubChem client).
- **New files:** `src/eln/safety/{__init__,pubchem,reactivity,checker}.py`,
  `src/eln/crew_safety.py`, `data/sds_cache.json`, `data/reactivity_rules.json`,
  `tests/test_safety.py`.
- **Modified files:** `data/generate_dataset.py` (reseed with real reagents/CAS),
  `src/eln/tools.py` (`check_safety_tool`), `src/eln/flow.py` (`safety_check` step +
  `state.safety` field on `ELNState`), `app/streamlit_app.py` (Safety tab +
  acknowledgement), `README.md`, `pyproject.toml`. The EHS agent lives in the new
  `src/eln/crew_safety.py` (not `crew_intake.py`); `flow.py` is the only wiring point.

## 11. Out of scope (v2)

- Live patent / regulatory lookups beyond PubChem.
- Quantitative reaction-energy modelling.
- Hybrid enrichment for proprietary reagents from an internal SDS store.

## 12. Language

All code, comments, docstrings, UI strings and documents are in **English** (project rule).
