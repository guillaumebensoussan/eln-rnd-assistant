# Agentic ELN — IP-safe R&D Lab Notebook

Final project — *AI Developers Cohort 8 (HUJI)*. Alternative track approved by Jonathan Zouari.
Co-authored by Guillaume Bensoussan · Eyal Ben Ari.

An **agentic Electronic Lab Notebook** for industrial R&D: log experiments in natural
language, manage reagent inventory, RAG Q&A over the lab history, predict yield/QC, and
generate a report — with an **IP-safe** guarantee: raw data stays local, only a sanitised
context ever reaches the cloud LLM.

## Architecture — two-crew CrewAI Flow

```
Crew 1 — Lab Intake & Knowledge          Validation gate           Crew 2 — Analytics & Reporting
  Intake (structuring)          ─┐                                    Prediction (scikit-learn)
  Inventory (stock, Pandas)      ├─►  contract + features OK  ──►      IP-Audit (sanitisation + log)
  Retrieval (RAG, ChromaDB)     ─┘    else: graceful failure          Synthesis/Report (Claude*)
```

The Flow (`src/eln/flow.py`) chains the steps **deterministically**, **validates** the
`dataset_contract.json` and the required features before modelling, **logs** every step and
**fails cleanly** (FLOW HALTED) if validation breaks.

\* The report uses the LLM configured via `ELN_MODEL` (defaults to Claude) if a key is set; otherwise it falls back to a local template.

## Stack

| Component | Role |
|---|---|
| CrewAI | Orchestrates the 6 agents + Flow (handoff, validation, graceful failure) |
| scikit-learn | `yield_pct` regression (LinearRegression vs RandomForest) + `qc_pass` classification |
| ChromaDB + local embeddings | RAG over experiments + papers/patents, **no raw data to the cloud** |
| PubChem + local rule engine | SDS hazards (GHS) + curated reactivity rules → compatibility, mixing order, IP-safe |
| Claude API | Complex reasoning (report) — sanitised context only |
| Pandas | Dataset, inventory, stock deductions |
| Streamlit | Lab UI: entry, inventory dashboard, audit log, history, assistant chat |
| uv / Git+GitHub | Environment + versioning (Pull Request workflow) |

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

## Chat assistant

The **Assistant** tab is a multi-turn chat over all ELN capabilities
(safety, prediction, history, inventory).

- **With a cloud key**: a CrewAI orchestrator (plus the EHS agent) answers,
  using a sanitised tool belt — the cloud LLM only ever sees aliases
  (`REAGENT_A1`), tool calls run locally on real names, results are scrubbed
  before returning, and every turn lands in the IP audit log.
- **Without a key**: a deterministic keyword router answers from local
  engines and templates. Nothing leaves the machine.

### Choosing the LLM provider

Set `ELN_MODEL` in `.env` (LiteLLM format) with the matching key:

| Provider | ELN_MODEL | Key |
|---|---|---|
| Anthropic (default) | `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| Ollama (fully local) | `ollama/gemma4:latest` | none — needs `ollama serve` with the model pulled |

With an Ollama model, the agentic chat runs entirely on this machine: the
sanitiser still applies (defense in depth), but the audit log records
`chars_sent_to_cloud = 0` because LLM inference never leaves the workstation.

## Installation

```bash
uv sync
ollama pull nomic-embed-text   # optional; otherwise Chroma's built-in local embedder
cp .env.example .env           # add ANTHROPIC_API_KEY (or GEMINI_API_KEY / OPENAI_API_KEY) and ELN_MODEL (optional)
```

## Usage

```bash
uv run python data/generate_dataset.py     # generate synthetic dataset + docs (deterministic)
uv run python src/eln/ml/train.py          # train the models -> artifacts/
PYTHONPATH=src uv run python -m eln --demo  # run the Flow end-to-end from the CLI
PYTHONPATH=src uv run streamlit run app/streamlit_app.py   # UI
PYTHONPATH=src uv run pytest -q             # tests + RAG eval
```

## Produced artifacts (HUJI grading grid)

- `data/clean_data.csv`, `data/dataset_contract.json`, `data/features.csv`
- `artifacts/eda_report.html`, `artifacts/model.pkl`
- `artifacts/evaluation_report.md`, `artifacts/model_card.md`
- `artifacts/ip_audit_log.jsonl` (local-vs-cloud evidence)

## Scope

v1 = defensible core: Flow + RAG + inventory + ML + report + IP-audit.
Optional (v2): voice input (Whisper), local LLM router (Ollama), live patent search (PatentsView).

## Course deliverables

- [x] Repo + code + artifacts + README
- [ ] 10–12 slide presentation / demo
- [ ] Video ≤ 5 min
