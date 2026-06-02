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

\* The report uses Claude if `ANTHROPIC_API_KEY` is set; otherwise it falls back to a local template.

## Stack

| Component | Role |
|---|---|
| CrewAI | Orchestrates the 6 agents + Flow (handoff, validation, graceful failure) |
| scikit-learn | `yield_pct` regression (LinearRegression vs RandomForest) + `qc_pass` classification |
| ChromaDB + local embeddings | RAG over experiments + papers/patents, **no raw data to the cloud** |
| Claude API | Complex reasoning (report) — sanitised context only |
| Pandas | Dataset, inventory, stock deductions |
| Streamlit | Lab UI: entry, inventory dashboard, audit log, history |
| uv / Git+GitHub | Environment + versioning (Pull Request workflow) |

## Installation

```bash
uv sync
ollama pull nomic-embed-text   # optional; otherwise Chroma's built-in local embedder
cp .env.example .env           # add ANTHROPIC_API_KEY (optional)
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
