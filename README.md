# Agentic ELN - an IP-safe AI lab notebook for industrial R&D

An **agentic Electronic Lab Notebook**: chemists log experiments in plain language, the system structures them, checks reagent stock and safety, searches the lab history, predicts yield and QC outcome, and writes the report.

The constraint that shapes everything: **in industrial R&D, raw experimental data is intellectual property.** It must not leak to a cloud LLM. So the data stays on the workstation, only a sanitised context (reagents replaced by aliases like `REAGENT_A1`) ever reaches the cloud, and every call is logged to an audit trail. With a local model (Ollama), nothing leaves the machine at all.

Built by a PhD chemist (Weizmann Institute) who has spent years writing paper lab notebooks, as the final project of the AI Developers program at the Hebrew University of Jerusalem (2026).

## What it does

| Capability | How |
|---|---|
| **Natural-language intake** | An intake agent turns a free-text lab entry into a structured, validated experiment record |
| **Inventory** | Stock levels and automatic deductions per experiment (Pandas) |
| **Lab-history Q&A** | RAG over past experiments, papers and patents (ChromaDB, local embeddings) |
| **Prediction** | Yield regression and QC pass/fail classification (scikit-learn) |
| **Safety check** | GHS hazards from PubChem, pairwise incompatibilities, mixing order and procedural warnings |
| **Reporting** | Synthesis report, plus Word, Excel and chart exports |
| **Chat assistant** | Multi-turn chat over all of the above, in a Streamlit UI |
| **IP audit** | Every cloud call logged: what was sent, what was redacted, what stayed local |

## Architecture

Two CrewAI crews chained by a deterministic Flow, with a validation gate in between:

```
Crew 1 - Intake & Knowledge          Validation gate              Crew 2 - Analytics & Reporting
  Intake (structuring)       ─┐                                     Prediction (scikit-learn)
  Inventory (Pandas)          ├─►  data contract + features OK ──►  IP audit (sanitise + log)
  Retrieval (RAG)            ─┘    else: clean halt                 Synthesis / report (LLM)
```

The Flow (`src/eln/flow.py`) validates the dataset contract and required features before any modelling, logs every step, and halts cleanly instead of producing a wrong answer.

## Design decisions

- **The LLM never decides a hazard.** Hazard data comes from PubChem, and incompatibility and mixing-order logic is a deterministic local rule engine (`data/reactivity_rules.json`). The LLM only phrases the result.
- **Safety warns, the chemist decides.** Critical findings raise a prominent alert that must be acknowledged, but never silently block an experiment.
- **Sanitisation is reversible and local.** The `Sanitizer` maps real names to aliases before a cloud call and maps them back for local display (`src/eln/sanitizer.py`). Tool calls run locally on real names; their results are scrubbed before going back to the model. IP-sensitive reagents are never sent to PubChem.
- **Works without any API key.** Without a cloud key, the chat falls back to a deterministic intent router and local templates.
- **Provider-agnostic.** The LLM is set by one variable (`ELN_MODEL`, LiteLLM format): Claude (default), Gemini, OpenAI, or a fully local Ollama model.

## Results

On the synthetic dataset (90 train / 30 test rows):

| Task | Best model | Score |
|---|---|---|
| Yield (`yield_pct`) | Linear regression | R² 0.80, MAE 4.8 pts |
| QC pass/fail (`qc_pass`) | Logistic regression | Accuracy 0.87, F1 0.71 |

Full comparison against random forests in [`artifacts/evaluation_report.md`](artifacts/evaluation_report.md), model card in [`artifacts/model_card.md`](artifacts/model_card.md).

> **Note on data:** the experiments, papers and patents in `data/` are **synthetic**, generated deterministically by `data/generate_dataset.py`. No real lab or proprietary data is included.

## Stack

`Python 3` · `CrewAI` · `LiteLLM` · `scikit-learn` · `ChromaDB` · `Pandas` · `Streamlit` · `PubChem API` · `Ollama` · `pytest` · `uv`

## Getting started

```bash
uv sync
cp .env.example .env                        # optional: set ELN_MODEL and a provider key
uv run python data/generate_dataset.py      # generate the synthetic dataset and documents
uv run python src/eln/ml/train.py           # train the models -> artifacts/
PYTHONPATH=src uv run python -m eln --demo  # run the full Flow from the CLI
PYTHONPATH=src uv run streamlit run app/streamlit_app.py   # launch the UI
PYTHONPATH=src uv run pytest -q             # tests and RAG evaluation
```

Optional: `ollama pull nomic-embed-text` for local embeddings (otherwise ChromaDB's built-in embedder is used).

## Project structure

```
src/eln/
  flow.py            Flow: chains both crews, validation gate, clean halt
  crew_intake.py     Crew 1: intake, inventory, retrieval agents
  crew_analytics.py  Crew 2: prediction, IP audit, synthesis agents
  crew_safety.py     EHS agent
  sanitizer.py       Alias mapping and IP audit log
  rag.py             ChromaDB retrieval
  safety/            PubChem client, reactivity rule engine, checker
  chat/              Chat engine, deterministic router, sanitised tool belt
  ml/                Model training
  reports.py         Word, Excel and chart generation
app/                 Streamlit UI
data/                Synthetic dataset, data contract, reactivity rules
artifacts/           Trained model, EDA report, evaluation, model card, audit log
tests/               pytest suite
```

## Roadmap

- A "data scientist" agent producing aggregate insights across all experiments (trends, solvent comparisons, QC failure patterns)
- A faster hosted LLM option (e.g. Amazon Bedrock) for when local inference is too slow
- Voice input and live patent search

## Author

Guillaume Bensoussan ([LinkedIn](https://www.linkedin.com/in/guillaume-bensoussan))
