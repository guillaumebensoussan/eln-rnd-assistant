# Agentic ELN — IP-safe R&D Lab Notebook

Projet final — *AI Developers Cohort 8 (HUJI)*. Voie alternative approuvée par Jonathan Zouari.
Co-réalisation Guillaume Bensoussan · Eyal Ben Ari.

Un **Electronic Lab Notebook agentique** pour la R&D industrielle : log d'expériences en langage
naturel, gestion d'inventaire de réactifs, Q&A RAG sur l'historique, prédiction de rendement/QC, et
génération de rapport — avec une garantie **IP-safe** : les données brutes restent locales, seul un
contexte assaini atteint le LLM cloud.

## Architecture — Flow CrewAI à deux crews

```
Crew 1 — Lab Intake & Knowledge          Gate de validation        Crew 2 — Analytics & Reporting
  Intake (structuring)          ─┐                                    Prediction (scikit-learn)
  Inventory (stock, Pandas)      ├─►  contract + features OK  ──►      IP-Audit (sanitisation + log)
  Retrieval (RAG, ChromaDB)     ─┘    sinon: échec gracieux           Synthesis/Report (Claude*)
```

Le Flow (`src/eln/flow.py`) enchaîne les étapes de façon **déterministe**, **valide** le
`dataset_contract.json` et les features requises avant la modélisation, **logge** chaque étape et
**échoue proprement** (FLOW HALTED) si la validation casse.

\* Le report utilise Claude si `ANTHROPIC_API_KEY` est défini ; sinon il bascule en template local.

## Stack

| Brique | Rôle |
|---|---|
| CrewAI | Orchestration des 6 agents + Flow (handoff, validation, échec gracieux) |
| scikit-learn | Régression `yield_pct` (LinearRegression vs RandomForest) + classif `qc_pass` |
| ChromaDB + embeddings locaux | RAG sur expériences + papiers/brevets, **aucune donnée brute au cloud** |
| Claude API | Raisonnement complexe (report) — contexte assaini uniquement |
| Pandas | Dataset, inventaire, déductions de stock |
| Streamlit | UI labo : saisie, dashboard inventaire, audit log, historique |
| uv / Git+GitHub | Env + versioning (workflow Pull Requests) |

## Installation

```bash
uv sync
ollama pull nomic-embed-text   # optionnel ; sinon embedder local intégré de Chroma
cp .env.example .env           # y mettre ANTHROPIC_API_KEY (optionnel)
```

## Utilisation

```bash
uv run python data/generate_dataset.py     # génère dataset synthétique + docs (déterministe)
uv run python src/eln/ml/train.py          # entraîne les modèles -> artifacts/
PYTHONPATH=src uv run python -m eln --demo  # exécute le Flow end-to-end en CLI
PYTHONPATH=src uv run streamlit run app/streamlit_app.py   # UI
PYTHONPATH=src uv run pytest -q             # tests + eval RAG
```

## Artefacts produits (grille HUJI)

- `data/clean_data.csv`, `data/dataset_contract.json`, `data/features.csv`
- `artifacts/eda_report.html`, `artifacts/model.pkl`
- `artifacts/evaluation_report.md`, `artifacts/model_card.md`
- `artifacts/ip_audit_log.jsonl` (preuve local-vs-cloud)

## Périmètre

v1 = cœur défendable : Flow + RAG + inventaire + ML + report + IP-audit.
Optionnel (v2) : saisie vocale (Whisper), routeur LLM local (Ollama), recherche brevets live (PatentsView).

## Livrables cours

- [x] Repo + code + artefacts + README
- [ ] Présentation 10–12 slides / démo
- [ ] Vidéo ≤ 5 min
