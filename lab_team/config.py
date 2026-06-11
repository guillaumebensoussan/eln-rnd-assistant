"""Central configuration: env loading, model IDs, paths, RAG config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from lab_team import paths  # noqa: F401  (extends sys.path)

REPO = paths.REPO
CHEM_RAG_DIR = paths.CHEM_RAG_DIR
ENV_FILE = CHEM_RAG_DIR / ".env"
RAG_CONFIG_FILE = CHEM_RAG_DIR / "config.yaml"
RESULTS_DIR = REPO / "lab_team" / "reports"

# ── LLM model IDs (LiteLLM-style provider/model strings) ──
MANAGER_MODEL = "ollama/qwen2.5:14b"            # fully local fallback; restore to anthropic/claude-sonnet-4-6 with valid key
WORKER_MODEL = "ollama/qwen2.5:14b"             # local; llama3.1:70b is the stronger fallback
OLLAMA_BASE_URL = "http://localhost:11434"

_loaded = False


def load_all() -> dict:
    """Load .env (ANTHROPIC_API_KEY) and the chem_rag config.yaml. Idempotent.

    Returns the RAG config dict with output_dir forced to an absolute path so it
    is independent of the current working directory.
    """
    global _loaded
    if not _loaded:
        load_dotenv(ENV_FILE)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _loaded = True
    return rag_config()


def rag_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if RAG_CONFIG_FILE.exists():
        cfg = yaml.safe_load(RAG_CONFIG_FILE.read_text(encoding="utf-8")) or {}
    # Force absolute output dir (run_pipeline defaults to a relative ./results).
    cfg["output_dir"] = str(RESULTS_DIR)
    return cfg
