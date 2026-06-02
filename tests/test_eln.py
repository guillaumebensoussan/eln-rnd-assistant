"""Tests ELN : eval RAG, échec gracieux du Flow, sanitizer, ML."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eln import flow as flow_mod
from eln.flow import run
from eln.rag import RAG
from eln.sanitizer import Sanitizer
from eln.tools import parse_experiment, predict_outcome

# Jeu de référence : question -> doc_id attendu en tête de recherche
RAG_GOLDEN = [
    ("which solvent gives the highest yield?", "paper_solvent_effects"),
    ("what is the QC purity threshold?", "paper_qc_purity"),
    ("how to reduce palladium catalyst loading?", "paper_pd_catalysis"),
]


@pytest.fixture(scope="module")
def rag():
    r = RAG()
    r.index()
    return r


@pytest.mark.parametrize("query,expected_id", RAG_GOLDEN)
def test_rag_retrieves_relevant_doc(rag, query, expected_id):
    top_ids = [h["id"] for h in rag.search(query, k=3)]
    assert expected_id in top_ids, f"{expected_id} not in top-3 for '{query}': {top_ids}"


def test_parse_experiment_extracts_fields():
    r = parse_experiment("toluene at 110 C for 8 h, 2 mol%, 3 bar, purity 97")
    assert r["temperature_c"] == 110.0
    assert r["solvent"] == "toluene"
    assert r["purity_in_pct"] == 97.0


def test_prediction_in_valid_range():
    r = parse_experiment("THF at 100 C for 8 h, 2 mol%, 3 bar, purity 96")
    pred = predict_outcome(r)
    assert 0 <= pred["predicted_yield_pct"] <= 100
    assert isinstance(pred["predicted_qc_pass"], bool)


def test_flow_happy_path():
    state = run("toluene at 110 C for 8 h, 2 mol%, 3 bar, purity 97", "best solvent?")
    assert not state.errors
    assert state.record["temperature_c"] == 110.0
    assert "predicted_yield_pct" in state.prediction
    assert state.report


def test_flow_fails_gracefully_on_missing_features(monkeypatch):
    # simule un intake défaillant : features requises manquantes
    monkeypatch.setattr(flow_mod, "parse_experiment", lambda t: {"solvent": "THF"})
    state = run("garbage input")
    assert state.errors, "expected validation errors"
    assert "FLOW HALTED" in state.report
    assert not state.prediction  # Crew 2 ne doit pas tourner


def test_sanitizer_redacts_ip_sensitive():
    s = Sanitizer()
    sensitive_name = next(iter(s.alias_map))
    clean, redacted = s.scrub(f"Used {sensitive_name} in the reaction.")
    assert sensitive_name not in clean
    assert sensitive_name in redacted
