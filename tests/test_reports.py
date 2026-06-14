"""Reports tests: Word/Excel/chart generators are deterministic and local."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eln import reports


def test_plot_chart_writes_png():
    path = reports.plot_chart("bar", ["A", "B"], [10, 20], "Yield", "test-chart")
    assert path.exists() and path.suffix == ".png" and path.stat().st_size > 0


def test_plot_chart_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        reports.plot_chart("bar", ["A", "B"], [1], "bad", "test-bad")


def test_to_excel_writes_workbook():
    path = reports.to_excel("Sheet", ["h1", "h2"], [[1, 2], [3, 4]], "test-xlsx")
    assert path.exists() and path.suffix == ".xlsx" and path.stat().st_size > 0


def test_to_word_writes_document():
    sections = [{"heading": "Intro", "text": "hello", "table": [["k", "v"], ["a", 1]]}]
    path = reports.to_word("Title", sections, "test-docx")
    assert path.exists() and path.suffix == ".docx" and path.stat().st_size > 0


def test_out_strips_path_traversal():
    """An LLM tool call passing a full path must not escape REPORTS_DIR."""
    path = reports._out("../../etc/passwd", ".docx")
    assert path.parent == reports.REPORTS_DIR


def test_experiment_report_assembles_docx():
    path = reports.experiment_report(
        record={"temperature_c": 110.0, "solvent": "toluene"},
        prediction={"predicted_yield_pct": 82.0, "predicted_qc_pass": True},
        report_text="Methods: coupling reaction.",
        safety={"severity": "warning"},
        output_name="test-experiment",
    )
    assert path.exists() and path.suffix == ".docx" and path.stat().st_size > 0
