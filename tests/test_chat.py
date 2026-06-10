"""Chat assistant tests: sanitizer round-trip, router, safe tools, engine, llm config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eln.sanitizer import Sanitizer


def test_sanitizer_unscrub_is_inverse_of_scrub():
    s = Sanitizer()
    sensitive = next(iter(s.alias_map))  # a real ip_sensitive name from inventory
    original = f"Mix {sensitive} with toluene at 80C."
    clean, redacted = s.scrub(original)
    assert sensitive not in clean
    assert redacted == [sensitive]
    assert s.unscrub(clean) == original


def test_sanitizer_unscrub_handles_text_without_aliases():
    s = Sanitizer()
    assert s.unscrub("plain text, no alias") == "plain text, no alias"
