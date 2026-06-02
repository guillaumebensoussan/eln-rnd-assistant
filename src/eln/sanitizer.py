"""IP-safe sanitizer + audit log — the heart of the "lab data stays local" pitch.

Before any cloud LLM call (Claude), reagent names tagged `ip_sensitive` are replaced
with stable aliases (IntermediateX-123 -> REAGENT_A1). The audit log records, for each
operation, what stayed local vs what was sent to the cloud.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
AUDIT_LOG = ROOT / "artifacts" / "ip_audit_log.jsonl"


class Sanitizer:
    """Builds an alias table from the inventory and scrubs outgoing text."""

    def __init__(self, inventory_csv: Path | None = None):
        inv = pd.read_csv(inventory_csv or DATA_DIR / "inventory.csv")
        sensitive = inv[inv["ip_sensitive"]]["name"].tolist()
        # deterministic, stable alias
        self.alias_map = {name: f"REAGENT_{chr(65 + i // 10)}{i % 10}" for i, name in enumerate(sorted(sensitive))}
        # compiled regex, longest names first to avoid overlaps
        names = sorted(self.alias_map, key=len, reverse=True)
        self._pattern = re.compile("|".join(re.escape(n) for n in names)) if names else None

    def scrub(self, text: str) -> tuple[str, list[str]]:
        """Return (sanitised_text, list_of_redacted_names)."""
        if not self._pattern:
            return text, []
        redacted: list[str] = []

        def repl(m: re.Match) -> str:
            name = m.group(0)
            redacted.append(name)
            return self.alias_map[name]

        return self._pattern.sub(repl, text), redacted


def audit(operation: str, sent_to_cloud: str, redacted: list[str], stayed_local: list[str]) -> None:
    """Append-only audit log: evidence of what left the machine."""
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "redacted_count": len(redacted),
        "redacted_names": redacted,
        "stayed_local": stayed_local,
        "chars_sent_to_cloud": len(sent_to_cloud),
    }
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_audit() -> list[dict]:
    if not AUDIT_LOG.exists():
        return []
    return [json.loads(line) for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    s = Sanitizer()
    sample = f"Used {next(iter(s.alias_map))} at 80C, got 72% yield with Reagent-A0."
    clean, red = s.scrub(sample)
    print("original:", sample)
    print("scrubbed:", clean)
    print("redacted:", red)
    audit("demo", clean, red, stayed_local=["raw experiment record", "embeddings"])
    print("audit entries:", len(read_audit()))
