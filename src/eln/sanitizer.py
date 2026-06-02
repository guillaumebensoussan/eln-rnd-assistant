"""Sanitizer IP-safe + audit log — le cœur du pitch "données labo restent locales".

Avant tout appel au LLM cloud (Claude), on remplace les noms de réactifs taggés
`ip_sensitive` par des alias stables (IntermediateX-123 -> REAGENT_A1). L'audit log
journalise, pour chaque opération, ce qui est resté local vs parti au cloud.
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
    """Construit une table d'alias depuis l'inventaire et scrub le texte sortant."""

    def __init__(self, inventory_csv: Path | None = None):
        inv = pd.read_csv(inventory_csv or DATA_DIR / "inventory.csv")
        sensitive = inv[inv["ip_sensitive"]]["name"].tolist()
        # alias déterministe et stable
        self.alias_map = {name: f"REAGENT_{chr(65 + i // 10)}{i % 10}" for i, name in enumerate(sorted(sensitive))}
        # regex compilée, plus longs noms d'abord pour éviter les chevauchements
        names = sorted(self.alias_map, key=len, reverse=True)
        self._pattern = re.compile("|".join(re.escape(n) for n in names)) if names else None

    def scrub(self, text: str) -> tuple[str, list[str]]:
        """Retourne (texte_assaini, liste_des_noms_caviardés)."""
        if not self._pattern:
            return text, []
        redacted: list[str] = []

        def repl(m: re.Match) -> str:
            name = m.group(0)
            redacted.append(name)
            return self.alias_map[name]

        return self._pattern.sub(repl, text), redacted


def audit(operation: str, sent_to_cloud: str, redacted: list[str], stayed_local: list[str]) -> None:
    """Append-only audit log : preuve de ce qui a quitté la machine."""
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
