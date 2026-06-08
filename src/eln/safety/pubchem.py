"""Cache-first PubChem client for GHS hazard data.

Resolves CAS -> CID -> GHS Classification (pictograms, signal word, H-codes)
via PUG REST / PUG View. Network is optional: the cache is consulted first and
any failure degrades to source="unknown" instead of raising.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "data" / "sds_cache.json"

_CID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{cas}/cids/JSON"
_GHS_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification"
_HCODE_RE = re.compile(r"\b(H\d{3})\b")
_UNKNOWN = {"cid": None, "ghs_pictograms": [], "h_codes": [], "signal_word": None, "source": "unknown"}


def _walk_ghs_sections(node):
    """Yield every section whose TOCHeading == 'GHS Classification'."""
    if isinstance(node, dict):
        if node.get("TOCHeading") == "GHS Classification":
            yield node
        for child in node.get("Section", []):
            yield from _walk_ghs_sections(child)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_ghs_sections(item)


def parse_ghs_view(payload: dict) -> dict:
    """Parse a PUG View GHS Classification payload into a hazard dict."""
    pictograms: list[str] = []
    signal: str | None = None
    h_codes: list[str] = []
    for section in _walk_ghs_sections(payload.get("Record", {})):
        for info in section.get("Information", []):
            name = info.get("Name")
            swm = info.get("Value", {}).get("StringWithMarkup", [])
            if name == "Pictogram(s)":
                for entry in swm:
                    for mark in entry.get("Markup", []):
                        extra = mark.get("Extra")
                        if extra and extra not in pictograms:
                            pictograms.append(extra)
            elif name == "Signal" and swm and signal is None:
                signal = swm[0].get("String")
            elif name == "GHS Hazard Statements":
                for entry in swm:
                    m = _HCODE_RE.search(entry.get("String", ""))
                    if m and m.group(1) not in h_codes:
                        h_codes.append(m.group(1))
    return {"cid": None, "ghs_pictograms": pictograms, "h_codes": h_codes,
            "signal_word": signal, "source": "pubchem"}


class PubChemClient:
    def __init__(self, cache_path: Path | None = None, allow_network: bool = True,
                 timeout: float = 20.0):
        self.cache_path = cache_path or DEFAULT_CACHE
        self.allow_network = allow_network
        self.timeout = timeout
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def get_hazard(self, cas: str) -> dict:
        if cas in self._cache:
            return {"cas": cas, **self._cache[cas]}
        if not self.allow_network:
            return {"cas": cas, **_UNKNOWN}
        haz = self._fetch(cas)
        self._cache[cas] = {k: v for k, v in haz.items() if k != "cas"}
        self._save_cache()
        return haz

    def _fetch(self, cas: str) -> dict:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(_CID_URL.format(cas=cas))
                r.raise_for_status()
                cids = r.json().get("IdentifierList", {}).get("CID", [])
                if not cids:
                    return {"cas": cas, **_UNKNOWN}
                cid = cids[0]
                time.sleep(0.25)  # PubChem politeness (<=5 req/s)
                g = client.get(_GHS_URL.format(cid=cid))
                g.raise_for_status()
                parsed = parse_ghs_view(g.json())
                parsed["cid"] = cid
                return {"cas": cas, **parsed}
        except (httpx.HTTPError, ValueError, KeyError):
            return {"cas": cas, **_UNKNOWN}
