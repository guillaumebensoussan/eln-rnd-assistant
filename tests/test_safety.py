"""Safety layer tests: reactivity engine, PubChem cache, checker, Flow gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eln.safety import reactivity as rx


def test_hcode_maps_to_oxidizer():
    groups = rx.groups_for(cas="0-0-0", h_codes=["H271"])
    assert "oxidizer" in groups


def test_cas_override_adds_acid_group():
    groups = rx.groups_for(cas="7647-01-0", h_codes=["H314"])
    assert "acid" in groups          # from CAS override
    assert "corrosive" in groups     # from H314


def test_pair_acid_base_is_critical():
    inc = rx.evaluate_pair(["acid"], ["base"])
    assert inc is not None
    assert inc["severity"] == "critical"
    assert "neutralisation" in inc["reason"].lower()


def test_pair_order_independent():
    assert rx.evaluate_pair(["base"], ["acid"]) is not None


def test_pair_no_conflict_returns_none():
    assert rx.evaluate_pair(["flammable"], ["flammable"]) is None


def test_order_acid_with_water_not_inverted():
    groups_by_reagent = {"HCl": ["acid"], "water": ["water"]}
    warnings = rx.mixing_warnings(groups_by_reagent)
    joined = " ".join(warnings).lower()
    assert "add the acid to the water" in joined
    assert "water to the acid" in joined  # phrased as the prohibition
    # explicit anti-inversion: must NOT instruct adding water to acid as the action
    assert "add the water to the acid" not in joined


from eln.safety.pubchem import PubChemClient, parse_ghs_view

FIXTURE_CACHE = ROOT / "tests" / "fixtures" / "sds_cache_test.json"


def test_pubchem_returns_cached_hazard_without_network():
    client = PubChemClient(cache_path=FIXTURE_CACHE, allow_network=False)
    haz = client.get_hazard("7647-01-0")
    assert haz["cid"] == 313
    assert "H314" in haz["h_codes"]
    assert haz["signal_word"] == "Danger"
    assert haz["source"] == "cache"


def test_pubchem_unknown_cas_offline_is_graceful():
    client = PubChemClient(cache_path=FIXTURE_CACHE, allow_network=False)
    haz = client.get_hazard("0000-00-0")
    assert haz["source"] == "unknown"
    assert haz["h_codes"] == []
    assert haz["cid"] is None


def test_parse_ghs_view_extracts_codes_and_pictograms():
    sample = {
        "Record": {"Section": [{"TOCHeading": "Safety and Hazards", "Section": [
            {"TOCHeading": "Hazards Identification", "Section": [
                {"TOCHeading": "GHS Classification", "Information": [
                    {"Name": "Pictogram(s)", "Value": {"StringWithMarkup": [
                        {"String": "  ", "Markup": [{"Type": "Icon", "Extra": "Corrosive"}]}]}},
                    {"Name": "Signal", "Value": {"StringWithMarkup": [{"String": "Danger"}]}},
                    {"Name": "GHS Hazard Statements", "Value": {"StringWithMarkup": [
                        {"String": "H314 (99.9%): Causes severe skin burns"},
                        {"String": "H290: May be corrosive to metals"}]}},
                ]}]}]}]}}
    parsed = parse_ghs_view(sample)
    assert parsed["signal_word"] == "Danger"
    assert parsed["ghs_pictograms"] == ["Corrosive"]
    assert parsed["h_codes"] == ["H314", "H290"]
