"""Génère le dataset synthétique R&D de l'ELN.

Déterministe (seed fixe) pour la reproductibilité exigée par le brief HUJI.
Produit :
  - data/inventory.csv        (~50 réactifs, avec tag ip_sensitive)
  - data/clean_data.csv       (~200 expériences, 4 types)
  - data/dataset_contract.json
  - data/docs/*.txt           (papiers + brevets synthétiques pour le RAG)
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).resolve().parent
DOCS_DIR = DATA_DIR / "docs"

rng = np.random.default_rng(SEED)

SOLVENTS = ["THF", "DCM", "toluene", "methanol", "DMF", "water", "acetonitrile"]
EXP_TYPES = ["synthesis", "characterisation", "formulation", "qc"]


def make_inventory(n: int = 50) -> pd.DataFrame:
    suppliers = ["Sigma-Aldrich", "Merck", "TCI", "Fisher", "Internal-Synth"]
    rows = []
    for i in range(n):
        # ~30% de réactifs propriétaires (codes internes) => IP-sensitive
        proprietary = rng.random() < 0.30
        if proprietary:
            name = f"IntermediateX-{rng.integers(100, 999)}"
            supplier = "Internal-Synth"
        else:
            name = f"Reagent-{chr(65 + i % 26)}{i}"
            supplier = str(rng.choice(suppliers[:-1]))
        rows.append(
            {
                "reagent_id": f"R{i:03d}",
                "name": name,
                "cas": f"{rng.integers(50, 9999)}-{rng.integers(10, 99)}-{rng.integers(1, 9)}",
                "supplier": supplier,
                "batch": f"B{rng.integers(1000, 9999)}",
                "quantity_g": round(float(rng.uniform(5, 500)), 1),
                "expiry": str(date(2026, 1, 1) + timedelta(days=int(rng.integers(30, 900)))),
                "location": f"Shelf-{rng.integers(1, 12)}",
                "ip_sensitive": bool(proprietary),
            }
        )
    return pd.DataFrame(rows)


def make_experiments(inventory: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Génère des expériences ; le rendement des synthèses suit un modèle latent
    (signal apprenable + bruit) pour que le bloc ML ait quelque chose à apprendre."""
    reagent_ids = inventory["reagent_id"].tolist()
    rows = []
    # répartition : synthesis majoritaire pour nourrir le ML
    type_counts = {"synthesis": 120, "characterisation": 30, "formulation": 25, "qc": 25}
    eid = 0
    for etype, count in type_counts.items():
        for _ in range(count):
            eid += 1
            base = {
                "experiment_id": f"EXP{eid:04d}",
                "date": str(date(2026, 1, 1) + timedelta(days=int(rng.integers(0, 150)))),
                "operator": str(rng.choice(["G.Bensoussan", "E.BenAri", "lab-tech-1", "lab-tech-2"])),
                "experiment_type": etype,
                "reagent_id": str(rng.choice(reagent_ids)),
            }
            if etype == "synthesis":
                temp = float(rng.uniform(20, 160))
                time_h = float(rng.uniform(0.5, 24))
                cat_load = float(rng.uniform(0.5, 10))  # mol%
                pressure = float(rng.uniform(1, 5))      # bar
                purity_in = float(rng.uniform(85, 99.9))
                solvent = str(rng.choice(SOLVENTS))
                solvent_bonus = {"THF": 4, "toluene": 6, "DMF": 3}.get(solvent, 0)
                # modèle latent du rendement
                yield_pct = (
                    35
                    + 0.18 * temp
                    - 0.9 * (time_h - 8) ** 2 / 8
                    + 2.1 * cat_load
                    + 3.0 * pressure
                    + 0.25 * (purity_in - 90)
                    + solvent_bonus
                    + rng.normal(0, 6)
                )
                yield_pct = float(np.clip(yield_pct, 0, 100))
                purity_out = float(np.clip(purity_in - rng.uniform(0, 8) + yield_pct * 0.03, 0, 100))
                base.update(
                    {
                        "temperature_c": round(temp, 1),
                        "time_h": round(time_h, 2),
                        "catalyst_loading_molpct": round(cat_load, 2),
                        "pressure_bar": round(pressure, 2),
                        "solvent": solvent,
                        "purity_in_pct": round(purity_in, 2),
                        "yield_pct": round(yield_pct, 2),
                        "purity_out_pct": round(purity_out, 2),
                        "qc_pass": bool(purity_out >= 95 and yield_pct >= 50),
                    }
                )
            elif etype == "characterisation":
                base.update(
                    {
                        "method": str(rng.choice(["XRD", "SEM", "DSC"])),
                        "crystallinity_pct": round(float(rng.uniform(40, 99)), 1),
                        "particle_size_nm": round(float(rng.uniform(20, 800)), 1),
                        "melting_point_c": round(float(rng.uniform(60, 320)), 1),
                    }
                )
            elif etype == "formulation":
                r = float(rng.uniform(0.1, 0.9))
                base.update(
                    {
                        "component_a_ratio": round(r, 3),
                        "component_b_ratio": round(1 - r, 3),
                        "stability_days": int(rng.integers(7, 365)),
                        "viscosity_cp": round(float(rng.uniform(1, 5000)), 1),
                    }
                )
            else:  # qc
                purity = float(rng.uniform(80, 100))
                base.update(
                    {
                        "batch_number": f"QC{rng.integers(1000, 9999)}",
                        "measured_purity_pct": round(purity, 2),
                        "qc_pass": bool(purity >= 95),
                    }
                )
            rows.append(base)
    return pd.DataFrame(rows)


def make_contract(experiments: pd.DataFrame) -> dict:
    """Dataset contract : schéma, types, contraintes — consommé par le Flow pour valider."""
    return {
        "version": "1.0",
        "primary_key": "experiment_id",
        "experiment_types": EXP_TYPES,
        "columns": {c: str(experiments[c].dtype) for c in experiments.columns},
        "constraints": {
            "yield_pct": {"min": 0, "max": 100},
            "purity_out_pct": {"min": 0, "max": 100},
            "temperature_c": {"min": -80, "max": 300},
            "experiment_type": {"allowed": EXP_TYPES},
        },
        "ml_targets": {
            "regression": {"target": "yield_pct", "scope": "synthesis"},
            "classification": {"target": "qc_pass", "scope": "synthesis"},
        },
        "required_features_for_ml": [
            "temperature_c",
            "time_h",
            "catalyst_loading_molpct",
            "pressure_bar",
            "purity_in_pct",
            "solvent",
        ],
    }


PAPERS = {
    "paper_pd_catalysis.txt": (
        "Title: Palladium-Catalysed Cross-Coupling at Reduced Catalyst Loading\n"
        "Abstract: We report that Pd catalyst loadings as low as 1 mol% in toluene at 110 C "
        "afford cross-coupling yields above 85% when substrate purity exceeds 95%. Elevated "
        "pressure (3-4 bar) further improves conversion. THF and toluene outperform protic solvents.\n"
        "Keywords: palladium, cross-coupling, catalyst loading, yield optimisation."
    ),
    "paper_solvent_effects.txt": (
        "Title: Solvent Effects on Reaction Yield in Organometallic Synthesis\n"
        "Abstract: A systematic study across THF, DCM, toluene, DMF and methanol shows aprotic "
        "solvents (THF, toluene, DMF) give higher yields than protic media. Optimal reaction time "
        "is near 8 hours; longer times reduce yield via decomposition.\n"
        "Keywords: solvent, yield, organometallic, reaction time."
    ),
    "paper_qc_purity.txt": (
        "Title: Quality Control Thresholds for Pharmaceutical Intermediates\n"
        "Abstract: Batches with measured purity below 95% fail QC. Output purity correlates with "
        "input purity and yield. We recommend a 95% purity gate for downstream release.\n"
        "Keywords: quality control, purity, QC gate, release criteria."
    ),
}

PATENTS = {
    "patent_US1234567.txt": (
        "Patent US1234567: Process for high-yield synthesis using proprietary intermediate.\n"
        "Claims: A method comprising reacting a substrate with a palladium catalyst at 100-150 C "
        "under 2-4 bar pressure in an aprotic solvent to obtain yields exceeding 80%. Prior art "
        "discloses loadings above 5 mol%; the present invention reduces loading to 1-2 mol%."
    ),
    "patent_EP7654321.txt": (
        "Patent EP7654321: Stabilised formulation with extended shelf life.\n"
        "Claims: A formulation comprising components A and B in a ratio between 0.3 and 0.7 "
        "exhibiting stability beyond 180 days. Viscosity is controlled below 2000 cP."
    ),
}


def write_docs() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    for name, text in {**PAPERS, **PATENTS}.items():
        (DOCS_DIR / name).write_text(text, encoding="utf-8")


def main() -> None:
    inventory = make_inventory()
    experiments = make_experiments(inventory)
    contract = make_contract(experiments)

    inventory.to_csv(DATA_DIR / "inventory.csv", index=False)
    experiments.to_csv(DATA_DIR / "clean_data.csv", index=False)
    (DATA_DIR / "dataset_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    write_docs()

    print(f"inventory: {len(inventory)} reagents ({inventory['ip_sensitive'].sum()} IP-sensitive)")
    print(f"experiments: {len(experiments)} rows")
    print(f"  by type: {experiments['experiment_type'].value_counts().to_dict()}")
    print(f"docs: {len(PAPERS) + len(PATENTS)} files in {DOCS_DIR}")
    print("contract written to dataset_contract.json")


if __name__ == "__main__":
    main()
