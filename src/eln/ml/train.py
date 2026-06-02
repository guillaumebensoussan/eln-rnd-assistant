"""Bloc ML — comble le volet prédictif (Crew 2 "Data Scientist") du brief HUJI.

- EDA -> artifacts/eda_report.html
- Feature engineering -> data/features.csv
- Régression yield_pct : LinearRegression vs RandomForest (>= 2 variantes)
- Classification qc_pass : LogisticRegression vs RandomForest
- Artefacts : artifacts/model.pkl, artifacts/evaluation_report.md, artifacts/model_card.md

Déterministe (random_state fixe) pour la reproductibilité.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

NUM_FEATURES = [
    "temperature_c",
    "time_h",
    "catalyst_loading_molpct",
    "pressure_bar",
    "purity_in_pct",
]
CAT_FEATURES = ["solvent"]
REG_TARGET = "yield_pct"
CLF_TARGET = "qc_pass"


def load_synthesis() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "clean_data.csv")
    syn = df[df["experiment_type"] == "synthesis"].copy()
    return syn.dropna(subset=NUM_FEATURES + CAT_FEATURES + [REG_TARGET, CLF_TARGET])


def write_eda(df: pd.DataFrame) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    desc = df[NUM_FEATURES + [REG_TARGET]].describe().round(2)
    corr = df[NUM_FEATURES + [REG_TARGET]].corr().round(3)
    html = (
        "<html><head><meta charset='utf-8'><title>ELN EDA Report</title></head><body>"
        "<h1>ELN — Exploratory Data Analysis</h1>"
        f"<p>Synthesis experiments: {len(df)}</p>"
        "<h2>Descriptive statistics</h2>" + desc.to_html()
        + "<h2>Correlation matrix</h2>" + corr.to_html()
        + f"<h2>QC pass rate</h2><p>{df[CLF_TARGET].mean():.1%}</p>"
        + "</body></html>"
    )
    (ARTIFACTS / "eda_report.html").write_text(html, encoding="utf-8")


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )


def train() -> None:
    df = load_synthesis()
    write_eda(df)

    X = df[NUM_FEATURES + CAT_FEATURES]
    y_reg = df[REG_TARGET]
    y_clf = df[CLF_TARGET].astype(int)

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=0.25, random_state=RANDOM_STATE
    )

    # features.csv (jeu transformé exporté pour traçabilité)
    feat_out = X.copy()
    feat_out[REG_TARGET] = y_reg
    feat_out[CLF_TARGET] = y_clf
    feat_out.to_csv(DATA_DIR / "features.csv", index=False)

    # --- Régression : 2 variantes ---
    reg_models = {
        "LinearRegression": Pipeline([("prep", make_preprocessor()), ("m", LinearRegression())]),
        "RandomForestRegressor": Pipeline(
            [("prep", make_preprocessor()), ("m", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE))]
        ),
    }
    reg_results = {}
    for name, model in reg_models.items():
        model.fit(X_tr, yr_tr)
        pred = model.predict(X_te)
        reg_results[name] = {
            "r2": round(float(r2_score(yr_te, pred)), 4),
            "mae": round(float(mean_absolute_error(yr_te, pred)), 4),
        }

    # --- Classification : 2 variantes ---
    clf_models = {
        "LogisticRegression": Pipeline([("prep", make_preprocessor()), ("m", LogisticRegression(max_iter=1000))]),
        "RandomForestClassifier": Pipeline(
            [("prep", make_preprocessor()), ("m", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))]
        ),
    }
    clf_results = {}
    for name, model in clf_models.items():
        model.fit(X_tr, yc_tr)
        pred = model.predict(X_te)
        clf_results[name] = {
            "accuracy": round(float(accuracy_score(yc_te, pred)), 4),
            "f1": round(float(f1_score(yc_te, pred, zero_division=0)), 4),
        }

    best_reg = max(reg_results, key=lambda k: reg_results[k]["r2"])
    best_clf = max(clf_results, key=lambda k: clf_results[k]["f1"])

    # On persiste le meilleur de chaque tâche
    bundle = {
        "regressor": reg_models[best_reg],
        "classifier": clf_models[best_clf],
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "reg_target": REG_TARGET,
        "clf_target": CLF_TARGET,
    }
    joblib.dump(bundle, ARTIFACTS / "model.pkl")

    write_reports(reg_results, clf_results, best_reg, best_clf, len(X_tr), len(X_te))
    print("ML done.")
    print(f"  regression  -> {reg_results}  (best: {best_reg})")
    print(f"  classification -> {clf_results}  (best: {best_clf})")


def write_reports(reg_results, clf_results, best_reg, best_clf, n_train, n_test) -> None:
    eval_md = (
        "# Evaluation Report\n\n"
        f"Train rows: {n_train} · Test rows: {n_test} · random_state={RANDOM_STATE}\n\n"
        "## Regression — target `yield_pct`\n\n"
        "| Model | R2 | MAE |\n|---|---|---|\n"
        + "".join(f"| {k} | {v['r2']} | {v['mae']} |\n" for k, v in reg_results.items())
        + f"\nBest: **{best_reg}**\n\n"
        "## Classification — target `qc_pass`\n\n"
        "| Model | Accuracy | F1 |\n|---|---|---|\n"
        + "".join(f"| {k} | {v['accuracy']} | {v['f1']} |\n" for k, v in clf_results.items())
        + f"\nBest: **{best_clf}**\n"
    )
    (ARTIFACTS / "evaluation_report.md").write_text(eval_md, encoding="utf-8")

    card = (
        "# Model Card — ELN Yield & QC Predictor\n\n"
        "## Purpose\n"
        "Predict reaction `yield_pct` (regression) and `qc_pass` (classification) from synthesis "
        "parameters, to give R&D chemists an expected outcome before running an experiment.\n\n"
        "## Training data\n"
        "Synthetic R&D dataset (synthesis subset). Features: "
        f"{', '.join(NUM_FEATURES + CAT_FEATURES)}.\n\n"
        "## Metrics\n"
        f"- Best regressor: {best_reg} (R2={reg_results[best_reg]['r2']}, MAE={reg_results[best_reg]['mae']})\n"
        f"- Best classifier: {best_clf} (acc={clf_results[best_clf]['accuracy']}, F1={clf_results[best_clf]['f1']})\n\n"
        "## Limitations\n"
        "- Trained on synthetic data with a known latent model; real lab data will differ.\n"
        "- Does not extrapolate beyond observed parameter ranges.\n"
        "- Solvent encoded as categorical; unseen solvents are ignored.\n\n"
        "## Ethical considerations\n"
        "- Decision support only — does not replace chemist judgement or safety review.\n"
        "- No personal data; operator field is a pseudonymous label.\n"
    )
    (ARTIFACTS / "model_card.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    train()
