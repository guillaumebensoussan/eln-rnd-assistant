"""Agentic ELN — Streamlit interface (lab UI)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from eln.flow import run
from eln.inventory import Inventory
from eln.llm import has_cloud
from eln.sanitizer import read_audit

st.set_page_config(page_title="Agentic ELN", layout="wide")
st.title("Agentic ELN — IP-safe R&D Lab Notebook")
st.caption(
    ("Cloud LLM: Claude (key detected)" if has_cloud() else "Local mode (no API key — report falls back to template)")
)

tab_log, tab_inv, tab_audit, tab_hist = st.tabs(
    ["Log experiment", "Inventory", "IP Audit log", "History"]
)

with tab_log:
    st.subheader("Log a new experiment")
    text = st.text_area(
        "Describe the experiment (natural language)",
        value="Coupling with IntermediateX-119 in toluene at 110 C for 8 h, 2 mol% catalyst, 3 bar, purity 97.",
        height=110,
    )
    question = st.text_input("Question for the knowledge base", value="which solvent gives the highest yield?")
    if st.button("Run agentic flow", type="primary"):
        with st.spinner("Crew 1 -> validation -> Crew 2…"):
            state = run(text, question)
        if state.errors:
            st.error("Flow halted: " + "; ".join(state.errors))
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Structured record**")
                st.json(state.record)
                st.markdown("**Inventory deduction**")
                st.json(state.inventory_result)
            with c2:
                st.markdown("**Prediction (scikit-learn)**")
                st.metric("Predicted yield %", state.prediction.get("predicted_yield_pct", "n/a"))
                st.metric("Predicted QC pass", str(state.prediction.get("predicted_qc_pass", "n/a")))
            st.markdown("**Report**")
            st.markdown(state.report)

with tab_inv:
    st.subheader("Reagent inventory")
    inv = Inventory()
    alerts = inv.alerts()
    c1, c2 = st.columns(2)
    c1.metric("Low stock", len(alerts["low_stock"]))
    c2.metric("Expired", len(alerts["expired"]))
    st.dataframe(inv.df, use_container_width=True, hide_index=True)

with tab_audit:
    st.subheader("IP audit log — what left the server vs stayed local")
    entries = read_audit()
    if not entries:
        st.info("No audit entries yet. Run an experiment first.")
    else:
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)

with tab_hist:
    st.subheader("Experiment history")
    df = pd.read_csv(ROOT / "data" / "clean_data.csv")
    etype = st.selectbox("Filter by type", ["all"] + sorted(df["experiment_type"].unique()))
    view = df if etype == "all" else df[df["experiment_type"] == etype]
    st.dataframe(view, use_container_width=True, hide_index=True)
