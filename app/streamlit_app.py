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
from eln.chat import ChatEngine
from eln.llm import get_model, has_cloud
from eln.sanitizer import read_audit
from eln.safety import check_safety

st.set_page_config(page_title="Agentic ELN", layout="wide")
st.title("Agentic ELN — IP-safe R&D Lab Notebook")
st.caption(
    ("Cloud LLM: Claude (key detected)" if has_cloud() else "Local mode (no API key — report falls back to template)")
)

tab_chat, tab_log, tab_inv, tab_safety, tab_audit, tab_hist = st.tabs(
    ["Assistant", "Log experiment", "Inventory", "Safety check", "IP Audit log", "History"]
)

with tab_chat:
    st.subheader("Lab assistant")
    st.caption(
        f"Cloud LLM: {get_model()}" if has_cloud()
        else "Local mode — deterministic answers (no API key set)"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_pending_ack" not in st.session_state:
        st.session_state.chat_pending_ack = False

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.chat_pending_ack:
        st.error("⛔ SAFETY ALERT — critical incompatibility. "
                 "Read the warnings above before continuing.")
        if st.checkbox("I have read these warnings and take responsibility.",
                       key="chat_ack"):
            from eln.sanitizer import audit
            audit(operation="safety_acknowledgement", sent_to_cloud="",
                  redacted=[], stayed_local=["chat critical-alert acknowledgement"])
            st.session_state.chat_pending_ack = False
            st.rerun()

    prompt = st.chat_input("Describe your next reaction or ask about the lab",
                           disabled=st.session_state.chat_pending_ack)
    if prompt:
        with st.spinner("Thinking…"):
            reply = ChatEngine().reply(st.session_state.chat_history, prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": reply.text})
        if reply.critical:
            st.session_state.chat_pending_ack = True
        st.rerun()

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

with tab_safety:
    st.subheader("SDS safety & compatibility check")
    inv_s = Inventory()
    chosen = st.multiselect("Reagents to combine", options=sorted(inv_s.df["name"]))
    if st.button("Run safety check", type="primary") and chosen:
        report = check_safety(chosen)
        if report.severity == "critical":
            st.error("⛔ SAFETY ALERT — critical incompatibility. Read carefully before proceeding.")
        elif report.severity == "warning":
            st.warning("⚠️ Safety warnings present.")
        else:
            st.success("No critical incompatibility detected for this set.")

        if report.incompatibilities:
            st.markdown("**Incompatibilities**")
            st.dataframe(
                pd.DataFrame([{"A": i.a, "B": i.b, "severity": i.severity, "reason": i.reason}
                              for i in report.incompatibilities]),
                use_container_width=True, hide_index=True,
            )
        if report.procedural_warnings:
            st.markdown("**Mixing order**")
            for w in report.procedural_warnings:
                st.markdown(f"- {w}")
            st.caption("Suggested addition order: " + " → ".join(report.addition_order))
        if report.hazards:
            st.markdown("**Per-reagent hazards**")
            st.dataframe(
                pd.DataFrame([{"reagent": h.name, "H-codes": ", ".join(h.h_codes),
                               "pictograms": ", ".join(h.ghs_pictograms), "source": h.source}
                              for h in report.hazards]),
                use_container_width=True, hide_index=True,
            )
        if report.unknowns:
            st.info("No public SDS for: " + ", ".join(report.unknowns) + " — check the internal safety sheet.")

        if report.severity == "critical":
            ack = st.checkbox("I have read these warnings and take responsibility.")
            if ack:
                from eln.sanitizer import audit
                audit(operation="safety_acknowledgement", sent_to_cloud="",
                      redacted=[], stayed_local=[f"ack:{', '.join(chosen)}"])
                st.success("Acknowledged and logged. You may proceed at your discretion.")
            else:
                st.stop()

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
