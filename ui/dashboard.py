"""Main Streamlit UI for ORION-X (Orbital Risk & Intelligence Operations Network).

The UI keeps the scientifically distinct historical and present-day workflows
separate, while exposing the orbital tracker and the QKD-vs-classical security
comparison used by the research project.

Run:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules import data_loader
from modules import sgp4_propagation
from modules import visualization as viz
from modules import classical_security
from modules import qkd
from ui.historical_live_dashboard import render_historical_tab, render_live_tab
from ui.historical_rocket_tab import render_historical_rocket_tab


st.set_page_config(
    page_title="ORION-X — Orbital Risk & Intelligence Operations Network",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0b1220; }
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #f8fafc; }
    p, label { color: #cbd5e1; }
    [data-testid="stMetric"] {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 14px;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8; }
    [data-testid="stMetricValue"] { color: #e2e8f0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_orbital_data():
    try:
        return data_loader.load_orbital_data()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_result_json(path_text: str):
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def render_overview():
    st.header("🛰️ ORION-X — Orbital Risk & Intelligence Operations Network")
    st.caption(
        "Detection → propagation → conjunction screening → collision probability → "
        "risk forecasting → secure warning communication"
    )

    orbital = load_orbital_data()
    conjunction_path = Path(getattr(config, "CONJUNCTIONS_FILE", PROJECT_ROOT / "data" / "processed" / "conjunctions.csv"))
    predictions_path = PROJECT_ROOT / "results" / "predictions.csv"

    conjunctions = None
    predictions = None
    if conjunction_path.exists():
        try:
            conjunctions = pd.read_csv(conjunction_path)
        except Exception:
            pass
    if predictions_path.exists():
        try:
            predictions = pd.read_csv(predictions_path)
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current orbital objects", int(len(orbital)) if orbital is not None else 0)
    c2.metric("Screened conjunctions", int(len(conjunctions)) if conjunctions is not None else 0)
    c3.metric("30-day forecast rows", int(len(predictions)) if predictions is not None else 0)

    risk_count = 0
    if predictions is not None and not predictions.empty:
        risk_col = next(
            (c for c in ["COMPOSITE_RISK_LEVEL", "RISK_LEVEL", "ANALYTIC_RISK_LEVEL"] if c in predictions.columns),
            None,
        )
        if risk_col:
            risk_count = int(predictions[risk_col].astype(str).str.upper().isin(["HIGH", "CRITICAL"]).sum())
    c4.metric("HIGH / CRITICAL forecast", risk_count)

    st.subheader("System architecture")
    st.markdown(
        "**Current orbital data** → **SGP4 propagation** → **conjunction detection** → "
        "**collision-probability model** → **risk classification** → "
        "**QAOA/QAE analysis** → **QKD-secured warning communication**"
    )

    st.subheader("Research modes")
    a, b, c = st.columns(3)
    with a:
        st.markdown("### ⏪ Historical")
        st.write("Replay a confirmed collision using archived orbital information and evaluate the forecast against known ground truth.")
    with b:
        st.markdown("### 📡 Present / Live")
        st.write("Use current orbital data to propagate objects and screen the next 30 days without feeding historical outcomes into the live predictor.")
    with c:
        st.markdown("### 🔐 Secure warning")
        st.write("Compare classical ECDH/AES-GCM communication with simulated BB84 QKD and its QBER-based eavesdropping detection.")


def render_orbital_tracker():
    st.header("🌍 Orbital Tracker")
    st.caption("Current orbital objects, their orbital elements, and on-demand SGP4 position propagation.")

    orbital = load_orbital_data()
    if orbital is None or orbital.empty:
        st.warning("No current orbital dataset is available.")
        return

    id_col = next((c for c in ["NORAD_CAT_ID", "NORAD_ID", "OBJECT_ID"] if c in orbital.columns), None)
    name_col = next((c for c in ["OBJECT_NAME", "NAME", "SATNAME"] if c in orbital.columns), None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tracked objects", int(orbital[id_col].nunique()) if id_col else int(len(orbital)))
    c2.metric("Data rows", int(len(orbital)))
    c3.metric("Data columns", int(len(orbital.columns)))

    display = orbital.copy()
    if name_col:
        search = st.text_input("Filter by object name", "")
        if search:
            display = display[display[name_col].astype(str).str.contains(search, case=False, na=False)]

    st.subheader("Orbital elements")
    st.dataframe(display, width="stretch", height=330)

    st.subheader("Live position propagation")
    st.caption("Propagation is performed only when requested; this keeps the dashboard responsive on refresh.")
    if st.button("🛰️ Propagate current objects now", key="tracker_propagate", type="primary", width="stretch"):
        try:
            with st.spinner("Propagating current orbital elements with SGP4…"):
                positions, failed = sgp4_propagation.propagate_all_now(orbital, verbose=False)
            if positions is None or positions.empty:
                st.warning("No orbital states could be propagated.")
            else:
                st.session_state["tracker_positions"] = positions
                st.session_state["tracker_failed"] = failed
                st.success(f"Propagation complete: {len(positions):,} states.")
        except Exception as exc:
            st.error(f"Propagation failed: {exc}")

    positions = st.session_state.get("tracker_positions")
    if positions is not None and not positions.empty:
        failed = st.session_state.get("tracker_failed", [])
        c1, c2, c3 = st.columns(3)
        c1.metric("Propagated states", f"{len(positions):,}")
        c2.metric("Propagation failures", len(failed) if failed is not None else 0)
        c3.metric("Timestamp", datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))

        try:
            st.plotly_chart(
                viz.live_globe_figure(
                    positions,
                    timestamp_label=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                ),
                width="stretch",
            )
        except Exception as exc:
            st.info(f"3D orbital globe unavailable: {exc}")

        st.subheader("Propagated states")
        st.dataframe(positions.head(1000), width="stretch", height=300)


def render_qkd_classical():
    st.header("🔐 QKD vs Classical Security")
    st.caption(
        "The same collision-warning message is protected using a classical ECDH/AES-GCM baseline "
        "and a simulated BB84-derived key. QBER is used to detect the intercept-resend scenario."
    )

    classical_path = PROJECT_ROOT / "results" / "classical_security_results.json"
    qkd_path = PROJECT_ROOT / "results" / "qkd_results.json"
    classical_saved = load_result_json(str(classical_path))
    qkd_saved = load_result_json(str(qkd_path))

    with st.expander("Run security comparison", expanded=True):
        n_qubits = st.selectbox("BB84 qubits", [128, 256, 512], index=2)
        run_security = st.button("▶ Run QKD + classical comparison", key="security_run", type="primary", width="stretch")

    if run_security:
        try:
            with st.spinner(f"Running BB84 comparison with {n_qubits} qubits…"):
                classical_result = classical_security.run_and_save()
                honest, intercepted, qkd_summary = qkd.run_and_save(n_qubits=int(n_qubits))
            st.session_state["security_classical"] = classical_result
            st.session_state["security_honest"] = honest
            st.session_state["security_intercepted"] = intercepted
            st.session_state["security_qkd_summary"] = qkd_summary
            st.success("Security comparison completed and results were saved.")
        except Exception as exc:
            st.error(f"Security comparison failed: {exc}")

    classical = st.session_state.get("security_classical", classical_saved)
    honest = st.session_state.get("security_honest")
    intercepted = st.session_state.get("security_intercepted")
    qkd_summary = st.session_state.get("security_qkd_summary", qkd_saved)

    if classical is None and qkd_summary is None:
        st.info("No saved security results yet. Run the comparison above.")
        return

    st.subheader("Key results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classical handshake", f"{float(classical.get('handshake_runtime_sec', 0)):.4f} s" if classical else "—")
    c2.metric("Classical round trip", "PASS" if classical and classical.get("round_trip_verified") else "—")
    c3.metric("Honest QBER", f"{float((honest or {}).get('qber', qkd_summary.get('honest_qber', 0) if qkd_summary else 0)):.2%}")
    c4.metric("Intercept QBER", f"{float((intercepted or {}).get('qber', qkd_summary.get('intercepted_qber', 0) if qkd_summary else 0)):.2%}")

    comparison = pd.DataFrame([
        {
            "Method": "Classical — ECDH + AES-256-GCM",
            "Key establishment": "ECDH",
            "Encryption": "AES-256-GCM",
            "Eavesdropping signal": "Not inherent to ECDH handshake",
            "Round-trip verified": bool(classical.get("round_trip_verified")) if classical else None,
        },
        {
            "Method": "Quantum — BB84 + AES-GCM",
            "Key establishment": "BB84 QKD",
            "Encryption": "AES-GCM using QKD-derived key",
            "Eavesdropping signal": "QBER",
            "Round-trip verified": bool(qkd_summary.get("message_round_trip_verified")) if qkd_summary else None,
        },
    ])
    st.dataframe(comparison, width="stretch", hide_index=True)

    if honest or intercepted or qkd_summary:
        st.subheader("BB84 channel behavior")
        qber_df = pd.DataFrame({
            "Scenario": ["Honest channel", "Intercept-resend"],
            "QBER": [
                float((honest or {}).get("qber", qkd_summary.get("honest_qber", 0) if qkd_summary else 0)),
                float((intercepted or {}).get("qber", qkd_summary.get("intercepted_qber", 0) if qkd_summary else 0)),
            ],
        }).set_index("Scenario")
        st.bar_chart(qber_df)

        detected = bool(
            (intercepted or {}).get(
                "eavesdropping_detected",
                qkd_summary.get("intercepted_eavesdropping_detected", False) if qkd_summary else False,
            )
        )
        if detected:
            st.success("✓ BB84 detected the intercept-resend disturbance using the configured QBER threshold.")
        else:
            st.warning("The current QKD result did not flag the intercept-resend run. Re-run to obtain a new random trial.")

    with st.expander("Saved result details"):
        if classical:
            st.json(classical)
        if qkd_summary:
            st.json(qkd_summary)


def main():
    st.title("🛰️ ORION-X")
    st.caption("Orbital Risk & Intelligence Operations Network")
    st.caption("PAST: historical replay  |  PAST-ROCKET: documented rocket-body collisions  |  PRESENT: live forecasting  |  TRACK: current orbit state  |  SECURITY: QKD vs classical")

    overview_tab, historical_tab, rocket_tab, live_tab, tracker_tab, security_tab = st.tabs([
        "🏠 Overview",
        "⏪ Historical Replay / Validation",
        "🚀 Historical Rocket Collisions",
        "📡 Live Tracker / 30-Day Forecast",
        "🌍 Orbital Tracker",
        "🔐 QKD vs Classical",
    ])

    with overview_tab:
        render_overview()
    with historical_tab:
        render_historical_tab()
    with rocket_tab:
        render_historical_rocket_tab()
    with live_tab:
        render_live_tab()
    with tracker_tab:
        render_orbital_tracker()
    with security_tab:
        render_qkd_classical()


if __name__ == "__main__":
    main()
