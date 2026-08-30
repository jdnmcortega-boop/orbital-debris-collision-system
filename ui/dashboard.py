"""
Streamlit dashboard — the "product" view of the pipeline. Run with:
    streamlit run ui/dashboard.py

Ties together every stage's output: object tracking (styled after the
satellitemap.space reference), conjunctions, risk/predictions, QAE vs
classical comparison, security/QKD, and geopolitical coordination.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make the project root importable regardless of how Streamlit is launched
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

import config
from modules import data_loader
from modules import sgp4_propagation
from modules import orbital_mechanics as om
from modules import visualization as viz


st.set_page_config(
    page_title="Orbital Debris Collision Risk — Research Dashboard",
    page_icon="🛰️",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0b1220; }
    [data-testid="stMetricValue"] { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING (cached — files only re-read when their content changes)
# ============================================================

@st.cache_data
def load_all():
    data = {}

    try:
        data["orbital_data"] = data_loader.load_orbital_data()
    except Exception:
        data["orbital_data"] = None

    files = {
        "propagated": config.PROPAGATED_GRID_FILE,
        "conjunctions": config.CONJUNCTIONS_FILE,
        "predictions": config.RESULTS_DIR / "predictions.csv",
        "false_positive": config.RESULTS_DIR / "false_positive_analysis.csv",
        "qae_comparison": config.RESULTS_DIR / "qae_comparison.csv",
        "qae_sweep": config.RESULTS_DIR / "qae_accuracy_sweep.csv",
        "reentry": config.RESULTS_DIR / "reentry_analysis.csv",
        "geopolitics": config.RESULTS_DIR / "geopolitical_coordination.csv",
    }
    if path.exists():
        if key == "propagated":
            data[key] = pd.read_csv(
                path,
                usecols=[
                    "OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "TIME",
                    "X_KM", "Y_KM", "Z_KM",
                    "VX_KM_S", "VY_KM_S", "VZ_KM_S",
                ]
            )
        else:
            data[key] = pd.read_csv(path)
    else:
        data[key] = None

    json_files = {
        "classical_security": config.RESULTS_DIR / "classical_security_results.json",
        "qkd": config.RESULTS_DIR / "qkd_results.json",
        "benchmark_summary": config.RESULTS_DIR / "benchmark_summary.json",
    }
    for key, path in json_files.items():
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)
        else:
            data[key] = None

    return data


data = load_all()

st.title("🛰️ Orbital Debris Collision Risk — Research Dashboard")
st.caption("Live view of the SGP4 propagation, conjunction screening, "
           "Monte Carlo / QAE collision-probability, and secure-communication pipeline.")

tabs = st.tabs([
    "Object Tracker", "Live Tracker", "Conjunctions", "Risk & Predictions",
    "QAE vs Classical", "Security", "Geopolitical Coordination",
])


# ============================================================
# TAB 1: OBJECT TRACKER (satellitemap.space-style current-status view)
# ============================================================

with tabs[0]:
    if data["orbital_data"] is None:
        st.warning("No orbital data loaded yet — run the pipeline first.")
    else:
        orbital_df = data["orbital_data"]
        option_labels = [
            f"{r['OBJECT_NAME']} (NORAD {r['NORAD_CAT_ID']})"
            for _, r in orbital_df.iterrows()
        ]
        label_to_index = dict(zip(option_labels, orbital_df.index))

        selected_label = st.selectbox("Select object", option_labels)
        row = orbital_df.loc[label_to_index[selected_label]]
        selected_name = row["OBJECT_NAME"]
        norad_id = row["NORAD_CAT_ID"]

        apogee, perigee = om.apogee_perigee_altitude_km(row["MEAN_MOTION"], row["ECCENTRICITY"])

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Current Status")

            if data["propagated"] is not None:
                obj_prop = data["propagated"][data["propagated"]["NORAD_CAT_ID"] == norad_id]
                if len(obj_prop) > 0:
                    latest = obj_prop.iloc[0]
                    lat, lon, alt = om.eci_to_geodetic(
                        latest["X_KM"], latest["Y_KM"], latest["Z_KM"], pd.to_datetime(latest["TIME"])
                    )
                    velocity = om.velocity_magnitude_km_s(
                        latest["VX_KM_S"], latest["VY_KM_S"], latest["VZ_KM_S"]
                    )
                    st.metric("Altitude", f"{alt:,.1f} km")
                    st.metric("Velocity", f"{velocity:.2f} km/s")
                    st.metric("Sub-Earth Point", f"{lat:.2f}°, {lon:.2f}°")
                else:
                    st.info("No propagation data for this object yet.")
            else:
                st.info("Run sgp4_propagation.py to see live position data.")

            st.metric("Apogee / Perigee", f"{apogee:,.0f} / {perigee:,.0f} km")

            st.subheader("Technical Details")
            obj_type = "Debris" if "DEB" in selected_name.upper() else "Satellite"
            st.write(f"**NORAD ID:** {norad_id}")
            st.write(f"**Type:** {obj_type}")
            st.write(f"**Inclination:** {row['INCLINATION']:.2f}°")
            st.write(f"**Eccentricity:** {row['ECCENTRICITY']:.5f}")
            st.write(f"**Epoch:** {row['EPOCH']}")

        with col1:
            if data["propagated"] is not None and len(obj_prop) > 0:
                st.plotly_chart(
                    viz.ground_track_figure(data["propagated"], norad_id, selected_name),
                    use_container_width=True,
                )
                st.plotly_chart(
                    viz.altitude_history_figure(data["propagated"], norad_id, selected_name),
                    use_container_width=True,
                )
            else:
                st.info("Run sgp4_propagation.py to see the ground track and altitude history.")


# ============================================================
# TAB 2: LIVE TRACKER (real SGP4 propagation at actual current time)
# ============================================================

with tabs[1]:
    st.subheader("Live Orbit Tracker")

    view_mode = st.radio(
        "View mode", ["Continuous (click satellite for orbit path)", "Discrete refresh (fallback)"],
        horizontal=True,
    )

    if view_mode == "Continuous (click satellite for orbit path)":
        st.caption("Runs SGP4 propagation directly in your browser via satellite.js — "
                   "genuinely continuous, not tied to any server refresh interval. "
                   "Click any marker to draw its orbit path for one full period; "
                   "click again to deselect. NEW FEATURE — first-load bugs are possible; "
                   "if the map doesn't render, switch to the fallback view above.")

        if data["orbital_data"] is not None:
            from modules.live_orbit_widget import build_live_orbit_html
            html = build_live_orbit_html(data["orbital_data"], width=900, height=550)
            st.components.v1.html(html, height=600, scrolling=False)
        else:
            st.info("Load orbital data first (run data_loader.py / main.py).")

    else:
        st.caption("Recomputes every object's current position via a fresh SGP4 "
                   "propagation each refresh — real physics at the actual current "
                   "moment. Positions update in discrete steps at the chosen "
                   "refresh interval, not continuous smooth motion.")

        if "live_tracking" not in st.session_state:
            st.session_state.live_tracking = False

        col_a, col_b, col_c = st.columns([1, 1, 2])
        if col_a.button("▶ Start Live Tracking"):
            st.session_state.live_tracking = True
        if col_b.button("⏸ Stop"):
            st.session_state.live_tracking = False
        refresh_seconds = col_c.slider("Refresh interval (seconds)", 2, 30, 5)

        if data["orbital_data"] is not None:
            now_utc = datetime.now(timezone.utc)
            now_utc8 = om.to_utc8(now_utc)
            st.write(f"**Last updated:** {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC  "
                     f"/  {now_utc8.strftime('%Y-%m-%d %H:%M:%S')} UTC+8")

            current_positions, failed_now = sgp4_propagation.propagate_all_now(
                data["orbital_data"], verbose=False
            )
            if len(current_positions) > 0:
                st.plotly_chart(
                    viz.live_globe_figure(current_positions, timestamp_label=now_utc.strftime("%H:%M:%S UTC")),
                    use_container_width=True,
                )
            else:
                st.warning("No objects could be propagated to the current time.")
        else:
            st.info("Load orbital data first (run data_loader.py / main.py).")

        if st.session_state.live_tracking:
            time.sleep(refresh_seconds)
            st.rerun()


# ============================================================
# TAB 3: CONJUNCTIONS
# ============================================================

with tabs[2]:
    if data["conjunctions"] is None or len(data["conjunctions"]) == 0:
        st.warning("No conjunctions found yet — run conjunction_detection.py first "
                    "(or your screening threshold may be too tight).")
    else:
        st.subheader(f"{len(data['conjunctions'])} Screened Conjunctions")
        st.plotly_chart(viz.miss_distance_scatter(data["conjunctions"]), use_container_width=True)
        st.dataframe(data["conjunctions"], use_container_width=True)


# ============================================================
# TAB 4: RISK & PREDICTIONS
# ============================================================

with tabs[3]:
    if data["predictions"] is None:
        st.warning("No predictions yet — run monte_carlo.py then prediction.py.")
    else:
        col1, col2, col3 = st.columns(3)
        counts = data["predictions"]["RISK_LEVEL"].value_counts()
        col1.metric("LOW risk", int(counts.get("LOW", 0)))
        col2.metric("MEDIUM risk", int(counts.get("MEDIUM", 0)))
        col3.metric("HIGH risk", int(counts.get("HIGH", 0)))

        st.plotly_chart(viz.risk_level_bar(data["predictions"]), use_container_width=True)
        st.dataframe(data["predictions"], use_container_width=True)

        if data["false_positive"] is not None:
            st.subheader("False-Positive Analysis (Analytic Pc)")
            fp_rate = (data["false_positive"]["CLASSIFICATION"] == "FALSE_POSITIVE").mean()
            st.metric("False-positive rate", f"{fp_rate:.1%}")
            st.dataframe(data["false_positive"], use_container_width=True)


# ============================================================
# TAB 5: QAE vs CLASSICAL
# ============================================================

with tabs[4]:
    if data["qae_sweep"] is not None:
        st.subheader("QAE vs Classical Monte Carlo — Accuracy Sweep")
        st.plotly_chart(viz.qae_vs_mc_error_chart(data["qae_sweep"]), use_container_width=True)
        st.dataframe(data["qae_sweep"], use_container_width=True)
    else:
        st.info("Run qae_accuracy_sweep.py to see the query-budget comparison.")

    if data["qae_comparison"] is not None:
        st.subheader("QAE vs Classical MC — Real Conjunction Data")
        st.dataframe(data["qae_comparison"], use_container_width=True)


# ============================================================
# TAB 6: SECURITY (CLASSICAL vs QKD)
# ============================================================

with tabs[5]:
    st.subheader("Repeated-Trial Benchmark (50 trials/condition)")
    st.caption("Delivery rate, secure delivery rate, and detection rate require "
               "many repeated trials to report as a rate — a single run only "
               "gives one delivered/not-delivered outcome. See qkd_benchmark.py.")

    if data["benchmark_summary"] is not None:
        summaries = data["benchmark_summary"]
        st.plotly_chart(viz.benchmark_comparison_chart(summaries), use_container_width=True)

        cols = st.columns(len(summaries))
        for col, s in zip(cols, summaries):
            with col:
                st.markdown(f"**{s['condition']}**")
                st.metric("Delivery rate", f"{s.get('delivery_rate', 0):.0%}")
                if s.get("secure_delivery_rate") is not None:
                    st.metric("Secure delivery rate", f"{s['secure_delivery_rate']:.0%}")
                if s.get("detection_rate") is not None:
                    st.metric("Detection rate", f"{s['detection_rate']:.0%}")
                else:
                    st.metric("Detection rate", "N/A (no mechanism)")
                if s.get("mean_qber") is not None:
                    st.metric("Mean QBER", f"{s['mean_qber']:.1%}")
    else:
        st.info("Run qkd_benchmark.py to see repeated-trial delivery and detection rates.")

    st.divider()
    st.subheader("Single-Run Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Classical (ECDH + AES-GCM)**")
        if data["classical_security"] is not None:
            for k, v in data["classical_security"].items():
                st.write(f"**{k}:** {v}")
        else:
            st.info("Run classical_security.py.")

    with col2:
        st.markdown("**Quantum Key Distribution (BB84)**")
        if data["qkd"] is not None:
            st.plotly_chart(viz.qber_comparison_bar(data["qkd"]), use_container_width=True)
            for k, v in data["qkd"].items():
                st.write(f"**{k}:** {v}")
        else:
            st.info("Run qkd.py.")


# ============================================================
# TAB 7: GEOPOLITICAL COORDINATION
# ============================================================

with tabs[6]:
    if data["geopolitics"] is None:
        st.warning("No geopolitical assessment yet — run geopolitics.py.")
    else:
        flagged = data["geopolitics"][data["geopolitics"]["COORDINATION_REQUIRED"] == True]
        st.metric("Events requiring international coordination", len(flagged))
        st.dataframe(data["geopolitics"], use_container_width=True)