
"""
Streamlit dashboard — Orbital Debris Collision Risk Research Dashboard

Run:
    streamlit run ui/dashboard.py

Integrates:
    - Orbital object tracking
    - SGP4 propagation
    - Conjunction detection
    - Risk / prediction analysis
    - QAE vs classical Monte Carlo
    - QKD vs classical security
    - Geopolitical coordination

NOTE:
The controlled rocket/debris collision-analysis component has been
removed from this dashboard.
"""

import json
import sys
import time
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PROJECT_ROOT / "modules") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "modules"))


# ============================================================
# PROJECT IMPORTS
# ============================================================

import config

from modules import data_loader
from modules import sgp4_propagation
from modules import orbital_mechanics as om
from modules import visualization as viz


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Orbital Debris Collision Risk — Research Dashboard",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #0b1220;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f8fafc;
}

p, label {
    color: #cbd5e1;
}

[data-testid="stMetric"] {
    background-color: #1e293b;
    border-radius: 10px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

[data-testid="stMetricValue"] {
    color: #e2e8f0;
}

.stTabs [data-baseweb="tab"] {
    color: #cbd5e1;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_all():

    data = {}

    # --------------------------------------------------------
    # Orbital data
    # --------------------------------------------------------

    try:
        data["orbital_data"] = (
            data_loader.load_orbital_data()
        )
    except Exception:
        data["orbital_data"] = None

    # --------------------------------------------------------
    # CSV result files
    # --------------------------------------------------------

    files = {
        "propagated": config.PROPAGATED_GRID_FILE,

        "conjunctions": config.CONJUNCTIONS_FILE,

        "predictions":
            config.RESULTS_DIR
            / "predictions.csv",

        "false_positive":
            config.RESULTS_DIR
            / "false_positive_analysis.csv",

        "qae_comparison":
            config.RESULTS_DIR
            / "qae_comparison.csv",

        "qae_sweep":
            config.RESULTS_DIR
            / "qae_accuracy_sweep.csv",

        "reentry":
            config.RESULTS_DIR
            / "reentry_analysis.csv",

        "geopolitics":
            config.RESULTS_DIR
            / "geopolitical_coordination.csv",
    }

    for key, path in files.items():

        try:

            if path.exists():

                if key == "propagated":

                    try:

                        data[key] = pd.read_csv(
                            path,
                            usecols=[
                                "OBJECT_NAME",
                                "OBJECT_ID",
                                "NORAD_CAT_ID",
                                "TIME",
                                "X_KM",
                                "Y_KM",
                                "Z_KM",
                                "VX_KM_S",
                                "VY_KM_S",
                                "VZ_KM_S",
                            ],
                        )

                    except Exception:

                        data[key] = pd.read_csv(path)

                else:

                    data[key] = pd.read_csv(path)

            else:

                data[key] = None

        except Exception as exc:

            st.warning(
                f"Could not load {key}: {exc}"
            )

            data[key] = None

    # --------------------------------------------------------
    # JSON result files
    # --------------------------------------------------------

    json_files = {

        "classical_security":
            config.RESULTS_DIR
            / "classical_security_results.json",

        "qkd":
            config.RESULTS_DIR
            / "qkd_results.json",

        "benchmark_summary":
            config.RESULTS_DIR
            / "benchmark_summary.json",
    }

    for key, path in json_files.items():

        try:

            if path.exists():

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    data[key] = json.load(f)

            else:

                data[key] = None

        except Exception as exc:

            st.warning(
                f"Could not load {key}: {exc}"
            )

            data[key] = None

    return data


data = load_all()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🛰️ Orbital Debris Collision Risk — Research Dashboard"
)

st.caption(
    "Live SGP4 orbital tracking, conjunction screening, "
    "collision-probability estimation, QAE vs classical Monte Carlo, "
    "and secure quantum communication."
)


# ============================================================
# HEADER METRICS
# ============================================================

conjunctions = data.get("conjunctions")
predictions = data.get("predictions")
qae_comparison = data.get("qae_comparison")
qae_sweep = data.get("qae_sweep")

conjunction_count = (
    len(conjunctions)
    if conjunctions is not None
    else 0
)

prediction_count = (
    len(predictions)
    if predictions is not None
    else 0
)

qae_count = (
    len(qae_comparison)
    if qae_comparison is not None
    else 0
)

qae_sweep_count = (
    len(qae_sweep)
    if qae_sweep is not None
    else 0
)


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Conjunctions",
    conjunction_count,
)

c2.metric(
    "Predictions",
    prediction_count,
)

c3.metric(
    "QAE Comparisons",
    qae_count,
)

c4.metric(
    "QAE Sweep Tests",
    qae_sweep_count,
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "🛰️ Object Tracker",
        "📡 Live Tracker",
        "⚠️ Conjunctions",
        "📊 Risk & Predictions",
        "🔬 QAE vs Classical",
        "🔐 QKD vs Classical",
        "🌐 Geopolitical Coordination",
    ]
)


# ============================================================
# TAB 1 — OBJECT TRACKER
# ============================================================

with tabs[0]:

    st.header(
        "🛰️ Orbital Object Tracker"
    )

    orbital_df = data.get("orbital_data")
    propagated = data.get("propagated")

    if (
        orbital_df is None
        or orbital_df.empty
    ):

        st.warning(
            "No orbital data loaded yet. "
            "Run the orbital data-loading pipeline first."
        )

    else:

        option_labels = []

        for _, row in orbital_df.iterrows():

            name = str(
                row.get(
                    "OBJECT_NAME",
                    "Unknown Object",
                )
            )

            norad = row.get(
                "NORAD_CAT_ID",
                "N/A",
            )

            option_labels.append(
                f"{name} (NORAD {norad})"
            )

        label_to_index = dict(
            zip(
                option_labels,
                orbital_df.index,
            )
        )

        selected_label = st.selectbox(
            "Select orbital object",
            option_labels,
        )

        row = orbital_df.loc[
            label_to_index[selected_label]
        ]

        selected_name = str(
            row.get(
                "OBJECT_NAME",
                "Unknown",
            )
        )

        norad_id = row.get(
            "NORAD_CAT_ID",
            None,
        )

        col1, col2 = st.columns(
            [2, 1]
        )

        # ----------------------------------------------------
        # Current status
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "Current Status"
            )

            obj_prop = None

            if (
                propagated is not None
                and "NORAD_CAT_ID" in propagated.columns
                and norad_id is not None
            ):

                try:

                    obj_prop = propagated[
                        propagated[
                            "NORAD_CAT_ID"
                        ] == norad_id
                    ]

                except Exception:

                    obj_prop = None

            if (
                obj_prop is not None
                and not obj_prop.empty
            ):

                latest = obj_prop.iloc[-1]

                # Position
                try:

                    required_position = [
                        "X_KM",
                        "Y_KM",
                        "Z_KM",
                        "TIME",
                    ]

                    if all(
                        column in latest.index
                        for column in required_position
                    ):

                        timestamp = pd.to_datetime(
                            latest["TIME"]
                        )

                        lat, lon, alt = (
                            om.eci_to_geodetic(
                                latest["X_KM"],
                                latest["Y_KM"],
                                latest["Z_KM"],
                                timestamp,
                            )
                        )

                        st.metric(
                            "Altitude",
                            f"{alt:,.1f} km",
                        )

                        st.metric(
                            "Sub-Earth Point",
                            f"{lat:.2f}°, {lon:.2f}°",
                        )

                except Exception:
                    pass

                # Velocity
                try:

                    required_velocity = [
                        "VX_KM_S",
                        "VY_KM_S",
                        "VZ_KM_S",
                    ]

                    if all(
                        column in latest.index
                        for column in required_velocity
                    ):

                        velocity = (
                            om.velocity_magnitude_km_s(
                                latest["VX_KM_S"],
                                latest["VY_KM_S"],
                                latest["VZ_KM_S"],
                            )
                        )

                        st.metric(
                            "Velocity",
                            f"{velocity:.2f} km/s",
                        )

                except Exception:
                    pass

            else:

                st.info(
                    "No propagated state available "
                    "for this object."
                )

            # ------------------------------------------------
            # Orbital parameters
            # ------------------------------------------------

            try:

                if (
                    "MEAN_MOTION" in row.index
                    and "ECCENTRICITY" in row.index
                ):

                    apogee, perigee = (
                        om.apogee_perigee_altitude_km(
                            row["MEAN_MOTION"],
                            row["ECCENTRICITY"],
                        )
                    )

                    st.metric(
                        "Apogee",
                        f"{apogee:,.0f} km",
                    )

                    st.metric(
                        "Perigee",
                        f"{perigee:,.0f} km",
                    )

            except Exception:
                pass

            st.subheader(
                "Technical Details"
            )

            st.write(
                f"**NORAD ID:** {norad_id}"
            )

            obj_type = (
                "Debris"
                if "DEB"
                in selected_name.upper()
                else "Satellite"
            )

            st.write(
                f"**Type:** {obj_type}"
            )

            if "INCLINATION" in row.index:

                try:

                    st.write(
                        f"**Inclination:** "
                        f"{float(row['INCLINATION']):.2f}°"
                    )

                except Exception:
                    pass

            if "ECCENTRICITY" in row.index:

                try:

                    st.write(
                        f"**Eccentricity:** "
                        f"{float(row['ECCENTRICITY']):.5f}"
                    )

                except Exception:
                    pass

            if "EPOCH" in row.index:

                st.write(
                    f"**Epoch:** {row['EPOCH']}"
                )

        # ----------------------------------------------------
        # Charts
        # ----------------------------------------------------

        with col1:

            if (
                propagated is not None
                and obj_prop is not None
                and not obj_prop.empty
            ):

                try:

                    st.subheader(
                        "Ground Track"
                    )

                    st.plotly_chart(
                        viz.ground_track_figure(
                            propagated,
                            norad_id,
                            selected_name,
                        ),
                        width="stretch",
                    )

                except Exception as exc:

                    st.info(
                        f"Ground track unavailable: {exc}"
                    )

                try:

                    st.subheader(
                        "Altitude History"
                    )

                    st.plotly_chart(
                        viz.altitude_history_figure(
                            propagated,
                            norad_id,
                            selected_name,
                        ),
                        width="stretch",
                    )

                except Exception as exc:

                    st.info(
                        f"Altitude history unavailable: {exc}"
                    )

            else:

                st.info(
                    "Run sgp4_propagation.py to generate "
                    "ground-track and altitude-history data."
                )


# ============================================================
# TAB 2 — LIVE TRACKER
# ============================================================

with tabs[1]:

    st.header(
        "📡 Live Orbit Tracker"
    )

    st.caption(
        "Real-time orbital visualization using current orbital "
        "elements and SGP4 propagation."
    )

    view_mode = st.radio(
        "View mode",
        [
            "Continuous (click satellite for orbit path)",
            "Discrete refresh (fallback)",
        ],
        horizontal=True,
    )

    # ========================================================
    # CONTINUOUS
    # ========================================================

    if (
        view_mode
        == "Continuous (click satellite for orbit path)"
    ):

        st.write(
            """
            The continuous tracker propagates orbital objects
            directly in the browser using the live-orbit
            visualization module.
            """
        )

        if data["orbital_data"] is not None:

            try:

                from modules.live_orbit_widget import (
                    build_live_orbit_html
                )

                html = build_live_orbit_html(
                    data["orbital_data"],
                    width=1100,
                    height=600,
                )

                st.components.v1.html(
                    html,
                    height=650,
                    scrolling=False,
                )

            except Exception as exc:

                st.error(
                    f"Continuous live tracker unavailable: {exc}"
                )

        else:

            st.info(
                "Load orbital data first."
            )

    # ========================================================
    # DISCRETE
    # ========================================================

    else:

        st.write(
            """
            The fallback tracker recalculates the current
            position of the tracked objects using SGP4
            whenever the dashboard refreshes.
            """
        )

        if "live_tracking" not in st.session_state:

            st.session_state.live_tracking = False

        col_a, col_b, col_c = st.columns(
            [1, 1, 2]
        )

        with col_a:

            if st.button(
                "▶ Start Live Tracking",
                width="stretch",
            ):

                st.session_state.live_tracking = True

        with col_b:

            if st.button(
                "⏸ Stop",
                width="stretch",
            ):

                st.session_state.live_tracking = False

        with col_c:

            refresh_seconds = st.slider(
                "Refresh interval (seconds)",
                min_value=2,
                max_value=30,
                value=5,
            )

        if data["orbital_data"] is None:

            st.info(
                "Load orbital data first."
            )

        else:

            now_utc = datetime.now(
                timezone.utc
            )

            try:

                now_utc8 = om.to_utc8(
                    now_utc
                )

                st.write(
                    f"**Last updated:** "
                    f"{now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC "
                    f"/ "
                    f"{now_utc8.strftime('%Y-%m-%d %H:%M:%S')} UTC+8"
                )

            except Exception:

                st.write(
                    f"**Last updated:** "
                    f"{now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )

            try:

                current_positions, failed_now = (
                    sgp4_propagation.propagate_all_now(
                        data["orbital_data"],
                        verbose=False,
                    )
                )

            except Exception as exc:

                current_positions = None

                st.error(
                    f"Live propagation failed: {exc}"
                )

            if (
                current_positions is not None
                and len(current_positions) > 0
            ):

                try:

                    st.plotly_chart(
                        viz.live_globe_figure(
                            current_positions,
                            timestamp_label=now_utc.strftime(
                                "%H:%M:%S UTC"
                            ),
                        ),
                        width="stretch",
                    )

                except Exception as exc:

                    st.error(
                        f"Live globe visualization failed: {exc}"
                    )

                try:

                    st.dataframe(
                        current_positions,
                        width="stretch",
                    )

                except Exception:
                    pass

            elif current_positions is not None:

                st.warning(
                    "No objects could be propagated "
                    "to the current time."
                )

        if st.session_state.live_tracking:

            time.sleep(
                refresh_seconds
            )

            st.rerun()


# ============================================================
# TAB 3 — CONJUNCTIONS
# ============================================================

with tabs[2]:

    st.header(
        "⚠️ Conjunction Detection"
    )

    conjunctions = data.get(
        "conjunctions"
    )

    if (
        conjunctions is None
        or conjunctions.empty
    ):

        st.warning(
            "No conjunctions found yet. "
            "Run conjunction_detection.py first."
        )

    else:

        st.subheader(
            f"{len(conjunctions):,} Screened Conjunctions"
        )

        # ----------------------------------------------------
        # Closest distance
        # ----------------------------------------------------

        if "MISS_DISTANCE_KM" in conjunctions.columns:

            try:

                minimum_distance = (
                    pd.to_numeric(
                        conjunctions[
                            "MISS_DISTANCE_KM"
                        ],
                        errors="coerce",
                    )
                    .min()
                )

                st.metric(
                    "Closest Miss Distance",
                    f"{minimum_distance:.6f} km",
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # Relative velocity
        # ----------------------------------------------------

        if (
            "RELATIVE_VELOCITY_KM_S"
            in conjunctions.columns
        ):

            try:

                maximum_velocity = (
                    pd.to_numeric(
                        conjunctions[
                            "RELATIVE_VELOCITY_KM_S"
                        ],
                        errors="coerce",
                    )
                    .max()
                )

                st.metric(
                    "Maximum Relative Velocity",
                    f"{maximum_velocity:.3f} km/s",
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        try:

            st.plotly_chart(
                viz.miss_distance_scatter(
                    conjunctions
                ),
                width="stretch",
            )

        except Exception as exc:

            st.info(
                f"Conjunction visualization unavailable: {exc}"
            )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        st.subheader(
            "Conjunction Results"
        )

        st.dataframe(
            conjunctions,
            width="stretch",
        )


# ============================================================
# TAB 4 — RISK & PREDICTIONS
# ============================================================

with tabs[3]:

    st.header(
        "📊 Risk & Predictions"
    )

    predictions_df = data.get(
        "predictions"
    )

    if (
        predictions_df is None
        or predictions_df.empty
    ):

        st.warning(
            "No prediction results found."
        )

        st.code(
            "python -m modules.monte_carlo",
            language="powershell",
        )

        st.code(
            "python -m modules.prediction",
            language="powershell",
        )

    else:

        # ----------------------------------------------------
        # Determine risk column
        # ----------------------------------------------------

        if (
            "COMPOSITE_RISK_LEVEL"
            in predictions_df.columns
        ):

            risk_column = (
                "COMPOSITE_RISK_LEVEL"
            )

        elif (
            "RISK_LEVEL"
            in predictions_df.columns
        ):

            risk_column = "RISK_LEVEL"

        else:

            risk_column = None

        if risk_column is not None:

            counts = (
                predictions_df[
                    risk_column
                ]
                .astype(str)
                .str.upper()
                .value_counts()
            )

            low = int(
                counts.get(
                    "LOW",
                    0,
                )
            )

            medium = int(
                counts.get(
                    "MEDIUM",
                    0,
                )
            )

            high = int(
                counts.get(
                    "HIGH",
                    0,
                )
            )

            critical = int(
                counts.get(
                    "CRITICAL",
                    0,
                )
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "LOW risk",
                low,
            )

            c2.metric(
                "MEDIUM risk",
                medium,
            )

            c3.metric(
                "HIGH risk",
                high,
            )

            c4.metric(
                "CRITICAL",
                critical,
            )

            try:

                st.plotly_chart(
                    viz.risk_level_bar(
                        predictions_df,
                        risk_column=risk_column,
                    ),
                    width="stretch",
                )

            except Exception:

                chart_df = pd.DataFrame(
                    {
                        "Risk Level": [
                            "LOW",
                            "MEDIUM",
                            "HIGH",
                            "CRITICAL",
                        ],
                        "Count": [
                            low,
                            medium,
                            high,
                            critical,
                        ],
                    }
                )

                st.bar_chart(
                    chart_df.set_index(
                        "Risk Level"
                    ),
                    width="stretch",
                )

        # ----------------------------------------------------
        # Prediction results
        # ----------------------------------------------------

        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            predictions_df,
            width="stretch",
        )

        # ----------------------------------------------------
        # False positives
        # ----------------------------------------------------

        false_positive = data.get(
            "false_positive"
        )

        if (
            false_positive is not None
            and not false_positive.empty
        ):

            st.subheader(
                "False-Positive Analysis"
            )

            if "CLASSIFICATION" in false_positive.columns:

                fp_rate = (
                    false_positive[
                        "CLASSIFICATION"
                    ]
                    .astype(str)
                    .str.upper()
                    .eq("FALSE_POSITIVE")
                    .mean()
                )

                st.metric(
                    "False-positive rate",
                    f"{fp_rate:.1%}",
                )

            st.dataframe(
                false_positive,
                width="stretch",
            )

        # ----------------------------------------------------
        # Re-entry
        # ----------------------------------------------------

        reentry = data.get(
            "reentry"
        )

        if (
            reentry is not None
            and not reentry.empty
        ):

            st.subheader(
                "Re-entry Risk Analysis"
            )

            st.dataframe(
                reentry,
                width="stretch",
            )


# ============================================================
# TAB 5 — QAE VS CLASSICAL
# ============================================================

with tabs[4]:

    st.header(
        "🔬 QAE vs Classical Monte Carlo"
    )

    st.write(
        """
        This section compares Quantum Amplitude Estimation (QAE)
        with classical Monte Carlo estimation of collision
        probability.
        """
    )

    qae = data.get(
        "qae_comparison"
    )

    qae_sweep = data.get(
        "qae_sweep"
    )


    # ========================================================
    # QUERY-BUDGET ACCURACY SWEEP
    # ========================================================

    st.subheader(
        "Query-Budget Accuracy Comparison"
    )

    st.write(
        """
        The accuracy experiment compares QAE and classical
        Monte Carlo estimation error as the computational
        query/sample budget increases.
        """
    )

    if (
        qae_sweep is not None
        and not qae_sweep.empty
    ):

        required = {
            "TRUE_PROBABILITY",
            "ORACLE_CALLS",
            "QAE_ERROR_MEAN",
            "MC_ERROR_MEAN",
        }

        if required.issubset(
            qae_sweep.columns
        ):

            probabilities = sorted(
                pd.to_numeric(
                    qae_sweep[
                        "TRUE_PROBABILITY"
                    ],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )

            if probabilities:

                selected_probability = (
                    st.selectbox(
                        "Select true probability",
                        probabilities,
                        format_func=lambda x:
                        f"{x:g}",
                        key="qae_probability",
                    )
                )

                graph_df = qae_sweep[
                    pd.to_numeric(
                        qae_sweep[
                            "TRUE_PROBABILITY"
                        ],
                        errors="coerce",
                    )
                    == selected_probability
                ].copy()

                numeric_columns = [
                    "ORACLE_CALLS",
                    "QAE_ERROR_MEAN",
                    "MC_ERROR_MEAN",
                ]

                for column in numeric_columns:

                    graph_df[column] = (
                        pd.to_numeric(
                            graph_df[column],
                            errors="coerce",
                        )
                    )

                graph_df = (
                    graph_df
                    .dropna(
                        subset=numeric_columns
                    )
                    .sort_values(
                        "ORACLE_CALLS"
                    )
                )

                if not graph_df.empty:

                    try:

                        st.plotly_chart(
                            viz.qae_vs_mc_error_chart(
                                graph_df
                            ),
                            width="stretch",
                        )

                    except Exception:

                        graph_df[
                            "Query Budget"
                        ] = (
                            graph_df[
                                "ORACLE_CALLS"
                            ]
                            .astype(int)
                            .map(
                                lambda x:
                                f"{x:,}"
                            )
                        )

                        chart_df = (
                            graph_df[
                                [
                                    "Query Budget",
                                    "QAE_ERROR_MEAN",
                                    "MC_ERROR_MEAN",
                                ]
                            ]
                            .set_index(
                                "Query Budget"
                            )
                        )

                        st.line_chart(
                            chart_df,
                            width="stretch",
                        )

                    st.caption(
                        "Lower estimation error is better. "
                        "QAE uses quantum oracle calls while "
                        "Monte Carlo uses classical samples."
                    )

                    # ------------------------------------------------
                    # Metrics
                    # ------------------------------------------------

                    wins = 0

                    if "QAE_WINS" in graph_df.columns:

                        wins = int(
                            graph_df[
                                "QAE_WINS"
                            ]
                            .fillna(False)
                            .astype(bool)
                            .sum()
                        )

                    total = len(
                        graph_df
                    )

                    q1, q2, q3 = st.columns(3)

                    q1.metric(
                        "Test budgets",
                        total,
                    )

                    q2.metric(
                        "QAE wins",
                        wins,
                    )

                    q3.metric(
                        "QAE win rate",
                        (
                            f"{wins / total * 100:.1f}%"
                            if total
                            else "N/A"
                        ),
                    )

                    # ------------------------------------------------
                    # Detailed sweep
                    # ------------------------------------------------

                    st.subheader(
                        "Query-Budget Results"
                    )

                    sweep_columns = [
                        "TRUE_PROBABILITY",
                        "EVAL_QUBITS",
                        "ORACLE_CALLS",
                        "N_TRIALS",
                        "QAE_ERROR_MEAN",
                        "MC_ERROR_MEAN",
                        "QAE_WINS",
                    ]

                    sweep_columns = [
                        column
                        for column in sweep_columns
                        if column in graph_df.columns
                    ]

                    st.dataframe(
                        graph_df[
                            sweep_columns
                        ],
                        width="stretch",
                    )

            else:

                st.warning(
                    "No valid TRUE_PROBABILITY values found."
                )

        else:

            st.warning(
                "QAE accuracy sweep is missing required columns."
            )

    else:

        st.info(
            "No QAE accuracy sweep found."
        )


    # ========================================================
    # REAL CONJUNCTION DATA
    # ========================================================

    st.subheader(
        "QAE vs Classical Monte Carlo — Real Conjunction Data"
    )

    if (
        qae is not None
        and not qae.empty
    ):

        # ----------------------------------------------------
        # Summary metrics
        # ----------------------------------------------------

        q1, q2 = st.columns(2)

        if "QAE_ORACLE_CALLS" in qae.columns:

            try:

                qae_calls = (
                    pd.to_numeric(
                        qae[
                            "QAE_ORACLE_CALLS"
                        ],
                        errors="coerce",
                    )
                    .dropna()
                )

                if not qae_calls.empty:

                    q1.metric(
                        "QAE Oracle Calls",
                        f"{int(qae_calls.iloc[0]):,}",
                    )

            except Exception:
                pass

        if "MC_SAMPLES" in qae.columns:

            try:

                mc_samples = (
                    pd.to_numeric(
                        qae[
                            "MC_SAMPLES"
                        ],
                        errors="coerce",
                    )
                    .dropna()
                )

                if not mc_samples.empty:

                    q2.metric(
                        "Monte Carlo Samples",
                        f"{int(mc_samples.iloc[0]):,}",
                    )

            except Exception:
                pass


        # ====================================================
        # ERROR COMPARISON
        # ====================================================

        if {
            "QAE_ERROR",
            "MC_ERROR",
        }.issubset(
            qae.columns
        ):

            st.subheader(
                "Estimation Error by Conjunction"
            )

            error_df = qae[
                [
                    "QAE_ERROR",
                    "MC_ERROR",
                ]
            ].copy()

            error_df[
                "QAE_ERROR"
            ] = pd.to_numeric(
                error_df[
                    "QAE_ERROR"
                ],
                errors="coerce",
            )

            error_df[
                "MC_ERROR"
            ] = pd.to_numeric(
                error_df[
                    "MC_ERROR"
                ],
                errors="coerce",
            )

            error_df = (
                error_df
                .dropna()
            )

            if not error_df.empty:

                if {
                    "OBJECT_A",
                    "OBJECT_B",
                }.issubset(
                    qae.columns
                ):

                    labels = (
                        qae[
                            "OBJECT_A"
                        ]
                        .astype(str)
                        + " ↔ "
                        + qae[
                            "OBJECT_B"
                        ].astype(str)
                    )

                    valid_indices = (
                        error_df.index
                    )

                    error_df.index = (
                        labels.loc[
                            valid_indices
                        ]
                    )

                st.bar_chart(
                    error_df,
                    width="stretch",
                )


        # ====================================================
        # COLLISION-PROBABILITY ESTIMATES
        # ====================================================

        if {
            "ANALYTIC_PC",
            "QAE_ESTIMATE",
            "MC_ESTIMATE",
        }.issubset(
            qae.columns
        ):

            st.subheader(
                "Collision-Probability Estimates"
            )

            probability_columns = [
                "ANALYTIC_PC",
                "QAE_ESTIMATE",
                "MC_ESTIMATE",
            ]

            probability_df = qae[
                probability_columns
            ].copy()

            # ------------------------------------------------
            # Convert everything to numeric
            # ------------------------------------------------

            for column in probability_columns:

                probability_df[column] = (
                    pd.to_numeric(
                        probability_df[column],
                        errors="coerce",
                    )
                )

            probability_df = (
                probability_df
                .dropna(
                    how="all"
                )
            )

            if not probability_df.empty:

                # =================================================
                # LOG-SCALE TRANSFORMATION
                #
                # The actual probabilities can be extremely small,
                # e.g. 10^-7 to 10^-11.
                #
                # We therefore use:
                #
                #     -log10(P)
                #
                # This is ONLY a visualization transformation.
                # The underlying QAE / MC values are unchanged.
                # =================================================

                probability_plot = pd.DataFrame(
                    index=probability_df.index
                )

                for column in probability_columns:

                    values = (
                        probability_df[
                            column
                        ]
                    )

                    # Preserve the existing methodology:
                    # values below 1e-15 are represented at
                    # 1e-15 for log visualization.
                    safe_values = (
                        values
                        .clip(
                            lower=1e-15
                        )
                    )

                    probability_plot[
                        column
                    ] = safe_values.apply(
                        lambda p:
                        -math.log10(p)
                        if pd.notna(p)
                        else float("nan")
                    )

                # =================================================
                # FIX:
                #
                # Use Plotly directly instead of st.line_chart.
                #
                # QAE is intentionally added LAST.
                #
                # Therefore if:
                #
                #     QAE_ESTIMATE = 0
                #     MC_ESTIMATE  = 0
                #
                # both become 15 after the visualization transform,
                # but the PINK QAE trace is drawn on top of the
                # BLUE Monte Carlo trace and remains visible.
                #
                # Markers are also enabled so individual QAE values
                # can be distinguished.
                # =================================================

                fig = go.Figure()

                # ------------------------------------------------
                # Analytic PC
                # ------------------------------------------------

                if (
                    "ANALYTIC_PC"
                    in probability_plot.columns
                ):

                    fig.add_trace(
                        go.Scatter(
                            x=list(
                                range(
                                    len(
                                        probability_plot
                                    )
                                )
                            ),
                            y=probability_plot[
                                "ANALYTIC_PC"
                            ],
                            mode="lines+markers",
                            name="Analytic PC (-log10)",
                            line={
                                "width": 2,
                            },
                            marker={
                                "size": 6,
                            },
                        )
                    )

                # ------------------------------------------------
                # Monte Carlo
                # ------------------------------------------------

                if (
                    "MC_ESTIMATE"
                    in probability_plot.columns
                ):

                    fig.add_trace(
                        go.Scatter(
                            x=list(
                                range(
                                    len(
                                        probability_plot
                                    )
                                )
                            ),
                            y=probability_plot[
                                "MC_ESTIMATE"
                            ],
                            mode="lines+markers",
                            name="Monte Carlo Estimate (-log10)",
                            line={
                                "width": 2,
                            },
                            marker={
                                "size": 6,
                            },
                        )
                    )

                # ------------------------------------------------
                # QAE — DRAW LAST
                # ------------------------------------------------

                if (
                    "QAE_ESTIMATE"
                    in probability_plot.columns
                ):

                    fig.add_trace(
                        go.Scatter(
                            x=list(
                                range(
                                    len(
                                        probability_plot
                                    )
                                )
                            ),
                            y=probability_plot[
                                "QAE_ESTIMATE"
                            ],
                            mode="lines+markers",
                            name="QAE Estimate (-log10)",
                            line={
                                "width": 3,
                            },
                            marker={
                                "size": 8,
                            },
                        )
                    )

                fig.update_layout(
                    title="Collision-Probability Estimates",
                    xaxis_title="Conjunction",
                    yaxis_title="−log₁₀(P)",
                    template="plotly_dark",
                    hovermode="x unified",
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": -0.25,
                        "xanchor": "left",
                        "x": 0,
                    },
                    margin={
                        "l": 60,
                        "r": 30,
                        "t": 70,
                        "b": 100,
                    },
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

                st.caption(
                    "Collision probabilities are shown as −log₁₀(P) "
                    "because the raw probabilities are extremely small. "
                    "Higher values indicate smaller probabilities. "
                    "The underlying QAE and Monte Carlo estimates are "
                    "not modified."
                )

                # ------------------------------------------------
                # Show raw probabilities directly underneath
                # ------------------------------------------------

                st.subheader(
                    "Raw Probability Values"
                )

                raw_probability_df = (
                    probability_df.copy()
                )

                raw_probability_df.columns = [
                    "Analytic PC",
                    "QAE Estimate",
                    "Monte Carlo Estimate",
                ]

                st.dataframe(
                    raw_probability_df,
                    width="stretch",
                )


        # ====================================================
        # FULL QAE RESULTS
        # ====================================================

        st.subheader(
            "QAE Comparison Results"
        )

        st.dataframe(
            qae,
            width="stretch",
        )

    else:

        st.info(
            "No QAE comparison results found."
        )


# ============================================================
# TAB 6 — QKD VS CLASSICAL
# ============================================================

with tabs[5]:

    st.header(
        "🔐 QKD vs Classical Security"
    )

    st.write(
        """
        This section presents the security layer used to protect
        the exchange of sensitive orbital and collision-risk
        information.
        """
    )

    qkd_results = data.get(
        "qkd"
    )

    classical_results = data.get(
        "classical_security"
    )

    benchmark_summary = data.get(
        "benchmark_summary"
    )


    # ========================================================
    # REPEATED-TRIAL BENCHMARK
    # ========================================================

    st.subheader(
        "Repeated-Trial Benchmark"
    )

    st.caption(
        "Repeated trials provide delivery-rate, secure-delivery, "
        "detection, and QBER measurements."
    )

    if (
        benchmark_summary is not None
        and isinstance(
            benchmark_summary,
            list,
        )
        and benchmark_summary
    ):

        try:

            st.plotly_chart(
                viz.benchmark_comparison_chart(
                    benchmark_summary
                ),
                width="stretch",
            )

        except Exception as exc:

            st.info(
                f"Benchmark chart unavailable: {exc}"
            )

        cols = st.columns(
            len(
                benchmark_summary
            )
        )

        for col, summary in zip(
            cols,
            benchmark_summary,
        ):

            with col:

                condition = summary.get(
                    "condition",
                    "Unknown",
                )

                st.markdown(
                    f"**{condition}**"
                )

                st.metric(
                    "Delivery rate",
                    f"{summary.get('delivery_rate', 0):.0%}",
                )

                if (
                    summary.get(
                        "secure_delivery_rate"
                    )
                    is not None
                ):

                    st.metric(
                        "Secure delivery rate",
                        f"{summary['secure_delivery_rate']:.0%}",
                    )

                if (
                    summary.get(
                        "detection_rate"
                    )
                    is not None
                ):

                    st.metric(
                        "Detection rate",
                        f"{summary['detection_rate']:.0%}",
                    )

                else:

                    st.metric(
                        "Detection rate",
                        "N/A",
                    )

                if (
                    summary.get(
                        "mean_qber"
                    )
                    is not None
                ):

                    st.metric(
                        "Mean QBER",
                        f"{summary['mean_qber']:.1%}",
                    )

    else:

        st.info(
            "No repeated-trial benchmark found. "
            "Run qkd_benchmark.py."
        )


    # ========================================================
    # SINGLE-RUN COMPARISON
    # ========================================================

    st.divider()

    st.subheader(
        "Single-Run Comparison"
    )

    col1, col2 = st.columns(2)


    # ========================================================
    # CLASSICAL
    # ========================================================

    with col1:

        st.markdown(
            "### Classical Security"
        )

        st.caption(
            "ECDH + AES-GCM"
        )

        if classical_results is not None:

            if isinstance(
                classical_results,
                dict,
            ):

                for key, value in (
                    classical_results.items()
                ):

                    st.write(
                        f"**{key}:** {value}"
                    )

            else:

                st.write(
                    classical_results
                )

        else:

            st.info(
                "Run classical_security.py."
            )


    # ========================================================
    # QKD
    # ========================================================

    with col2:

        st.markdown(
            "### Quantum Key Distribution"
        )

        st.caption(
            "BB84"
        )

        if qkd_results is not None:

            if isinstance(
                qkd_results,
                dict,
            ):

                try:

                    st.plotly_chart(
                        viz.qber_comparison_bar(
                            qkd_results
                        ),
                        width="stretch",
                    )

                except Exception:
                    pass

                for key, value in (
                    qkd_results.items()
                ):

                    st.write(
                        f"**{key}:** {value}"
                    )

            else:

                st.write(
                    qkd_results
                )

        else:

            st.info(
                "Run qkd.py."
            )


# ============================================================
# TAB 7 — GEOPOLITICAL COORDINATION
# ============================================================

with tabs[6]:

    st.header(
        "🌐 Geopolitical Coordination"
    )

    geopolitics = data.get(
        "geopolitics"
    )

    if (
        geopolitics is None
        or geopolitics.empty
    ):

        st.info(
            "No geopolitical coordination results found."
        )

        st.code(
            "python -m modules.geopolitics",
            language="powershell",
        )

    else:

        st.write(
            """
            This section presents coordination information for
            potential orbital collision events, including affected
            operators/countries and notification information.
            """
        )

        if (
            "COORDINATION_REQUIRED"
            in geopolitics.columns
        ):

            flagged = geopolitics[
                geopolitics[
                    "COORDINATION_REQUIRED"
                ]
                .astype(str)
                .str.upper()
                .isin(
                    [
                        "TRUE",
                        "1",
                        "YES",
                    ]
                )
            ]

            st.metric(
                "Events requiring international coordination",
                len(flagged),
            )

        st.dataframe(
            geopolitics,
            width="stretch",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Orbital Debris Collision Detection System | "
    "QAE vs Classical Monte Carlo | "
    "QKD Security | Research Prototype"
)