"""
Two-mode research dashboard for orbital-debris collision forecasting.

PAST / HISTORICAL:
    Replays a confirmed collision using archived historical TLE/3LE data.
    The prediction stage only sees element sets whose epoch is <= the replay
    timestamp. The known collision is used only as ground truth.

PRESENT / LIVE:
    Uses the current orbital dataset and existing 30-day forecast outputs.

Run directly with:
    streamlit run ui/historical_live_dashboard.py

The main dashboard can import this module's render functions without changing
any of the scientific calculation modules.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from modules import data_loader
from modules import historical_replay
from modules.historical_replay import HistoricalEvent, merge_tle_archives
from modules.historical_validation import (
    HistoricalValidationEvent,
    run_historical_validation,
    utc_datetime,
)
from modules import orbital_mechanics as om
from modules import sgp4_propagation
from modules import visualization as viz


HISTORICAL_EVENTS_FILE = PROJECT_ROOT / "data" / "historical_events" / "events.csv"
HISTORICAL_ROOT = PROJECT_ROOT / "data" / "historical_events"
HISTORICAL_RESULTS_DIR = PROJECT_ROOT / "results" / "historical"


@st.cache_data(show_spinner=False)
def load_current_orbital_data():
    try:
        return data_loader.load_orbital_data()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_csv(path_text: str):
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_historical_catalog():
    if not HISTORICAL_EVENTS_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORICAL_EVENTS_FILE)
    except Exception:
        return pd.DataFrame()

    if "status_for_backtest" in df.columns:
        df["AVAILABLE_FOR_REPLAY"] = False
        for idx, row in df.iterrows():
            event_id = str(row.get("event_id", ""))
            event_dir = HISTORICAL_ROOT / event_id.lower()
            if event_dir.exists():
                archive_files = list(event_dir.glob("*.tle")) + list(event_dir.glob("*.3le"))
                df.loc[idx, "AVAILABLE_FOR_REPLAY"] = bool(archive_files)
    else:
        df["AVAILABLE_FOR_REPLAY"] = False

    return df


def find_event_archives(event_id: str):
    event_dir = HISTORICAL_ROOT / event_id.lower()
    if not event_dir.exists():
        return []
    files = list(event_dir.glob("*.tle")) + list(event_dir.glob("*.3le"))
    return sorted(set(files))


def format_days(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1f} d"


def format_probability(value):
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if value == 0:
        return "0"
    return f"{value:.3e}"


def format_warning_level(value):
    """Return a readable warning level without changing the underlying result."""
    if value is None or pd.isna(value):
        return "—"
    return str(value).strip().upper()


def build_historical_event(row) -> HistoricalValidationEvent:
    return HistoricalValidationEvent(
        event_id=str(row["event_id"]),
        event_time_utc=utc_datetime(str(row["event_time_utc"])),
        object_a_norad=int(row["object_a_norad"]),
        object_b_norad=int(row["object_b_norad"]),
    )


@st.cache_data(show_spinner=False)
def execute_historical_validation(
    event_id: str,
    event_time_text: str,
    norad_a: int,
    norad_b: int,
    archive_text: tuple[str, ...],
    rewind_days: int,
    snapshot_step_hours: int,
    forecast_step_minutes: int,
    qae_eval_qubits: int,
    qae_shots: int,
):
    event = HistoricalValidationEvent(
        event_id=event_id,
        event_time_utc=utc_datetime(event_time_text),
        object_a_norad=norad_a,
        object_b_norad=norad_b,
    )
    archives = merge_tle_archives(Path(p) for p in archive_text)
    return run_historical_validation(
        event,
        archives,
        rewind_days=rewind_days,
        snapshot_step_hours=snapshot_step_hours,
        forecast_step_minutes=forecast_step_minutes,
        qae_eval_qubits=qae_eval_qubits,
        qae_shots=qae_shots,
        sigma_km=getattr(config, "POSITION_UNCERTAINTY_KM", None),
        hard_body_radius_km=getattr(config, "HARD_BODY_RADIUS_KM", None),
    )


def historical_replay_figure(result: pd.DataFrame, selected_day: float | None = None):
    fig = go.Figure()
    x = pd.to_numeric(result["DAYS_BEFORE_EVENT"], errors="coerce")

    if "FORECAST_MISS_DISTANCE_KM" in result.columns:
        y = pd.to_numeric(result["FORECAST_MISS_DISTANCE_KM"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name="Forecast miss distance",
                hovertemplate="T-%{x:.1f} d<br>%{y:.6f} km<extra></extra>",
            )
        )

    fig.add_hline(
        y=0.02,
        line_dash="dash",
        annotation_text="Hard-body radius",
        annotation_position="top left",
    )
    fig.add_vline(x=0, line_dash="dot", annotation_text="Actual event")

    if selected_day is not None:
        fig.add_vline(
            x=selected_day,
            line_dash="dash",
            annotation_text="Replay position",
            annotation_position="top right",
        )

    fig.update_xaxes(title="Days before actual collision", autorange="reversed")
    fig.update_yaxes(title="Forecast closest-approach distance (km)", type="log")
    fig.update_layout(height=480, margin=dict(l=50, r=30, t=50, b=50))
    return fig


def qae_mc_figure(result: pd.DataFrame):
    fig = go.Figure()
    x = pd.to_numeric(result["DAYS_BEFORE_EVENT"], errors="coerce")

    for column, name in [
        ("ANALYTIC_PC", "Analytic probability"),
        ("QAE_ESTIMATE", "QAE estimate"),
        ("MC_ESTIMATE", "Monte Carlo estimate"),
    ]:
        if column in result.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pd.to_numeric(result[column], errors="coerce"),
                    mode="lines+markers",
                    name=name,
                )
            )

    fig.add_vline(x=0, line_dash="dot", annotation_text="Actual event")
    fig.update_xaxes(title="Days before actual collision", autorange="reversed")
    fig.update_yaxes(title="Collision probability", type="log")
    fig.update_layout(height=480, margin=dict(l=50, r=30, t=50, b=50))
    return fig


def render_historical_tab():
    st.header("⏪ Historical Replay & Validation")
    st.caption(
        "PAST: replay a confirmed collision using only historical orbital information "
        "that would have been available at each prediction timestamp."
    )

    catalog = load_historical_catalog()
    if catalog.empty:
        st.error("Historical event catalog is unavailable.")
        return

    confirmed = catalog.copy()
    if "event_type" in confirmed.columns:
        confirmed = confirmed[confirmed["event_type"].notna()]

    available = confirmed[confirmed["AVAILABLE_FOR_REPLAY"] == True].copy()

    if available.empty:
        st.warning(
            "No historical event currently has an archived TLE/3LE file. "
            "Add verified historical archives under data/historical_events/<event_id>/."
        )
        st.dataframe(catalog, width="stretch")
        return

    labels = []
    label_to_index = {}
    for idx, row in available.iterrows():
        label = (
            f"{row['event_name']} — {row['event_time_utc']} "
            f"(NORAD {int(row['object_a_norad'])} / {int(row['object_b_norad'])})"
        )
        labels.append(label)
        label_to_index[label] = idx

    selected = st.selectbox("Historical collision dataset", labels)
    row = available.loc[label_to_index[selected]]
    archives = find_event_archives(str(row["event_id"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Event", str(row["event_id"]))
    c2.metric("Object A", int(row["object_a_norad"]))
    c3.metric("Object B", int(row["object_b_norad"]))
    c4.metric("Replay archive files", len(archives))

    st.info(
        "Ground truth is kept separate from prediction. The collision timestamp is "
        "the evaluation boundary; future TLEs are never used for an earlier replay snapshot."
    )

    with st.expander("Historical event details", expanded=True):
        st.write(f"**Event:** {row['event_name']}")
        st.write(f"**Event time:** {row['event_time_utc']}")
        st.write(f"**Object A:** {row['object_a']} — NORAD {int(row['object_a_norad'])}")
        st.write(f"**Object B:** {row['object_b']} — NORAD {int(row['object_b_norad'])}")
        st.write(f"**Status:** {row.get('status_for_backtest', 'confirmed')}")
        st.write(f"**Archive:** {', '.join(str(p.relative_to(PROJECT_ROOT)) for p in archives)}")

    with st.expander("Replay controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            rewind_days = st.number_input("Rewind (days)", 1, 90, 30, 1)
        with c2:
            snapshot_step_hours = st.selectbox("Replay step", [6, 12, 24], index=2)
        with c3:
            forecast_step_minutes = st.selectbox("Forecast step", [30, 60, 120], index=0)

        c1, c2 = st.columns(2)
        with c1:
            qae_eval_qubits = st.slider("QAE evaluation qubits", 4, 10, 6)
        with c2:
            qae_shots = st.slider("QAE shots", 50, 500, 200, 50)

        run = st.button("▶ Run historical replay + QAE/MC validation", type="primary", width="stretch")

    result = st.session_state.get("historical_validation_result")
    result_key = st.session_state.get("historical_validation_key")
    current_key = (
        str(row["event_id"]),
        int(rewind_days),
        int(snapshot_step_hours),
        int(forecast_step_minutes),
        int(qae_eval_qubits),
        int(qae_shots),
    )

    if run:
        try:
            with st.spinner("Replaying the historical encounter and running QAE/MC validation…"):
                result = execute_historical_validation(
                    str(row["event_id"]),
                    str(row["event_time_utc"]),
                    int(row["object_a_norad"]),
                    int(row["object_b_norad"]),
                    tuple(str(p) for p in archives),
                    int(rewind_days),
                    int(snapshot_step_hours),
                    int(forecast_step_minutes),
                    int(qae_eval_qubits),
                    int(qae_shots),
                )
            st.session_state["historical_validation_result"] = result
            st.session_state["historical_validation_key"] = current_key
            st.success(f"Replay complete: {len(result)} historical snapshots.")
        except Exception as exc:
            st.error(f"Historical replay failed: {exc}")
            return

    if result is None or result_key != current_key:
        st.warning("Run the replay above to populate the historical validation charts.")
        return

    result = result.copy()
    result["DAYS_BEFORE_EVENT"] = pd.to_numeric(result["DAYS_BEFORE_EVENT"], errors="coerce")

    high_alert = result[result["FORECAST_PROXIMITY_ALERT_LEVEL"].astype(str).str.upper() == "HIGH"]
    high_probability = result[result["ANALYTIC_RISK_LEVEL"].astype(str).str.upper() == "HIGH"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshots", len(result))
    c2.metric("First HIGH proximity", format_days(high_alert["DAYS_BEFORE_EVENT"].max() if not high_alert.empty else None))
    c3.metric("First HIGH probability", format_days(high_probability["DAYS_BEFORE_EVENT"].max() if not high_probability.empty else None))
    c4.metric("Actual collision", str(row["event_time_utc"]).replace("T", " ").replace("Z", " UTC"))

    st.subheader("Two warning channels")
    st.caption(
        "The historical predictor reports geometric proximity and modeled collision probability "
        "as separate warning channels. A HIGH proximity alert does not automatically mean HIGH "
        "collision probability. Both values are shown so the replay can be interpreted correctly."
    )

    channel_a, channel_b = st.columns(2)
    with channel_a:
        st.markdown("### 📏 Channel 1 — Proximity warning")
        st.metric(
            "Current proximity alert",
            format_warning_level(selected_row["FORECAST_PROXIMITY_ALERT_LEVEL"])
            if "selected_row" in locals()
            else "—",
        )
        st.caption(
            "Geometric screening channel based on the forecast closest-approach distance. "
            "It answers: **How close are the objects expected to pass?**"
        )
    with channel_b:
        st.markdown("### 🎯 Channel 2 — Collision probability")
        st.metric(
            "Current probability risk",
            format_warning_level(selected_row["ANALYTIC_RISK_LEVEL"])
            if "selected_row" in locals()
            else "—",
        )
        st.caption(
            "Probability channel from the analytic collision-probability model. "
            "It answers: **Given the modeled uncertainty, how likely is collision?**"
        )

    selected_day = st.slider(
        "Replay timeline — select historical prediction state",
        min_value=float(result["DAYS_BEFORE_EVENT"].min()),
        max_value=float(result["DAYS_BEFORE_EVENT"].max()),
        value=float(result["DAYS_BEFORE_EVENT"].max()),
        step=1.0,
    )
    closest = result.iloc[(result["DAYS_BEFORE_EVENT"] - selected_day).abs().argsort()[:1]]
    selected_row = closest.iloc[0]

    # Re-render the channel summary after the timeline selection so the displayed
    # warning levels always correspond to the selected historical state.
    channel_a, channel_b = st.columns(2)
    with channel_a:
        st.markdown("### 📏 Proximity warning")
        st.metric(
            "Alert level",
            format_warning_level(selected_row.get("FORECAST_PROXIMITY_ALERT_LEVEL")),
        )
        if "FORECAST_MISS_DISTANCE_KM" in selected_row.index:
            st.metric(
                "Forecast miss distance",
                f"{float(selected_row['FORECAST_MISS_DISTANCE_KM']):.6f} km",
            )
        st.caption("Geometric screening: forecast closest-approach distance.")

    with channel_b:
        st.markdown("### 🎯 Collision probability")
        st.metric(
            "Probability risk",
            format_warning_level(selected_row.get("ANALYTIC_RISK_LEVEL")),
        )
        st.metric(
            "Analytic P(collision)",
            format_probability(selected_row.get("ANALYTIC_PC")),
        )
        st.caption("Probabilistic model: collision likelihood under the configured uncertainty assumptions.")

    st.info(
        f"At the selected replay state (T-{float(selected_row['DAYS_BEFORE_EVENT']):.1f} d), "
        f"the proximity channel is **{format_warning_level(selected_row.get('FORECAST_PROXIMITY_ALERT_LEVEL'))}** "
        f"while the probability channel is **{format_warning_level(selected_row.get('ANALYTIC_RISK_LEVEL'))}** "
        f"with P(collision) = **{format_probability(selected_row.get('ANALYTIC_PC'))}**. "
        "These channels are intentionally not collapsed into one label."
    )

    st.plotly_chart(historical_replay_figure(result, selected_day), width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Replay state", f"T-{selected_row['DAYS_BEFORE_EVENT']:.1f} d")
    c2.metric("Forecast TCA", str(selected_row["FORECAST_TCA"]).replace("T", " ").replace("Z", " UTC"))
    c3.metric("Forecast miss", f"{float(selected_row['FORECAST_MISS_DISTANCE_KM']):.6f} km")
    c4.metric("Probability P(collision)", format_probability(selected_row.get("ANALYTIC_PC")))

    st.subheader("QAE vs Monte Carlo — historical validation")
    st.plotly_chart(qae_mc_figure(result), width="stretch")

    display_cols = [
        "SNAPSHOT_TIME",
        "DAYS_BEFORE_EVENT",
        "FORECAST_TCA",
        "FORECAST_LEAD_TIME_DAYS",
        "FORECAST_MISS_DISTANCE_KM",
        "FORECAST_RELATIVE_VELOCITY_KM_S",
        "FORECAST_PROXIMITY_ALERT_LEVEL",
        "ANALYTIC_PC",
        "ANALYTIC_RISK_LEVEL",
        "QAE_ESTIMATE",
        "MC_ESTIMATE",
        "QAE_EVAL_QUBITS_USED",
    ]
    display_cols = [c for c in display_cols if c in result.columns]
    st.subheader("Replay results")
    st.dataframe(result[display_cols], width="stretch")

    HISTORICAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = HISTORICAL_RESULTS_DIR / f"{row['event_id']}_validation.csv"
    if st.button("💾 Save replay result to results/historical", width="stretch"):
        result.to_csv(output_path, index=False)
        st.success(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")


def render_live_tab():
    st.header("📡 Live 30-Day Forecast")
    st.caption(
        "PRESENT: current orbital elements are the input. The system forecasts the next "
        "30 days and identifies conjunctions that become high-risk under the existing pipeline."
    )

    orbital_df = load_current_orbital_data()
    predictions = load_csv(str(PROJECT_ROOT / "results" / "predictions.csv"))
    conjunctions = load_csv(str(config.CONJUNCTIONS_FILE))

    c1, c2, c3 = st.columns(3)
    c1.metric("Current orbital objects", int(len(orbital_df)) if orbital_df is not None else 0)
    c2.metric("30-day prediction rows", int(len(predictions)) if predictions is not None else 0)
    c3.metric("Screened conjunctions", int(len(conjunctions)) if conjunctions is not None else 0)

    if orbital_df is None or orbital_df.empty:
        st.warning("Current orbital data is unavailable. Run the current data-loading pipeline first.")
    else:
        st.subheader("Current orbital data")
        st.dataframe(orbital_df, width="stretch", height=260)

        if st.button("🔄 Propagate current objects now", width="stretch"):
            try:
                with st.spinner("Propagating current orbital elements with SGP4…"):
                    current_positions, failed = sgp4_propagation.propagate_all_now(
                        orbital_df,
                        verbose=False,
                    )
                if current_positions is not None and not current_positions.empty:
                    st.session_state["live_positions"] = current_positions
                    st.session_state["live_failed"] = failed
                    st.success(f"Current propagation complete: {len(current_positions)} states.")
                else:
                    st.warning("No current states could be propagated.")
            except Exception as exc:
                st.error(f"Current propagation failed: {exc}")

    live_positions = st.session_state.get("live_positions")
    if live_positions is not None and not live_positions.empty:
        try:
            st.plotly_chart(
                viz.live_globe_figure(
                    live_positions,
                    timestamp_label=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                ),
                width="stretch",
            )
        except Exception as exc:
            st.info(f"Live globe unavailable: {exc}")

    if conjunctions is not None and not conjunctions.empty:
        st.subheader("Current conjunction screening")
        try:
            st.plotly_chart(viz.miss_distance_scatter(conjunctions), width="stretch")
        except Exception as exc:
            st.info(f"Conjunction chart unavailable: {exc}")
        st.dataframe(conjunctions, width="stretch")

    if predictions is not None and not predictions.empty:
        st.subheader("30-day risk forecast")
        risk_col = next(
            (c for c in ["COMPOSITE_RISK_LEVEL", "RISK_LEVEL", "ANALYTIC_RISK_LEVEL"] if c in predictions.columns),
            None,
        )
        if risk_col:
            counts = predictions[risk_col].astype(str).str.upper().value_counts()
            st.bar_chart(counts)
            high = predictions[predictions[risk_col].astype(str).str.upper().isin(["HIGH", "CRITICAL"])]
            if high.empty:
                st.info("No HIGH/CRITICAL forecast rows are present in the current result file.")
            else:
                st.warning(f"{len(high)} HIGH/CRITICAL forecast rows require attention.")
                st.dataframe(high, width="stretch")
        else:
            st.dataframe(predictions, width="stretch")

    st.markdown(
        "**Live rule:** current data drives the forecast. Historical collision outcomes are not "
        "fed into the live predictor."
    )


def main():
    st.set_page_config(
        page_title="Orbital Debris — Past vs Present",
        page_icon="🛰️",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp { background-color: #0b1220; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #f8fafc; }
        p, label { color: #cbd5e1; }
        [data-testid="stMetric"] { background-color: #1e293b; border-radius: 10px; padding: 15px; }
        [data-testid="stMetricLabel"] { color: #94a3b8; }
        [data-testid="stMetricValue"] { color: #e2e8f0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛰️ Orbital Debris Collision Risk")
    st.caption("PAST: historical replay and validation  |  PRESENT: live 30-day forecasting")

    past_tab, live_tab = st.tabs(["⏪ PAST — Historical Replay / Validation", "📡 PRESENT — Live 30-Day Forecast"])

    with past_tab:
        render_historical_tab()

    with live_tab:
        render_live_tab()


if __name__ == "__main__":
    main()
