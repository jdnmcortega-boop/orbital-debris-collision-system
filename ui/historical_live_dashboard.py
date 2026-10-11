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


# Defensive display-only normalization for mixed object columns. This module
# may also be launched directly, bypassing ui/dashboard.py's wrapper.
_original_st_dataframe = st.dataframe


def _display_scalar(value):
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, type(pd.NA))) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or value is pd.NA:
        return None
    return str(value)


def _safe_display_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        return frame
    display = frame.copy()
    for column in display.columns:
        if display[column].dtype == object:
            display[column] = display[column].map(_display_scalar)
    return display


def _safe_st_dataframe(data=None, *args, **kwargs):
    is_styler = hasattr(data, "data") and isinstance(getattr(data, "data", None), pd.DataFrame)
    if isinstance(data, pd.DataFrame):
        data = _safe_display_frame(data)
    elif is_styler:
        data = _safe_display_frame(data.data).style
    try:
        return _original_st_dataframe(data, *args, **kwargs)
    except Exception:
        # Last resort affects only the rendered copy, never the scientific data.
        if isinstance(data, pd.DataFrame):
            fallback = data.copy()
            for column in fallback.columns:
                fallback[column] = fallback[column].map(_display_scalar)
            return _original_st_dataframe(fallback, *args, **kwargs)
        if hasattr(data, "data") and isinstance(getattr(data, "data", None), pd.DataFrame):
            fallback = data.data.copy()
            for column in fallback.columns:
                fallback[column] = fallback[column].map(_display_scalar)
            return _original_st_dataframe(fallback, *args, **kwargs)
        raise


st.dataframe = _safe_st_dataframe

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
from modules import monte_carlo
from modules import conjunction_detection
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
    confirmation = str(row.get("confirmation", "confirmed")).strip().lower()
    outcome = "CONFIRMED_COLLISION" if confirmation == "confirmed" else (
        "CANDIDATE_UNCONFIRMED" if confirmation.startswith("candidate") else "UNKNOWN"
    )
    return HistoricalValidationEvent(
        event_id=str(row["event_id"]),
        event_time_utc=utc_datetime(str(row["event_time_utc"])),
        object_a_norad=int(row["object_a_norad"]),
        object_b_norad=int(row["object_b_norad"]),
        ground_truth_outcome=outcome,
    )


@st.cache_data(show_spinner=False)
def execute_historical_validation(
    event_id: str,
    event_time_text: str,
    norad_a: int,
    norad_b: int,
    ground_truth_outcome: str,
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
        ground_truth_outcome=ground_truth_outcome,
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


def conjunction_signature_figure(frame: pd.DataFrame):
    """Distinctive live conjunction bubble plot; display-only, never changes risk labels."""
    required = {"MISS_DISTANCE_KM", "RELATIVE_VELOCITY_KM_S", "ANALYTIC_PC"}
    if not required.issubset(frame.columns):
        return None

    plot = frame.copy()
    for col in required:
        plot[col] = pd.to_numeric(plot[col], errors="coerce")
    plot = plot.dropna(subset=list(required))
    plot = plot[(plot["MISS_DISTANCE_KM"] >= 0) & (plot["RELATIVE_VELOCITY_KM_S"] >= 0)]
    if plot.empty:
        return None

    # The floor is used only to display zero/underflow values on a logarithmic
    # visual scale; the source probability and official risk classification stay unchanged.
    pc_floor = 1e-300
    plot["_PC_LOG10"] = plot["ANALYTIC_PC"].clip(lower=pc_floor)
    plot["_PC_ORDER"] = -plot["_PC_LOG10"].apply(lambda value: __import__("math").log10(value))
    plot["_BUBBLE_SIZE"] = plot["_PC_ORDER"].clip(lower=1, upper=300).apply(lambda value: 7 + min(value, 300) / 18)

    # Color by probability order of magnitude rather than raw tiny Pc values,
    # which would otherwise collapse to nearly one color on a linear color scale.
    plot["_PC_COLOR"] = plot["_PC_ORDER"]
    custom = plot[["ANALYTIC_PC"]].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot["MISS_DISTANCE_KM"],
        y=plot["RELATIVE_VELOCITY_KM_S"],
        mode="markers",
        name="Conjunctions",
        customdata=custom,
        marker=dict(            size=plot["_BUBBLE_SIZE"],
            color=plot["_PC_COLOR"],
            colorscale="Turbo",
            colorbar=dict(title="−log10(Pc)"),
            showscale=True,
            opacity=0.82,
            line=dict(width=0.6, color="#e2e8f0"),
        ),
        hovertemplate=(
            "Miss distance: %{x:.6g} km<br>Relative velocity: %{y:.6g} km/s"
            "<br>Analytical Pc: %{customdata[0]:.3e}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Conjunction Signature Map",
        height=520,
        margin=dict(l=55, r=45, t=65, b=55),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
    )
    fig.update_xaxes(title="Miss distance at TCA (km)", zeroline=False, gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(title="Relative velocity at TCA (km/s)", zeroline=False, gridcolor="rgba(148,163,184,0.18)")
    return fig


def qae_analytical_parity_figure(result: pd.DataFrame):
    """Compare QAE estimates with analytical Pc on a log-log parity plot."""
    needed = {"ANALYTIC_PC", "QAE_ESTIMATE"}
    if not needed.issubset(result.columns):
        return None

    plot = result.copy()
    plot["ANALYTIC_PC"] = pd.to_numeric(plot["ANALYTIC_PC"], errors="coerce")
    plot["QAE_ESTIMATE"] = pd.to_numeric(plot["QAE_ESTIMATE"], errors="coerce")
    plot = plot.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["ANALYTIC_PC", "QAE_ESTIMATE"])
    # Log axes cannot show zero/negative values. Keep those rows in the table;
    # this figure includes only strictly positive estimates.
    plot = plot[(plot["ANALYTIC_PC"] > 0) & (plot["QAE_ESTIMATE"] > 0)]
    if plot.empty:
        return None

    min_value = float(min(plot["ANALYTIC_PC"].min(), plot["QAE_ESTIMATE"].min()))
    max_value = float(max(plot["ANALYTIC_PC"].max(), plot["QAE_ESTIMATE"].max()))
    lo = 10 ** __import__("math").floor(__import__("math").log10(min_value))
    hi = 10 ** __import__("math").ceil(__import__("math").log10(max_value))
    if lo == hi:
        lo /= 10
        hi *= 10

    hover_cols = [c for c in ["DAYS_BEFORE_EVENT", "FORECAST_MISS_DISTANCE_KM", "FORECAST_RELATIVE_VELOCITY_KM_S"] if c in plot.columns]
    custom = plot[hover_cols].astype(str).values if hover_cols else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect agreement (QAE = analytical)",
        line=dict(color="#94a3b8", dash="dash", width=2),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=plot["ANALYTIC_PC"], y=plot["QAE_ESTIMATE"], mode="markers",
        name="Historical snapshots", customdata=custom,
        marker=dict(
            size=10,
            color=pd.to_numeric(plot["DAYS_BEFORE_EVENT"], errors="coerce") if "DAYS_BEFORE_EVENT" in plot.columns else "#38bdf8",
            colorscale="Viridis",
            showscale="DAYS_BEFORE_EVENT" in plot.columns,
            colorbar=dict(title="Days before event") if "DAYS_BEFORE_EVENT" in plot.columns else None,
            line=dict(width=0.8, color="#e2e8f0"),
            opacity=0.9,
        ),
        hovertemplate="Analytical Pc: %{x:.3e}<br>QAE estimate: %{y:.3e}<extra></extra>",
    ))
    fig.update_xaxes(title="Analytical collision probability Pc", type="log", range=[__import__("math").log10(lo), __import__("math").log10(hi)])
    fig.update_yaxes(title="QAE estimated probability", type="log", range=[__import__("math").log10(lo), __import__("math").log10(hi)])
    fig.update_layout(
        title="QAE vs Analytical Pc — Agreement Map",
        height=540,
        margin=dict(l=65, r=45, t=65, b=60),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
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

    ground_truth_label = str(row.get("confirmation", "confirmed")).strip().upper()
    if ground_truth_label == "CONFIRMED":
        st.success("Historical ground truth: CONFIRMED COLLISION. This is the recorded outcome, not a prediction made by ORION-X.")
    elif ground_truth_label.startswith("CANDIDATE"):
        st.warning("Historical ground truth: UNCONFIRMED CANDIDATE. Do not count this event as a proven collision.")
    else:
        st.info("Historical outcome is not confirmed in the event catalog.")
    st.info(
        "Ground truth is kept separate from prediction. The collision timestamp is "
        "the evaluation boundary; future TLEs are never used for an earlier replay snapshot. "
        "A LOW modeled probability for a confirmed collision is recorded as a possible missed warning, not rewritten as HIGH."
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
        str(row.get("confirmation", "confirmed")),
    )

    if run:
        try:
            with st.spinner("Replaying the historical encounter and running QAE/MC validation…"):
                result = execute_historical_validation(
                    str(row["event_id"]),
                    str(row["event_time_utc"]),
                    int(row["object_a_norad"]),
                    int(row["object_b_norad"]),
                    ("CONFIRMED_COLLISION" if str(row.get("confirmation", "confirmed")).strip().lower() == "confirmed" else "CANDIDATE_UNCONFIRMED" if str(row.get("confirmation", "")).strip().lower().startswith("candidate") else "UNKNOWN"),
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

    pre_event_result = result[result.get("PREDICTION_TIME_IS_PRE_EVENT", result["DAYS_BEFORE_EVENT"] > 0).astype(bool)]
    high_alert = pre_event_result[pre_event_result["FORECAST_PROXIMITY_ALERT_LEVEL"].astype(str).str.upper() == "HIGH"]
    high_probability = pre_event_result[pre_event_result["ANALYTIC_RISK_LEVEL"].astype(str).str.upper() == "HIGH"]
    medium_high_probability = pre_event_result[pre_event_result["ANALYTIC_RISK_LEVEL"].astype(str).str.upper().isin(["MEDIUM", "HIGH"])]
    medium_high_proximity = pre_event_result[pre_event_result["FORECAST_PROXIMITY_ALERT_LEVEL"].astype(str).str.upper().isin(["MEDIUM", "HIGH"])]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshots", len(result))
    c2.metric("First HIGH proximity", format_days(high_alert["DAYS_BEFORE_EVENT"].max() if not high_alert.empty else None))
    c3.metric("First HIGH probability", format_days(high_probability["DAYS_BEFORE_EVENT"].max() if not high_probability.empty else None))
    c4.metric("Actual outcome", "CONFIRMED COLLISION" if str(row.get("confirmation", "confirmed")).lower() == "confirmed" else "UNCONFIRMED")

    diag_cols = st.columns(3)
    diag_cols[0].metric("Pre-event Pc warnings", int(len(medium_high_probability)))
    diag_cols[1].metric("Pre-event proximity warnings", int(len(medium_high_proximity)))
    diag_cols[2].metric("Pre-event snapshots", int(len(pre_event_result)))
    if "HIGH_THRESHOLD_REACHABLE_UNDER_ASSUMED_SIGMA" in result.columns and not bool(result["HIGH_THRESHOLD_REACHABLE_UNDER_ASSUMED_SIGMA"].all()):
        max_pc = pd.to_numeric(result["PC_MAX_AT_ZERO_MISS_UNDER_ASSUMED_SIGMA"], errors="coerce").max()
        threshold = pd.to_numeric(result["HIGH_PC_THRESHOLD"], errors="coerce").max()
        st.warning(
            f"The configured uncertainty makes the HIGH Pc threshold unreachable even at zero miss distance: "
            f"maximum modeled Pc ≈ {max_pc:.3e}, HIGH threshold = {threshold:.3e}. "
            "This is a limitation of the assumptions, not evidence the real collision was low-risk. "
            "Do not force Pc higher; validate historical covariance or report sensitivity scenarios separately."
        )
    if str(row.get("confirmation", "confirmed")).strip().lower() == "confirmed" and len(medium_high_probability) == 0:
        st.error("Validation result: MISSED PC WARNING in the pre-event replay window. The event is confirmed, but no pre-event snapshot reached MEDIUM/HIGH modeled Pc.")
    elif str(row.get("confirmation", "confirmed")).strip().lower() == "confirmed":
        st.info("Validation result: at least one pre-event snapshot reached MEDIUM/HIGH modeled Pc. Check the warning lead time and false-alarm behavior against non-collision events.")

    st.subheader("Two warning channels")
    st.caption(
        "The historical predictor reports geometric proximity and modeled collision probability "
        "as separate warning channels. A HIGH proximity alert does not automatically mean HIGH "
        "collision probability. Both values are shown so the replay can be interpreted correctly."
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
        "These channels are intentionally not collapsed into one label. The historical outcome remains separate from both predictions."
    )

    st.plotly_chart(historical_replay_figure(result, selected_day), width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Replay state", f"T-{selected_row['DAYS_BEFORE_EVENT']:.1f} d")
    c2.metric("Forecast TCA", str(selected_row["FORECAST_TCA"]).replace("T", " ").replace("Z", " UTC"))
    c3.metric("Forecast miss", f"{float(selected_row['FORECAST_MISS_DISTANCE_KM']):.6f} km")
    c4.metric("Probability P(collision)", format_probability(selected_row.get("ANALYTIC_PC")))

    st.subheader("QAE vs Analytical Pc — agreement analysis")
    st.caption(
        "Each point is one historical replay snapshot. The dashed diagonal is perfect agreement; "
        "points above it mean QAE estimated a higher probability than the analytical model, and "
        "points below it mean QAE estimated a lower probability. Zero-valued estimates are omitted "
        "from this log-scale chart only and remain in the results table."
    )
    qae_analytical_fig = qae_analytical_parity_figure(result)
    if qae_analytical_fig is not None:
        st.plotly_chart(qae_analytical_fig, width="stretch")
    else:
        st.info("No positive QAE and analytical Pc values are available for a log-scale agreement plot.")

    st.subheader("QAE vs Monte Carlo — time-series validation")
    st.plotly_chart(qae_mc_figure(result), width="stretch")

    display_cols = [
        "SNAPSHOT_TIME",
        "DAYS_BEFORE_EVENT",
        "FORECAST_TCA",
        "FORECAST_LEAD_TIME_DAYS",
        "FORECAST_MISS_DISTANCE_KM",
        "FORECAST_RELATIVE_VELOCITY_KM_S",
        "FORECAST_PROXIMITY_ALERT_LEVEL",
        "GROUND_TRUTH_OUTCOME",
        "PREDICTION_TIME_IS_PRE_EVENT",
        "ANALYTIC_PC",
        "ANALYTIC_RISK_LEVEL",
        "PC_MAX_AT_ZERO_MISS_UNDER_ASSUMED_SIGMA",
        "HIGH_THRESHOLD_REACHABLE_UNDER_ASSUMED_SIGMA",
        "MODEL_DIAGNOSTIC",
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

    # Current live workflow: analytical collision probability is primary.
    # Monte Carlo is not required for the present-day forecast.
    st.divider()
    st.subheader("🛰️ CDM covariance import and validation")
    st.caption(
        "Upload either the public Space-Track cdm_public CSV to compare its reported Pc "
        "against ORION-X analytical results, or a flattened CCSDS CDM CSV to validate "
        "the two objects' 3×3 RTN position-covariance blocks. This step does not replace "
        "the live Pc model with CDM covariance."
    )
    cdm_upload = st.file_uploader(
        "Upload CDM CSV",
        type=["csv"],
        key="orion_cdm_covariance_upload",
        help="Accepts public Space-Track cdm_public CSV (SAT_1_ID, SAT_2_ID, TCA, PC, MIN_RNG) for reference comparison, or a flattened CCSDS CDM CSV with object IDs and covariance fields for covariance validation.",
    )
    if cdm_upload is not None:
        try:
            from modules.cdm_covariance import (
                load_cdm_csv, validate_cdm_rows, match_cdms_to_conjunctions,
                has_position_covariance_fields,
            )

            cdm_frame = load_cdm_csv(cdm_upload)
            cdm_validation = validate_cdm_rows(cdm_frame)
            valid_both = int((cdm_validation["COVARIANCE_VALIDATION_STATUS"] == "BOTH_POSITION_BLOCKS_VALID").sum())
            reported_pc_count = int(pd.to_numeric(cdm_validation.get("CDM_REPORTED_PC"), errors="coerce").notna().sum()) if "CDM_REPORTED_PC" in cdm_validation.columns else 0
            covariance_fields_present = has_position_covariance_fields(cdm_frame.columns)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CDM records", int(len(cdm_frame)))
            c2.metric("Records with reported Pc", reported_pc_count)
            c3.metric("Both position blocks valid", valid_both)
            c4.metric("Covariance fields in file", "Yes" if covariance_fields_present else "No")
            if reported_pc_count:
                st.info(
                    "This public CDM CSV contains reported collision probabilities and can be used as a reference dataset. "
                    "It does not necessarily include covariance matrices, so the reported Pc is not a substitute for covariance-based recalculation."
                )
            st.subheader("CDM field validation")
            validation_cols = [
                "CDM_SOURCE_ROW", "CDM_TCA", "CDM_OBJECT1_ID", "CDM_OBJECT2_ID",
                "CDM_REPORTED_PC", "CDM_MIN_RANGE", "OBJECT1_COVARIANCE_STATUS",
                "OBJECT2_COVARIANCE_STATUS", "COVARIANCE_VALIDATION_STATUS", "NOTE",
            ]
            validation_cols = [column for column in validation_cols if column in cdm_validation.columns]
            st.dataframe(cdm_validation[validation_cols], width="stretch", hide_index=True)
            # Compare against whichever available ORION-X source has the greatest
            # overlap with the uploaded CDM object pairs. An old analytical result
            # file must not silently take precedence over a better-matching current
            # conjunction file.
            analytic_path_for_cdm = PROJECT_ROOT / "results" / "analytic_pc_results.csv"
            analytic_for_cdm = load_csv(str(analytic_path_for_cdm))
            required_match_cols = {"NORAD_A", "NORAD_B", "TCA"}

            def _diagnostic_id(value):
                if value is None or pd.isna(value):
                    return ""
                text_value = str(value).strip()
                try:
                    numeric = float(text_value)
                    if numeric.is_integer():
                        return str(int(numeric))
                except (TypeError, ValueError):
                    pass
                return text_value.upper()

            cdm_pairs = set()
            for _, cdm_row in cdm_frame.iterrows():
                id_a = _diagnostic_id(cdm_row.get("_CDM_OBJECT1"))
                id_b = _diagnostic_id(cdm_row.get("_CDM_OBJECT2"))
                if id_a and id_b:
                    cdm_pairs.add(tuple(sorted((id_a, id_b))))

            source_candidates = []
            if analytic_for_cdm is not None and not analytic_for_cdm.empty and required_match_cols.issubset(analytic_for_cdm.columns):
                source_candidates.append(("analytical Pc results", analytic_for_cdm))
            if conjunctions is not None and not conjunctions.empty and required_match_cols.issubset(conjunctions.columns):
                source_candidates.append(("screened conjunctions", conjunctions))

            def _source_pair_overlap(source_frame):
                pairs = set()
                for _, source_row in source_frame.iterrows():
                    id_a = _diagnostic_id(source_row.get("NORAD_A"))
                    id_b = _diagnostic_id(source_row.get("NORAD_B"))
                    if id_a and id_b:
                        pairs.add(tuple(sorted((id_a, id_b))))
                return len(cdm_pairs.intersection(pairs))

            if source_candidates:
                match_source_label, match_source = max(
                    source_candidates,
                    key=lambda item: _source_pair_overlap(item[1]),
                )
            else:
                match_source = None
                match_source_label = "no compatible ORION-X result source"
            if match_source is not None and not match_source.empty:
                # Diagnose why zero strict matches occur before comparing Pc.
                # Pair overlap ignores object order; TCA tolerance remains unchanged.
                cdm_tcas = pd.to_datetime(cdm_frame.get("_CDM_TCA"), utc=True, errors="coerce").dropna()
                orion_tcas = pd.to_datetime(match_source.get("TCA"), utc=True, errors="coerce").dropna()
                cdm_date_range = (
                    f"{cdm_tcas.min():%Y-%m-%d} to {cdm_tcas.max():%Y-%m-%d}"
                    if not cdm_tcas.empty else "Unavailable"
                )
                orion_date_range = (
                    f"{orion_tcas.min():%Y-%m-%d} to {orion_tcas.max():%Y-%m-%d}"
                    if not orion_tcas.empty else "Unavailable"
                )

                def _diagnostic_id(value):
                    if value is None or pd.isna(value):
                        return ""
                    text_value = str(value).strip()
                    try:
                        numeric = float(text_value)
                        if numeric.is_integer():
                            return str(int(numeric))
                    except (TypeError, ValueError):
                        pass
                    return text_value.upper()

                cdm_pairs = set()
                for _, cdm_row in cdm_frame.iterrows():
                    id_a = _diagnostic_id(cdm_row.get("_CDM_OBJECT1"))
                    id_b = _diagnostic_id(cdm_row.get("_CDM_OBJECT2"))
                    if id_a and id_b:
                        cdm_pairs.add(tuple(sorted((id_a, id_b))))
                orion_pairs = set()
                for _, orion_row in match_source.iterrows():
                    id_a = _diagnostic_id(orion_row.get("NORAD_A"))
                    id_b = _diagnostic_id(orion_row.get("NORAD_B"))
                    if id_a and id_b:
                        orion_pairs.add(tuple(sorted((id_a, id_b))))
                shared_pairs = cdm_pairs.intersection(orion_pairs)
                orion_events_with_pair = 0
                for _, orion_row in match_source.iterrows():
                    id_a = _diagnostic_id(orion_row.get("NORAD_A"))
                    id_b = _diagnostic_id(orion_row.get("NORAD_B"))
                    if id_a and id_b and tuple(sorted((id_a, id_b))) in cdm_pairs:
                        orion_events_with_pair += 1

                st.subheader("Why are there zero CDM event matches?")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("CDM TCA date range", cdm_date_range)
                d2.metric("ORION-X TCA date range", orion_date_range)
                d3.metric("Shared object pairs", len(shared_pairs))
                d4.metric("ORION-X events with a CDM pair", orion_events_with_pair)
                if not shared_pairs:
                    st.warning(
                        "No unordered object pairs overlap between this CDM file and the selected ORION-X result source. "
                        "A 10-minute TCA match is therefore impossible until the object catalog/results and CDM export overlap."
                    )
                elif orion_events_with_pair and cdm_tcas.empty:
                    st.warning("Object pairs overlap, but the CDM TCA values could not be parsed as dates.")
                else:
                    st.info(
                        "Object-pair overlap is checked independently of TCA. If pairs overlap but strict matches remain zero, "
                        "inspect the date ranges and the per-event TCA deltas below. The 10-minute tolerance is intentionally unchanged."
                    )

                matches = match_cdms_to_conjunctions(cdm_frame, match_source)
                if not matches.empty:
                    matched = matches["CDM_MATCH_STATUS"].eq("MATCHED_WITHIN_TOLERANCE")
                    pc_comparable = pd.to_numeric(matches.get("PC_ABSOLUTE_DIFFERENCE"), errors="coerce").notna()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Events matched within 10 minutes", int(matched.sum()))
                    m2.metric("Matched records with Pc comparison", int((matched & pc_comparable).sum()))
                    m3.metric("CDM records with reported Pc", reported_pc_count)
                    st.caption(
                        f"Matching source: {match_source_label}. Pc differences are shown only when both the CDM reference Pc "
                        "and ORION-X analytical Pc are present. They are comparisons, not recalibrations."
                    )
                    match_cols = [
                        "NORAD_A", "NORAD_B", "EVENT_TCA", "CDM_TCA",
                        "CDM_TCA_DELTA_MINUTES", "CDM_MATCH_STATUS",
                        "CDM_REPORTED_PC", "ORION_ANALYTIC_PC", "PC_ABSOLUTE_DIFFERENCE",
                        "CDM_MIN_RANGE", "ORION_MISS_DISTANCE_KM",
                        "COVARIANCE_VALIDATION_STATUS",
                    ]
                    match_cols = [column for column in match_cols if column in matches.columns]
                    st.dataframe(matches[match_cols], width="stretch", hide_index=True)
                    st.download_button(
                        "Download CDM matching and Pc comparison report",
                        data=matches.to_csv(index=False).encode("utf-8"),
                        file_name="orion_x_cdm_match_report.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Load or rebuild the conjunction file to match CDM rows against ORION-X events.")
            if valid_both:
                st.warning(
                    "The covariance blocks are structurally validated, but are not yet used for Pc. "
                    "A full encounter-plane covariance transformation and independent verification are required before substitution."
                )
            elif not covariance_fields_present and reported_pc_count:
                st.info(
                    "Detected a reported-Pc reference export: this CSV contains conjunction metadata and Pc, "
                    "but no flattened RTN position-covariance fields. Those records are valid for Pc comparison; "
                    "they are not malformed covariance records. ORION-X continues to use its labeled "
                    "TLE-age-scaled uncertainty model for its own Pc estimate."
                )
            elif not covariance_fields_present:
                st.warning(
                    "No expected flattened RTN position-covariance fields were found. "
                    "The file may still be useful as a metadata/reference export if it contains reported Pc."
                )
            else:
                st.warning(
                    "Covariance fields were detected, but no row contains two valid position-covariance blocks. "
                    "Review the per-object validation statuses and verify the source units and format."
                )
        except Exception as exc:
            st.error(f"Could not parse or validate this CDM CSV: {exc}")

    st.divider()
    st.subheader("🎯 Live conjunction + analytical collision probability")
    st.caption(
        "Rebuilds the 30-day SGP4 propagation grid, screens conjunctions, "
        "calculates analytical collision probability (Pc) using modeled uncertainty, "
        "and regenerates the risk forecast. QAE is an estimator comparison, not the "
        "source of the orbital collision probability."
    )

    propagated_path = Path(config.PROPAGATED_GRID_FILE)
    conjunction_path = Path(config.CONJUNCTIONS_FILE)
    analytic_path = PROJECT_ROOT / "results" / "analytic_pc_results.csv"

    s1, s2, s3 = st.columns(3)
    s1.metric("Propagation grid", "Ready" if propagated_path.exists() else "Missing")
    s2.metric("Conjunction file", "Ready" if conjunction_path.exists() else "Missing")
    s3.metric("Analytical Pc result", "Ready" if analytic_path.exists() else "Missing")

    run_live_pipeline = st.button(
        "▶ Rebuild live inputs and calculate analytical Pc",
        key="live_run_analytical_pc",
        type="primary",
        width="stretch",
        disabled=orbital_df is None or orbital_df.empty,
    )

    if orbital_df is None or orbital_df.empty:
        st.warning(
            "Current orbital data is unavailable, so the live calculation cannot run. "
            "Check data/orbital_data.csv and the data loader."
        )

    if run_live_pipeline:
        try:
            from modules import live_refresh
            import importlib
            importlib.invalidate_caches()
            importlib.reload(live_refresh)
            with st.spinner(
                "Rebuilding current-data propagation, conjunctions, analytical Pc, and risk forecast…"
            ):
                summary = live_refresh.refresh_all_live_results()
            st.session_state["live_analytical_pc_summary"] = summary
            for stale_key in (
                "live_hybrid_mc_result",
                "live_hybrid_mc_timestamp",
                "live_positions",
                "live_failed",
                "live_failed_objects",
                "tracker_positions",
                "tracker_failed",
            ):
                st.session_state.pop(stale_key, None)
            st.cache_data.clear()
            st.success(                "Analytical live pipeline completed: "
                f"{summary.get('conjunctions', 0):,} conjunctions, "
                f"{summary.get('analytical_pc_rows', 0):,} analytical Pc rows."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Live analytical-probability pipeline failed: {exc}")

    analytic_result = load_csv(str(analytic_path))
    if analytic_result is not None and not analytic_result.empty:
        st.subheader("Analytical collision-probability results")
        st.caption(
            "This conjunction signature plot maps miss distance against relative velocity. "
            "Bubble size/color show the analytical Pc scale for visual triage only; they do not "
            "modify the official probability values or risk labels."
        )
        signature_fig = conjunction_signature_figure(analytic_result)
        if signature_fig is not None:
            st.plotly_chart(signature_fig, width="stretch")
        else:
            st.info("The signature plot needs numeric miss distance, relative velocity, and analytical Pc columns.")
        display_cols = [
            "OBJECT_A", "OBJECT_B", "TCA", "MISS_DISTANCE_KM",
            "RELATIVE_VELOCITY_KM_S", "SIGMA_A_KM", "SIGMA_B_KM",
            "ANALYTIC_PC", "PC_SENSITIVITY_HALF_SIGMA",
            "PC_SENSITIVITY_DOUBLE_SIGMA", "HIGH_PC_THRESHOLD",
            "HIGH_THRESHOLD_REACHABLE_UNDER_SIGMA",
            "PROBABILITY_METHOD", "PROBABILITY_STATUS",
        ]
        display_cols = [column for column in display_cols if column in analytic_result.columns]
        st.dataframe(analytic_result[display_cols], width="stretch", hide_index=True)
    elif analytic_path.exists():
        st.info("The analytical Pc result file exists but contains no conjunction rows.")
    else:
        st.info("No analytical Pc results saved yet. Run the analytical live pipeline above.")

    if predictions is not None and not predictions.empty:
        st.subheader("30-day risk forecast")
        st.caption(
            "Collision probability and ORION-X response priority are separate outputs. "
            "A HIGH-priority event is not automatically a HIGH collision probability."
        )

        pc_col = next(
            (c for c in ["COLLISION_RISK_LEVEL", "RISK_LEVEL", "ANALYTIC_RISK_LEVEL"] if c in predictions.columns),
            None,
        )
        priority_col = next(
            (c for c in ["ORION_X_PRIORITY_LEVEL", "COMPOSITE_RISK_LEVEL"] if c in predictions.columns),
            None,
        )

        m1, m2 = st.columns(2)
        if pc_col:
            pc_levels = predictions[pc_col].astype(str).str.upper()
            high_pc = predictions[pc_levels.isin(["HIGH", "CRITICAL"])]
            m1.metric("HIGH collision-probability events", int(len(high_pc)))
            st.caption("Collision-probability classifications use the configured Pc threshold.")
            if not high_pc.empty:
                st.dataframe(high_pc, width="stretch", hide_index=True)
        else:
            high_pc = pd.DataFrame()
            m1.metric("HIGH collision-probability events", "Unavailable")

        if priority_col:
            priority_levels = predictions[priority_col].astype(str).str.upper()
            high_priority = predictions[priority_levels.isin(["HIGH", "CRITICAL"])]
            m2.metric("HIGH ORION-X response-priority events", int(len(high_priority)))
            if not high_priority.empty:
                st.dataframe(high_priority, width="stretch", hide_index=True)
            else:
                st.info("No HIGH response-priority rows are present in the current result file.")
        else:
            high_priority = pd.DataFrame()
            m2.metric("HIGH ORION-X response-priority events", "Unavailable")

        if (
            "HIGH_THRESHOLD_REACHABLE_UNDER_SIGMA" in predictions.columns
        ):
            reachable = predictions["HIGH_THRESHOLD_REACHABLE_UNDER_SIGMA"].astype(str).str.lower()
            unreachable = int(reachable.isin(["false", "0"]).sum())
            if unreachable:
                from modules import uncertainty_model as uncertainty_settings
                st.warning(
                    f"{unreachable} event(s) have a modeled maximum Pc below the configured HIGH threshold, "
                    "even in the mathematical zero-miss scenario under the current assumed uncertainty. "
                    "This is a model diagnostic—not proof that the events are safe. The current uncertainty "
                    "values are assumptions, not object-specific covariance or validated empirical estimates. "
                    "Sensitivity columns are illustrative only and do not change the official Pc label."
                )
                with st.expander("Uncertainty-model assumptions and calibration"):
                    st.write(
                        f"Base 1-sigma assumption: {uncertainty_settings.SIGMA_BASE_KM:.3f} km per object"
                    )
                    st.write(
                        f"Assumed growth: {uncertainty_settings.GROWTH_RATE_KM_PER_DAY:.3f} km/day"
                    )
                    st.write(f"Model: {uncertainty_settings.UNCERTAINTY_MODEL_NAME}")
                    st.write(f"Validated from historical residuals: {'Yes' if uncertainty_settings.UNCERTAINTY_MODEL_VALIDATED else 'No'}")
                    st.caption(
                        "To configure a sensitivity scenario, set ORION_SIGMA_BASE_KM and "
                        "ORION_SIGMA_GROWTH_KM_PER_DAY in the deployment environment, then rebuild "
                        "the live pipeline. Changing these values is not calibration: calibrate them "
                        "against archived, time-separated orbital solutions or use validated covariance."
                    )

        st.subheader("All forecast events")
        all_cols = [
            "OBJECT_A", "NORAD_A", "OBJECT_B", "NORAD_B", "TCA",
            "MISS_DISTANCE_KM", "RELATIVE_VELOCITY_KM_S", "ANALYTIC_PC",
            "COLLISION_RISK_LEVEL", "ORION_X_PRIORITY_SCORE",
            "ORION_X_PRIORITY_LEVEL", "PC_SENSITIVITY_HALF_SIGMA",
            "PC_SENSITIVITY_DOUBLE_SIGMA", "PROBABILITY_STATUS",
        ]
        all_cols = [column for column in all_cols if column in predictions.columns]
        st.dataframe(predictions[all_cols], width="stretch", hide_index=True)

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