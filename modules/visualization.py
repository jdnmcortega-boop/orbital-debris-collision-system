"""
Reusable Plotly figure-generating functions for the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
from modules import orbital_mechanics as om


# ============================================================
# DARK THEME
# ============================================================

DARK_LAYOUT = dict(
    paper_bgcolor="#0b1220",
    plot_bgcolor="#0b1220",
    font=dict(color="#e2e8f0"),
    margin=dict(l=50, r=30, t=60, b=50),
)


# ============================================================
# ALTITUDE HISTORY
# ============================================================

def altitude_history_figure(
    propagated_df,
    norad_id,
    object_name,
):
    obj = propagated_df[
        propagated_df["NORAD_CAT_ID"] == norad_id
    ].copy()

    obj["TIME"] = pd.to_datetime(obj["TIME"])
    obj = obj.sort_values("TIME")

    altitude = (
        np.sqrt(
            obj["X_KM"] ** 2
            + obj["Y_KM"] ** 2
            + obj["Z_KM"] ** 2
        )
        - om.EARTH_RADIUS_KM
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=obj["TIME"],
            y=altitude,
            mode="lines",
            line=dict(
                color="#f59e0b",
                width=2,
            ),
            name="Altitude",
        )
    )

    fig.update_layout(
        title=f"Altitude History — {object_name}",
        xaxis_title="Time (UTC)",
        yaxis_title="Altitude (km)",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# GROUND TRACK
# ============================================================

def ground_track_figure(
    propagated_df,
    norad_id,
    object_name,
    max_points=300,
):
    obj = propagated_df[
        propagated_df["NORAD_CAT_ID"] == norad_id
    ].copy()

    obj["TIME"] = pd.to_datetime(obj["TIME"])
    obj = obj.sort_values("TIME")

    if len(obj) > max_points:
        obj = obj.iloc[
            ::max(1, len(obj) // max_points)
        ]

    lats = []
    lons = []

    for _, row in obj.iterrows():

        lat, lon, _ = om.eci_to_geodetic(
            row["X_KM"],
            row["Y_KM"],
            row["Z_KM"],
            row["TIME"],
        )

        lats.append(lat)
        lons.append(lon)

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="lines",
            line=dict(
                width=1,
                color="#f59e0b",
            ),
            name="Ground track",
        )
    )

    if lats and lons:

        fig.add_trace(
            go.Scattergeo(
                lat=[lats[0]],
                lon=[lons[0]],
                mode="markers",
                marker=dict(
                    size=9,
                    color="#22d3ee",
                ),
                name=object_name,
            )
        )

    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="#1e293b",
        showocean=True,
        oceancolor="#0b1220",
        showcountries=True,
        countrycolor="#334155",
        bgcolor="#0b1220",
    )

    fig.update_layout(
        title=f"Ground Track — {object_name}",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# RISK LEVEL BAR
# ============================================================

def risk_level_bar(
    predictions_df,
    risk_column="RISK_LEVEL",
):
    if risk_column not in predictions_df.columns:

        if "RISK_LEVEL" in predictions_df.columns:
            risk_column = "RISK_LEVEL"

        elif "COMPOSITE_RISK_LEVEL" in predictions_df.columns:
            risk_column = "COMPOSITE_RISK_LEVEL"

        else:
            raise ValueError(
                "No recognized risk-level column found."
            )

    counts = predictions_df[risk_column].value_counts()

    order = [
        level
        for level in ["LOW", "MEDIUM", "HIGH"]
        if level in counts.index
    ]

    colors = {
        "LOW": "#22c55e",
        "MEDIUM": "#f59e0b",
        "HIGH": "#ef4444",
    }

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=order,
            y=[
                counts[level]
                for level in order
            ],
            marker_color=[
                colors[level]
                for level in order
            ],
        )
    )

    fig.update_layout(
        title="Conjunctions by Risk Level",
        xaxis_title="Risk Level",
        yaxis_title="Number of Conjunctions",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# CONJUNCTION COUNT
# ============================================================

def conjunction_count_figure(conjunctions_df):
    """
    Displays the total number of detected conjunction pairs.
    """

    count = len(conjunctions_df)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Detected Conjunctions"],
            y=[count],
            text=[count],
            textposition="auto",
            marker_color="#22d3ee",
        )
    )

    fig.update_layout(
        title="Number of Detected Conjunction Pairs",
        xaxis_title="",
        yaxis_title="Number of Conjunction Pairs",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# MISS DISTANCE DISTRIBUTION
# ============================================================

def miss_distance_distribution(conjunctions_df):
    """
    Distribution of miss distances at Time of Closest Approach (TCA).
    """

    df = conjunctions_df.copy()

    if "MISS_DISTANCE_KM" not in df.columns:
        raise ValueError(
            "MISS_DISTANCE_KM column not found."
        )

    df["MISS_DISTANCE_KM"] = pd.to_numeric(
        df["MISS_DISTANCE_KM"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["MISS_DISTANCE_KM"]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["MISS_DISTANCE_KM"],
            nbinsx=20,
            marker_color="#f59e0b",
            name="Conjunctions",
        )
    )

    fig.update_layout(
        title="Distribution of Miss Distance at TCA",
        xaxis_title="Miss Distance at TCA (km)",
        yaxis_title="Number of Conjunctions",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# CLOSING SPEED DISTRIBUTION
# ============================================================

def closing_speed_distribution(conjunctions_df):
    """
    Distribution of relative/closing velocity at TCA.
    """

    df = conjunctions_df.copy()

    if "RELATIVE_VELOCITY_KM_S" not in df.columns:
        raise ValueError(
            "RELATIVE_VELOCITY_KM_S column not found."
        )

    df["RELATIVE_VELOCITY_KM_S"] = pd.to_numeric(
        df["RELATIVE_VELOCITY_KM_S"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["RELATIVE_VELOCITY_KM_S"]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["RELATIVE_VELOCITY_KM_S"],
            nbinsx=20,
            marker_color="#22d3ee",
            name="Conjunctions",
        )
    )

    fig.update_layout(
        title="Distribution of Closing Speed at TCA",
        xaxis_title="Closing Speed at TCA (km/s)",
        yaxis_title="Number of Conjunctions",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# MISS DISTANCE VS CLOSING SPEED
# ============================================================

def miss_distance_scatter(conjunctions_df):
    """
    Scatter plot of miss distance against closing speed
    at Time of Closest Approach (TCA).

    One point represents one detected conjunction pair.

    Required columns:
        MISS_DISTANCE_KM
        RELATIVE_VELOCITY_KM_S
    """

    df = conjunctions_df.copy()

    required_columns = [
        "MISS_DISTANCE_KM",
        "RELATIVE_VELOCITY_KM_S",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["MISS_DISTANCE_KM"] = pd.to_numeric(
        df["MISS_DISTANCE_KM"],
        errors="coerce",
    )

    df["RELATIVE_VELOCITY_KM_S"] = pd.to_numeric(
        df["RELATIVE_VELOCITY_KM_S"],
        errors="coerce",
    )

    df = df.dropna(
        subset=required_columns
    )

    if (
        "OBJECT_A" in df.columns
        and "OBJECT_B" in df.columns
    ):

        hover_text = (
            df["OBJECT_A"].astype(str)
            + " vs "
            + df["OBJECT_B"].astype(str)
        )

    else:

        hover_text = [
            f"Conjunction {i + 1}"
            for i in range(len(df))
        ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["MISS_DISTANCE_KM"],
            y=df["RELATIVE_VELOCITY_KM_S"],
            mode="markers",
            marker=dict(
                size=9,
                color="#22d3ee",
                opacity=0.8,
                line=dict(
                    width=1,
                    color="#e2e8f0",
                ),
            ),
            text=hover_text,
            hovertemplate=(
                "<b>%{text}</b>"
                "<br>Miss distance at TCA: "
                "%{x:.3f} km"
                "<br>Closing speed at TCA: "
                "%{y:.3f} km/s"
                "<extra></extra>"
            ),
            name="Conjunction pairs",
        )
    )

    fig.update_layout(
        title=(
            "Miss Distance vs Closing Speed "
            "at TCA"
        ),
        xaxis_title=(
            "Miss Distance at TCA (km)"
        ),
        yaxis_title=(
            "Closing Speed at TCA (km/s)"
        ),
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# COLLISION PROBABILITY VS MISS DISTANCE
# ============================================================

def collision_probability_vs_miss_distance(
    monte_carlo_df,
):
    """
    Collision probability versus miss distance.

    Uses monte_carlo_results.csv because this dataset
    contains COLLISION_PROBABILITY_MC.
    """

    df = monte_carlo_df.copy()

    required_columns = [
        "MISS_DISTANCE_KM",
        "COLLISION_PROBABILITY_MC",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["MISS_DISTANCE_KM"] = pd.to_numeric(
        df["MISS_DISTANCE_KM"],
        errors="coerce",
    )

    df["COLLISION_PROBABILITY_MC"] = pd.to_numeric(
        df["COLLISION_PROBABILITY_MC"],
        errors="coerce",
    )

    df = df.dropna(
        subset=required_columns
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["MISS_DISTANCE_KM"],
            y=df["COLLISION_PROBABILITY_MC"],
            mode="markers",
            marker=dict(
                size=9,
                color="#ef4444",
                opacity=0.8,
            ),
            hovertemplate=(
                "Miss distance: %{x:.3f} km"
                "<br>Collision probability: %{y:.3e}"
                "<extra></extra>"
            ),
            name="Monte Carlo Pc",
        )
    )

    fig.update_layout(
        title=(
            "Collision Probability vs "
            "Miss Distance"
        ),
        xaxis_title="Miss Distance at TCA (km)",
        yaxis_title="Monte Carlo Collision Probability",
        yaxis_type="log",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# ALTITUDE DIFFERENCE VS MISS DISTANCE
# ============================================================

def altitude_difference_vs_miss_distance(
    conjunctions_df,
):
    df = conjunctions_df.copy()

    required_columns = [
        "MISS_DISTANCE_KM",
        "ALTITUDE_DIFFERENCE_KM",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["MISS_DISTANCE_KM"] = pd.to_numeric(
        df["MISS_DISTANCE_KM"],
        errors="coerce",
    )

    df["ALTITUDE_DIFFERENCE_KM"] = pd.to_numeric(
        df["ALTITUDE_DIFFERENCE_KM"],
        errors="coerce",
    )

    df = df.dropna(
        subset=required_columns
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["MISS_DISTANCE_KM"],
            y=df["ALTITUDE_DIFFERENCE_KM"],
            mode="markers",
            marker=dict(
                size=9,
                color="#8b5cf6",
            ),
            hovertemplate=(
                "Miss distance: %{x:.3f} km"
                "<br>Altitude difference: "
                "%{y:.3f} km"
                "<extra></extra>"
            ),
            name="Conjunction pairs",
        )
    )

    fig.update_layout(
        title=(
            "Altitude Difference vs "
            "Miss Distance"
        ),
        xaxis_title="Miss Distance at TCA (km)",
        yaxis_title="Altitude Difference (km)",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# INCLINATION DIFFERENCE VS MISS DISTANCE
# ============================================================

def inclination_difference_vs_miss_distance(
    conjunctions_df,
):
    df = conjunctions_df.copy()

    required_columns = [
        "MISS_DISTANCE_KM",
        "INCLINATION_DIFFERENCE_DEG",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["MISS_DISTANCE_KM"] = pd.to_numeric(
        df["MISS_DISTANCE_KM"],
        errors="coerce",
    )

    df["INCLINATION_DIFFERENCE_DEG"] = pd.to_numeric(
        df["INCLINATION_DIFFERENCE_DEG"],
        errors="coerce",
    )

    df = df.dropna(
        subset=required_columns
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["MISS_DISTANCE_KM"],
            y=df["INCLINATION_DIFFERENCE_DEG"],
            mode="markers",
            marker=dict(
                size=9,
                color="#f59e0b",
            ),
            hovertemplate=(
                "Miss distance: %{x:.3f} km"
                "<br>Inclination difference: "
                "%{y:.3f}°"
                "<extra></extra>"
            ),
            name="Conjunction pairs",
        )
    )

    fig.update_layout(
        title=(
            "Inclination Difference vs "
            "Miss Distance"
        ),
        xaxis_title="Miss Distance at TCA (km)",
        yaxis_title="Inclination Difference (degrees)",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# LIVE GLOBE
# ============================================================

def live_globe_figure(
    current_positions_df,
    timestamp_label=None,
):
    lats = []
    lons = []
    names = []
    colors = []

    for _, row in current_positions_df.iterrows():

        lat, lon, _ = om.eci_to_geodetic(
            row["X_KM"],
            row["Y_KM"],
            row["Z_KM"],
            pd.to_datetime(row["TIME"]),
        )

        lats.append(lat)
        lons.append(lon)

        names.append(
            f"{row['OBJECT_NAME']} "
            f"(NORAD {row['NORAD_CAT_ID']})"
        )

        colors.append(
            "#ef4444"
            if "DEB" in row["OBJECT_NAME"].upper()
            else "#22d3ee"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=dict(
                size=6,
                color=colors,
                line=dict(
                    width=0.5,
                    color="#0b1220",
                ),
            ),
            text=names,
            hovertemplate=(
                "%{text}<extra></extra>"
            ),
        )
    )

    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="#1e293b",
        showocean=True,
        oceancolor="#0b1220",
        showcountries=True,
        countrycolor="#334155",
        bgcolor="#0b1220",
    )

    title = "Live Orbit Tracker"

    if timestamp_label:
        title += f" — {timestamp_label}"

    fig.update_layout(
        title=title,
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# QKD BENCHMARK
# ============================================================

def benchmark_comparison_chart(summaries):

    conditions = [
        s["condition"]
        for s in summaries
    ]

    delivery = [
        s.get("delivery_rate")
        for s in summaries
    ]

    secure_delivery = [
        s.get("secure_delivery_rate")
        for s in summaries
    ]

    detection = [
        s.get("detection_rate")
        for s in summaries
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Delivery rate",
            x=conditions,
            y=delivery,
            marker_color="#22d3ee",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Secure delivery rate",
            x=conditions,
            y=secure_delivery,
            marker_color="#8b5cf6",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Detection rate",
            x=conditions,
            y=detection,
            marker_color="#ef4444",
        )
    )

    fig.update_layout(
        title=(
            "Benchmark: Delivery, Secure Delivery, "
            "and Detection Rate"
        ),
        yaxis_title="Rate",
        yaxis_tickformat=".0%",
        barmode="group",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# QAE VS MONTE CARLO
# ============================================================

def qae_vs_mc_error_chart(sweep_df):

    grouped = (
        sweep_df
        .groupby("EVAL_QUBITS")[
            [
                "QAE_ERROR_MEAN",
                "MC_ERROR_MEAN",
            ]
        ]
        .mean()
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=grouped["EVAL_QUBITS"],
            y=grouped["QAE_ERROR_MEAN"],
            mode="lines+markers",
            name="QAE error",
            line=dict(
                color="#22d3ee"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=grouped["EVAL_QUBITS"],
            y=grouped["MC_ERROR_MEAN"],
            mode="lines+markers",
            name="Classical MC error",
            line=dict(
                color="#f59e0b"
            ),
        )
    )

    fig.update_layout(
        title=(
            "QAE vs Classical Monte Carlo — "
            "Mean Error by Evaluation Qubits"
        ),
        xaxis_title="Evaluation Qubits",
        yaxis_title="Mean Absolute Error",
        yaxis_type="log",
        **DARK_LAYOUT,
    )

    return fig


# ============================================================
# QBER
# ============================================================

def qber_comparison_bar(qkd_summary):

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[
                "No eavesdropper",
                "Eavesdropper",
            ],
            y=[
                qkd_summary["honest_qber"],
                qkd_summary["intercepted_qber"],
            ],
            marker_color=[
                "#22c55e",
                "#ef4444",
            ],
        )
    )

    fig.add_hline(
        y=0.11,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Detection threshold (11%)",
    )

    fig.update_layout(
        title="BB84 QBER: Honest vs Eavesdropped Channel",
        yaxis_title="Quantum Bit Error Rate",
        yaxis_tickformat=".0%",
        **DARK_LAYOUT,
    )

    return fig