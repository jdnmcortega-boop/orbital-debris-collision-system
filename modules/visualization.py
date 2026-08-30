"""
Reusable Plotly figure-generating functions, shared by the Streamlit
dashboard (ui/dashboard.py) and available for standalone use/export.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
import orbital_mechanics as om


DARK_LAYOUT = dict(
    paper_bgcolor="#0b1220",
    plot_bgcolor="#0b1220",
    font=dict(color="#e2e8f0"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ============================================================
# ALTITUDE HISTORY (matches the reference "Altitude History" panel)
# ============================================================

def altitude_history_figure(propagated_df, norad_id, object_name):
    obj = propagated_df[propagated_df["NORAD_CAT_ID"] == norad_id].copy()
    obj["TIME"] = pd.to_datetime(obj["TIME"])
    obj = obj.sort_values("TIME")

    altitude = np.sqrt(obj["X_KM"]**2 + obj["Y_KM"]**2 + obj["Z_KM"]**2) - om.EARTH_RADIUS_KM

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obj["TIME"], y=altitude, mode="lines",
        line=dict(color="#f59e0b", width=2), name="Altitude (km)",
    ))
    fig.update_layout(
        title=f"Altitude History — {object_name}",
        xaxis_title="Time (UTC)", yaxis_title="Altitude (km)",
        **DARK_LAYOUT,
    )
    return fig


# ============================================================
# GROUND TRACK / SUB-SATELLITE POINT (matches the reference globe view)
# ============================================================

def ground_track_figure(propagated_df, norad_id, object_name, max_points=300):
    obj = propagated_df[propagated_df["NORAD_CAT_ID"] == norad_id].copy()
    obj["TIME"] = pd.to_datetime(obj["TIME"])
    obj = obj.sort_values("TIME")

    # Subsample for rendering performance on long time grids
    if len(obj) > max_points:
        obj = obj.iloc[:: len(obj) // max_points]

    lats, lons = [], []
    for _, row in obj.iterrows():
        lat, lon, _ = om.eci_to_geodetic(row["X_KM"], row["Y_KM"], row["Z_KM"], row["TIME"])
        lats.append(lat)
        lons.append(lon)

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons, mode="lines",
        line=dict(width=1, color="#f59e0b"), name="Ground track",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[lats[0]], lon=[lons[0]], mode="markers",
        marker=dict(size=9, color="#22d3ee"), name=object_name,
    ))
    fig.update_geos(
        projection_type="orthographic",
        showland=True, landcolor="#1e293b",
        showocean=True, oceancolor="#0b1220",
        showcountries=True, countrycolor="#334155",
        bgcolor="#0b1220",
    )
    fig.update_layout(title=f"Ground Track — {object_name}", **DARK_LAYOUT)
    return fig


# ============================================================
# CONJUNCTION / RISK CHARTS
# ============================================================

def risk_level_bar(predictions_df):
    counts = predictions_df["RISK_LEVEL"].value_counts()
    order = [lvl for lvl in ["LOW", "MEDIUM", "HIGH"] if lvl in counts.index]
    colors = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}

    fig = go.Figure(go.Bar(
        x=order, y=[counts[lvl] for lvl in order],
        marker_color=[colors[lvl] for lvl in order],
    ))
    fig.update_layout(title="Conjunctions by Risk Level",
                       xaxis_title="Risk level", yaxis_title="Count", **DARK_LAYOUT)
    return fig


def miss_distance_scatter(conjunctions_df):
    fig = go.Figure(go.Scatter(
        x=conjunctions_df["MISS_DISTANCE_KM"],
        y=conjunctions_df["RELATIVE_VELOCITY_KM_S"],
        mode="markers",
        marker=dict(size=8, color=conjunctions_df["MISS_DISTANCE_KM"],
                    colorscale="Turbo_r", showscale=True,
                    colorbar=dict(title="Miss dist (km)")),
        text=conjunctions_df["OBJECT_A"] + " vs " + conjunctions_df["OBJECT_B"],
        hovertemplate="%{text}<br>Miss: %{x:.2f} km<br>Rel. vel: %{y:.2f} km/s",
    ))
    fig.update_layout(title="Conjunctions: Miss Distance vs Relative Velocity",
                       xaxis_title="Miss distance (km)", yaxis_title="Relative velocity (km/s)",
                       **DARK_LAYOUT)
    return fig


def live_globe_figure(current_positions_df, timestamp_label=None):
    """
    All objects' CURRENT positions on one globe, colored by satellite (cyan)
    vs debris (red). Meant to be recomputed and redrawn on each live refresh.
    """
    lats, lons, names, colors = [], [], [], []
    for _, row in current_positions_df.iterrows():
        lat, lon, _ = om.eci_to_geodetic(
            row["X_KM"], row["Y_KM"], row["Z_KM"], pd.to_datetime(row["TIME"])
        )
        lats.append(lat)
        lons.append(lon)
        names.append(f"{row['OBJECT_NAME']} (NORAD {row['NORAD_CAT_ID']})")
        colors.append("#ef4444" if "DEB" in row["OBJECT_NAME"].upper() else "#22d3ee")

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, mode="markers",
        marker=dict(size=6, color=colors, line=dict(width=0.5, color="#0b1220")),
        text=names, hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_geos(
        projection_type="orthographic",
        showland=True, landcolor="#1e293b",
        showocean=True, oceancolor="#0b1220",
        showcountries=True, countrycolor="#334155",
        bgcolor="#0b1220",
    )
    title = "Live Orbit Tracker"
    if timestamp_label:
        title += f" — {timestamp_label}"
    fig.update_layout(title=title, **DARK_LAYOUT)
    return fig


def benchmark_comparison_chart(summaries):
    """
    Grouped bar chart comparing delivery rate, secure delivery rate, and
    detection rate across Classical / QKD-honest / QKD-eavesdropped, from
    qkd_benchmark.py's repeated-trial results.
    """
    conditions = [s["condition"] for s in summaries]

    delivery = [s.get("delivery_rate") for s in summaries]
    secure_delivery = [s.get("secure_delivery_rate") for s in summaries]
    detection = [s.get("detection_rate") for s in summaries]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Delivery rate", x=conditions, y=delivery,
                          marker_color="#22d3ee"))
    fig.add_trace(go.Bar(name="Secure delivery rate", x=conditions, y=secure_delivery,
                          marker_color="#8b5cf6"))
    fig.add_trace(go.Bar(name="Detection rate", x=conditions, y=detection,
                          marker_color="#ef4444"))

    fig.update_layout(
        title="Benchmark: Delivery, Secure Delivery, and Detection Rate (50 trials/condition)",
        yaxis_title="Rate", yaxis_tickformat=".0%", barmode="group",
        **DARK_LAYOUT,
    )
    return fig


# ============================================================
# QAE vs CLASSICAL COMPARISON
# ============================================================

def qae_vs_mc_error_chart(sweep_df):
    grouped = sweep_df.groupby("EVAL_QUBITS")[["QAE_ERROR_MEAN", "MC_ERROR_MEAN"]].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["EVAL_QUBITS"], y=grouped["QAE_ERROR_MEAN"],
                              mode="lines+markers", name="QAE error",
                              line=dict(color="#22d3ee")))
    fig.add_trace(go.Scatter(x=grouped["EVAL_QUBITS"], y=grouped["MC_ERROR_MEAN"],
                              mode="lines+markers", name="Classical MC error",
                              line=dict(color="#f59e0b")))
    fig.update_layout(title="QAE vs Classical Monte Carlo — Mean Error by Query Budget",
                       xaxis_title="Evaluation qubits (query budget)",
                       yaxis_title="Mean absolute error (log scale)",
                       yaxis_type="log",
                       **DARK_LAYOUT)
    return fig


# ============================================================
# SECURITY / QKD
# ============================================================

def qber_comparison_bar(qkd_summary):
    fig = go.Figure(go.Bar(
        x=["No eavesdropper", "Eavesdropper (intercept-resend)"],
        y=[qkd_summary["honest_qber"], qkd_summary["intercepted_qber"]],
        marker_color=["#22c55e", "#ef4444"],
    ))
    fig.add_hline(y=0.11, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Detection threshold (11%)")
    fig.update_layout(title="BB84 QBER: Honest vs Eavesdropped Channel",
                       yaxis_title="Quantum Bit Error Rate", yaxis_tickformat=".0%",
                       **DARK_LAYOUT)
    return fig