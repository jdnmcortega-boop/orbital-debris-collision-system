"""DSST experimental workspace.

This page is intentionally a readiness/preflight screen. It does not claim to
propagate TLEs with DSST. A scientifically defensible DSST run requires an
explicit conversion/initialization workflow from source elements to an
osculating Cartesian orbit and a configured Orekit force model.
"""
from __future__ import annotations

import importlib.util
import streamlit as st


def _orekit_available() -> bool:
    try:
        return importlib.util.find_spec("orekit") is not None
    except (ImportError, ValueError):
        return False


def render_dsst_experimental_tab() -> None:
    st.header("🧪 DSST Experimental — 10,000-Object Research Track")
    st.caption(
        "Separate experimental workspace. The existing SGP4 live forecast and "
        "orbital tracker remain unchanged."
    )

    st.warning(
        "DSST propagation is not implemented yet. This workspace is a readiness "
        "screen only; it does not generate orbital states, conjunctions, or risk results."
    )

    orekit_ready = _orekit_available()
    c1, c2, c3 = st.columns(3)
    c1.metric("Orekit Python package", "Detected" if orekit_ready else "Not detected")
    c2.metric("Target catalog size", "10,000")
    c3.metric("Forecast horizon", "30 days")

    st.subheader("Why the DSST run is gated")
    st.markdown(
        """
        The current catalog is TLE/GP data intended for SGP4. DSST is a
        semi-analytical orbit propagator and cannot be treated as a drop-in
        replacement that consumes the same TLE rows without a validated
        initialization process.

        Before enabling a run, this track must define and test:
        1. A documented conversion/initialization path to osculating Cartesian
           states, including epoch, frame, and time-scale handling.
        2. An Orekit environment and force model (gravity field, atmosphere/
           drag, solar radiation pressure, third-body effects, and required
           space-weather inputs as appropriate to the experiment).
        3. A benchmark against an independent reference trajectory and a
           documented error metric over the 30-day interval.
        4. A memory/runtime benchmark at increasing catalog sizes before
           attempting 10,000 objects on the hosted Streamlit service.
        """
    )

    st.subheader("Readiness status")
    st.write(
        "Orekit package visibility: "
        + ("available in this runtime" if orekit_ready else "not available in this runtime")
    )
    st.write("Orbit initialization and force-model validation: **Not implemented**")
    st.write("Reference-trajectory validation: **Not implemented**")
    st.write("10,000-object performance test: **Not run**")

    st.info(
        "No Run DSST button is shown until the propagator, input conversion, "
        "validation, and resource limits are implemented. This prevents an "
        "experimental placeholder from being mistaken for real DSST results."
    )

    st.subheader("Output contract planned for downstream compatibility")
    st.code(
        "OBJECT_NAME, OBJECT_ID, NORAD_CAT_ID, SOURCE_EPOCH, "
        "FORECAST_START, FORECAST_HORIZON_DAYS, TIME, "
        "X_KM, Y_KM, Z_KM, VX_KM_S, VY_KM_S, VZ_KM_S",
        language="text",
    )
    st.caption(
        "Matching this column contract will allow the conjunction-screening "
        "interface to be adapted later. Numerical results will not be identical "
        "to SGP4 and all downstream Pc/risk/QAE outputs must be regenerated."
    )
