# DSST Experimental Environment

This document describes the optional runtime for the separate DSST research
workspace. It does not enable DSST propagation by itself.

## Why this is optional

The production SGP4 path consumes TLE/GP elements. DSST requires a defensible
initial orbit state, a selected Orekit force model, and appropriate Orekit
environment data. Installing the Python wrapper alone does not solve those
scientific initialization requirements.

The project uses a separate requirements file so the existing SGP4 deployment
does not acquire a Java runtime dependency or become unavailable if the
experimental package fails to install.

## Development setup

Use a dedicated Python environment. The Orekit JPype wrapper documents Python
3.8+ and requires Java 11+ when using an external JDK. Its `jdk4py` extra can
provide a JDK through Python packaging.

Install the optional wrapper:

```bash
python -m pip install -r requirements-dsst.txt
```

The pinned wrapper version is an experimental baseline, not a claim that the
full ORION-X application or Streamlit Community Cloud deployment has been
validated with it. Record the Python version, Java runtime, operating system,
Orekit version, and data package version in every benchmark.

## Required implementation gates

1. Select and document the TLE-to-osculating-state initialization strategy.
   Do not interpret raw TLE mean elements as osculating Cartesian states.
2. Define UTC/time-scale and reference-frame handling.
3. Select force models and required environmental inputs for the intended
   orbit regime and 30-day interval.
4. Validate propagation against an independent reference and report position
   error over time.
5. Measure wall time and peak memory for 150, 1,000, 5,000, and 10,000 objects.
6. Only after validation, adapt the propagated-state output to the shared
   conjunction-screening contract.

## Current status

The Streamlit DSST workspace is a readiness screen. No DSST propagator,
initial-state converter, force model, or benchmark runner is implemented yet.
The existing SGP4 workflow remains the operational research baseline.
