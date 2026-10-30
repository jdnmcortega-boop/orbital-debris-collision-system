# DSST Experimental Environment

This document describes the optional runtime and data preparation workflow for
the separate DSST research workspace. It does not enable DSST propagation by
itself.

## LEO catalog acquisition

The first experiment is scoped to Low Earth Orbit (LEO). The acquisition script
uses CelesTrak's current ACTIVE GP group in OMM-compatible CSV format and
preserves the original download:

```bash
python scripts/build_dsst_leo_catalog.py
```

The script creates:

- `data/dsst_experiment/raw/celestrak_active_gp.csv` — unchanged source payload
- `data/dsst_experiment/objects_10000.csv` — normalized LEO-only records
- `data/dsst_experiment/catalog_audit.csv` — duplicate, invalid, excluded, and retained counts
- `data/dsst_experiment/source_manifest.json` — source, timestamp, filter, and limitations

The filter retains an orbit only when its calculated perigee altitude is above
0 km and its calculated apogee altitude is below 2,000 km. Semimajor axis is
derived from mean motion using Earth's gravitational parameter; perigee and
apogee are then derived using eccentricity and Earth's mean radius. These are
screening estimates based on the supplied GP mean elements, not independent
orbit determinations.

The script caches the source file for two hours and does not automatically retry
provider errors. CelesTrak requests should not be repeated more frequently than
its published update/fair-use interval. If fewer than 10,000 records qualify,
the script reports the actual count and does not pad the dataset with non-LEO
objects. The existing `data/orbital_data.csv` and SGP4 refresh workflow are
not read or modified.

CelesTrak GP OMM data identify the mean-element theory as SGP4. Therefore this
catalog is suitable for reproducible source-data preparation and orbital-regime
screening, but it is **not** a DSST-ready osculating-state catalog. Do not feed
these mean elements directly to DSST as Cartesian initial conditions.

## Why the DSST runtime is optional

DSST requires a defensible initial orbit state, a selected Orekit force model,
and appropriate Orekit environment data. Installing the Python wrapper alone
does not solve those scientific initialization requirements.

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

1. Select and document the GP mean-element to osculating-state initialization
   strategy. Do not interpret raw GP mean elements as osculating Cartesian states.
2. Define UTC/time-scale and reference-frame handling.
3. Select force models and required environmental inputs for the intended
   orbit regime and 30-day interval.
4. Validate propagation against an independent reference and report position
   error over time.
5. Measure wall time and peak memory for 150, 1,000, 5,000, and 10,000 objects.
6. Only after validation, adapt the propagated-state output to the shared
   conjunction-screening contract.

## Current status

The Streamlit DSST workspace remains a readiness screen. The catalog
acquisition script is implemented, but no DSST propagator, initial-state
converter, force model, or benchmark runner is implemented yet. The existing
SGP4 workflow remains the research baseline.
