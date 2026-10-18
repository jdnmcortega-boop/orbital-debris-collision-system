"""Historical CDM validation workbench.

Upload a public CDM reference CSV and a separate ORION-X conjunction CSV
generated for the same historical interval. This page diagnoses pair/time
overlap before comparing reported Pc values. It does not infer covariance
from Pc or claim that agreement alone validates the physical model.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.cdm_covariance import (
    load_cdm_csv,
    match_cdms_to_conjunctions,
    validate_cdm_rows,
    has_position_covariance_fields,
)

st.set_page_config(page_title="CDM Historical Validation", page_icon="🛰️", layout="wide")
st.title("🛰️ CDM Historical Validation Workbench")
st.caption(
    "Use a historical ORION-X event file from the same time period as the CDM export. "
    "The current live forecast is not expected to match older CDM events."
)

st.warning(
    "Do not compare different time windows as if they were the same conjunctions. "
    "A reported CDM Pc is a reference value; it cannot be used to reconstruct missing covariance."
)

left, right = st.columns(2)
with left:
    cdm_upload = st.file_uploader(
        "1. Public CDM CSV",
        type=["csv"],
        key="historical_cdm_upload",
        help="Expected to contain TCA, both object IDs, and optionally PC / MIN_RNG.",
    )
with right:
    orion_upload = st.file_uploader(
        "2. Historical ORION-X conjunction CSV",
        type=["csv"],
        key="historical_orion_upload",
        help="Must contain NORAD_A, NORAD_B, and TCA. Preferably also ANALYTIC_PC and MISS_DISTANCE_KM.",
    )

tolerance = st.number_input(
    "TCA matching tolerance (minutes)",
    min_value=1.0,
    max_value=60.0,
    value=10.0,
    step=1.0,
)

def norm_id(value):
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if not value:
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return value.upper()


def pair_set(frame, a_col, b_col):
    pairs = set()
    for _, row in frame.iterrows():
        a, b = norm_id(row.get(a_col)), norm_id(row.get(b_col))
        if a and b:
            pairs.add(tuple(sorted((a, b))))
    return pairs


def find_id_source_column(columns, sat):
    """Mirror the ID precedence used by modules.cdm_covariance."""
    candidates = [
        f"SAT_{sat}_NORAD_ID", f"SAT{sat}_NORAD_ID",
        f"SAT_{sat}_NORAD_CAT_ID", f"SAT{sat}_NORAD_CAT_ID",
        f"SAT_{sat}_ID", f"SAT{sat}_ID",
        f"OBJECT{sat}_NORAD_CAT_ID",
        f"SAT{sat}_OBJECT_DESIGNATOR", f"SAT_{sat}_OBJECT_DESIGNATOR",
        f"SAT{sat}_OBJECT", f"SAT_{sat}_OBJECT",
        f"OBJECT{sat}_OBJECT_DESIGNATOR",
    ]
    lookup = {str(col).strip().upper().replace("-", "_").replace(" ", "_"): col for col in columns}
    for candidate in candidates:
        key = candidate.upper()
        if key in lookup:
            return lookup[key]
    return None


if cdm_upload is not None:
    try:
        cdm = load_cdm_csv(cdm_upload)
        cdm_validation = validate_cdm_rows(cdm)
        cdm_tca = pd.to_datetime(cdm["_CDM_TCA"], utc=True, errors="coerce").dropna()
        cdm_pairs = pair_set(cdm, "_CDM_OBJECT1", "_CDM_OBJECT2")
        pc_count = (
            pd.to_numeric(cdm_validation["CDM_REPORTED_PC"], errors="coerce").notna().sum()
            if "CDM_REPORTED_PC" in cdm_validation else 0
        )
        cov_present = has_position_covariance_fields(cdm.columns)
        cdm_id_col1 = find_id_source_column(cdm.columns, 1)
        cdm_id_col2 = find_id_source_column(cdm.columns, 2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CDM records", f"{len(cdm):,}")
        m2.metric("Rows with both IDs", f"{sum(bool(norm_id(r.get('_CDM_OBJECT1'))) and bool(norm_id(r.get('_CDM_OBJECT2'))) for _, r in cdm.iterrows()):,}")
        m3.metric("Unique unordered pairs", f"{len(cdm_pairs):,}")
        m4.metric("Covariance fields", "Detected" if cov_present else "Not present")

        st.write(
            "**CDM TCA range:** "
            + (f"{cdm_tca.min():%Y-%m-%d %H:%M UTC} to {cdm_tca.max():%Y-%m-%d %H:%M UTC}" if not cdm_tca.empty else "Unavailable")
        )
        st.write(f"**Records with reported Pc:** {int(pc_count):,}")

        with st.expander("Inspect source columns and object-ID samples", expanded=True):
            st.write("**CDM ID columns selected by the parser**")
            st.write(f"Object 1: `{cdm_id_col1 or 'NOT FOUND'}`")
            st.write(f"Object 2: `{cdm_id_col2 or 'NOT FOUND'}`")
            st.caption(
                "Catalog IDs (usually numeric NORAD IDs) should be used for matching when available. "
                "International designators such as 1997-051A are a different identifier type."
            )
            sample = cdm.head(12).copy()
            sample_view = pd.DataFrame({
                "CDM source row": sample["_CDM_SOURCE_ROW"],
                "Selected object 1 column": sample[cdm_id_col1] if cdm_id_col1 else "",
                "Parsed object 1 ID": sample["_CDM_OBJECT1"],
                "Selected object 2 column": sample[cdm_id_col2] if cdm_id_col2 else "",
                "Parsed object 2 ID": sample["_CDM_OBJECT2"],
                "TCA": sample["_CDM_TCA"],
            })
            st.dataframe(sample_view, width="stretch", hide_index=True)

        if not cov_present:
            st.info(
                "This export has no recognized flattened RTN position-covariance fields. "
                "Reported Pc values can be used for reference comparison, but cannot reconstruct covariance."
            )

        if orion_upload is not None:
            orion = pd.read_csv(orion_upload)
            required = {"NORAD_A", "NORAD_B", "TCA"}
            if not required.issubset(orion.columns):
                st.error("The ORION-X CSV must contain NORAD_A, NORAD_B, and TCA columns.")
            else:
                orion = orion.copy()
                orion_tca = pd.to_datetime(orion["TCA"], utc=True, errors="coerce")
                orion_pairs = pair_set(orion, "NORAD_A", "NORAD_B")
                shared_pairs = cdm_pairs.intersection(orion_pairs)
                valid_tca_count = int(orion_tca.notna().sum())

                st.subheader("Overlap diagnostics")
                a, b, c, d = st.columns(4)
                a.metric("ORION-X rows", f"{len(orion):,}")
                b.metric("Unique ORION-X pairs", f"{len(orion_pairs):,}")
                c.metric("Shared object pairs", f"{len(shared_pairs):,}")
                d.metric("ORION-X rows with valid TCA", f"{valid_tca_count:,}")
                st.write(
                    "**ORION-X TCA range:** "
                    + (f"{orion_tca.min():%Y-%m-%d %H:%M UTC} to {orion_tca.max():%Y-%m-%d %H:%M UTC}" if orion_tca.notna().any() else "Unavailable")
                )

                with st.expander("Inspect ORION-X IDs and pair overlap", expanded=not bool(shared_pairs)):
                    st.write("**ORION-X identifier columns:** `NORAD_A`, `NORAD_B`")
                    st.dataframe(
                        orion[["NORAD_A", "NORAD_B", "TCA"]].head(12),
                        width="stretch",
                        hide_index=True,
                    )
                    st.write("**Example normalized object pairs**")
                    cdm_pair_examples = sorted(cdm_pairs)[:10]
                    orion_pair_examples = sorted(orion_pairs)[:10]
                    x, y = st.columns(2)
                    with x:
                        st.caption("CDM pairs")
                        st.dataframe(pd.DataFrame(cdm_pair_examples, columns=["Object 1", "Object 2"]), width="stretch", hide_index=True)
                    with y:
                        st.caption("ORION-X pairs")
                        st.dataframe(pd.DataFrame(orion_pair_examples, columns=["Object 1", "Object 2"]), width="stretch", hide_index=True)
                    st.caption(
                        f"Shared pairs: {len(shared_pairs):,}. Pair matching ignores object order, "
                        "but does not treat international designators as equivalent to NORAD catalog IDs."
                    )

                if not shared_pairs:
                    st.error(
                        "No object pairs overlap. Matching cannot proceed, even if dates overlap. "
                        "Check ID type (NORAD catalog ID versus international designator), the selected ORION-X file, "
                        "and whether both datasets cover the same catalog."
                    )
                elif cdm_tca.empty or not orion_tca.notna().any():
                    st.error("Object pairs overlap, but at least one dataset has no parseable TCA values.")
                else:
                    cdm_min, cdm_max = cdm_tca.min(), cdm_tca.max()
                    orion_valid_tca = orion_tca.dropna()
                    ranges_overlap = max(cdm_min, orion_valid_tca.min()) <= min(cdm_max, orion_valid_tca.max())
                    if not ranges_overlap:
                        st.warning(
                            "Object pairs overlap, but the TCA date ranges do not. Upload or generate ORION-X events "
                            "for the CDM's historical interval before expecting 10-minute matches."
                        )
                    matches = match_cdms_to_conjunctions(cdm, orion, tolerance_minutes=float(tolerance))
                    if not matches.empty:
                        status = matches["CDM_MATCH_STATUS"].astype(str)
                        matched = status.eq("MATCHED_WITHIN_TOLERANCE")
                        pair_only = status.eq("PAIR_MATCH_TCA_OUTSIDE_TOLERANCE")
                        x, y, z = st.columns(3)
                        x.metric("Matched within tolerance", int(matched.sum()))
                        y.metric("Same pair, TCA too far", int(pair_only.sum()))
                        z.metric("No pair match", int(status.eq("NO_OBJECT_PAIR_MATCH").sum()))
                        if matched.any():
                            st.success("Matched events are available for reported-Pc comparison.")
                        else:
                            st.warning(
                                "No event matched within the tolerance. Inspect pair matches and TCA deltas; "
                                "do not interpret zero matches as a model result."
                            )
                        st.dataframe(matches, width="stretch", hide_index=True)
                        st.download_button(
                            "Download historical CDM comparison report",
                            data=matches.to_csv(index=False).encode("utf-8"),
                            file_name="orion_x_historical_cdm_comparison.csv",
                            mime="text/csv",
                        )
        else:
            st.info(
                "Upload a historical ORION-X conjunction CSV generated over the CDM's TCA interval. "
                "The current live forecast may use a different date range and different object pairs."
            )

        with st.expander("CDM field-level covariance validation"):
            st.dataframe(cdm_validation, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Could not process the uploaded files: {exc}")
else:
    st.info("Upload the public CDM CSV to begin. Then upload a matching-period ORION-X event CSV.")
