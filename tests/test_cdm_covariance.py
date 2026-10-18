"""Regression tests for CDM import, covariance aliases, and event matching."""
from io import StringIO

import pandas as pd

from modules.cdm_covariance import (
    load_cdm_csv,
    match_cdms_to_conjunctions,
    validate_cdm_rows,
)


def test_matches_unordered_numeric_ids_and_decimal_csv_ids():
    source = StringIO(
        "SAT_1_ID,SAT_2_ID,TCA,PC,MIN_RNG\n"
        "63921.0,33678.0,2026-09-14T03:08:40Z,0.0003,205\n"
    )
    cdm = load_cdm_csv(source)
    events = pd.DataFrame(
        [{
            "NORAD_A": 33678,
            "NORAD_B": 63921,
            "TCA": "2026-09-14T03:13:40Z",
            "ANALYTIC_PC": 0.0002,
            "MISS_DISTANCE_KM": 205.0,
        }]
    )

    matches = match_cdms_to_conjunctions(cdm, events)

    assert len(matches) == 1
    assert matches.iloc[0]["CDM_MATCH_STATUS"] == "MATCHED_WITHIN_TOLERANCE"
    assert abs(matches.iloc[0]["CDM_TCA_DELTA_MINUTES"] - 5.0) < 1e-9
    assert abs(matches.iloc[0]["PC_ABSOLUTE_DIFFERENCE"] - 0.0001) < 1e-12


def test_recognizes_sat_prefixed_covariance_fields_for_both_objects():
    columns = [
        "SAT_1_ID", "SAT_2_ID", "TCA", "PC",
        "SAT_1_CR_R", "SAT_1_CT_R", "SAT_1_CT_T",
        "SAT_1_CN_R", "SAT_1_CN_T", "SAT_1_CN_N",
        "SAT_2_CR_R", "SAT_2_CT_R", "SAT_2_CT_T",
        "SAT_2_CN_R", "SAT_2_CN_T", "SAT_2_CN_N",
    ]
    row = {
        "SAT_1_ID": 10001, "SAT_2_ID": 10002,
        "TCA": "2026-09-14T03:08:40Z", "PC": 1e-5,
    }
    for sat in (1, 2):
        row.update({
            f"SAT_{sat}_CR_R": 4.0,
            f"SAT_{sat}_CT_R": 0.0,
            f"SAT_{sat}_CT_T": 9.0,
            f"SAT_{sat}_CN_R": 0.0,
            f"SAT_{sat}_CN_T": 0.0,
            f"SAT_{sat}_CN_N": 16.0,
        })
    cdm = load_cdm_csv(StringIO(pd.DataFrame([row], columns=columns).to_csv(index=False)))
    validation = validate_cdm_rows(cdm)

    assert validation.iloc[0]["COVARIANCE_VALIDATION_STATUS"] == "BOTH_POSITION_BLOCKS_VALID"
    assert validation.iloc[0]["OBJECT1_SIGMA_R_KM"] == 0.002
    assert validation.iloc[0]["OBJECT2_SIGMA_N_KM"] == 0.004
