import pandas as pd

from tools.inspect_cdm_csv import build_diagnostic


def test_diagnostic_distinguishes_absent_covariance_from_unsupported_schema():
    absent = pd.DataFrame({
        "SAT_1_ID": [10001],
        "SAT_2_ID": [10002],
        "TCA": ["2026-09-14T03:08:40Z"],
        "PC": [1e-5],
    })
    absent_report = build_diagnostic(absent, "pc_only.csv")
    assert "no covariance/uncertainty-like column names" in absent_report.lower()
    assert "reported pc cannot reconstruct the missing covariance" in absent_report.lower()

    unsupported = pd.DataFrame({
        "SAT_1_ID": [10001],
        "SAT_2_ID": [10002],
        "TCA": ["2026-09-14T03:08:40Z"],
        "SAT_1_POSITION_COVARIANCE": ["1,0,0,0,1,0,0,0,1"],
    })
    unsupported_report = build_diagnostic(unsupported, "alternate_schema.csv")
    assert "uncertainty-like fields exist" in unsupported_report.lower()
    assert "possible parser/schema compatibility issue" in unsupported_report.lower()


def test_diagnostic_counts_recognized_covariance_components_for_both_objects():
    row = {
        "SAT_1_ID": 10001,
        "SAT_2_ID": 10002,
        "TCA": "2026-09-14T03:08:40Z",
        "PC": 1e-5,
    }
    for sat in (1, 2):
        for key, value in {
            "CR_R": 4.0, "CT_R": 0.0, "CT_T": 9.0,
            "CN_R": 0.0, "CN_T": 0.0, "CN_N": 16.0,
        }.items():
            row[f"SAT_{sat}_{key}"] = value
    report = build_diagnostic(pd.DataFrame([row]), "complete.csv")
    assert "Parser-supported covariance component columns found: 12/12" in report
    assert "all 12 expected component columns are recognized" in report.lower()
