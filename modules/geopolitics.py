"""
Geopolitical coordination assessment: identifies which detected
conjunctions require cross-border warning coordination — either because
the two objects belong to different countries, or because the predicted
reentry likelihood means resulting debris could come down anywhere under
the orbit's ground track, not just within the involved operators' own
territory. Flagged events get a simulated secure warning transmission to
each affected country, reusing the classical/QKD security layer.
"""

import json

import pandas as pd

import config
from modules.classical_security import establish_classical_key, encrypt_message, decrypt_message


# Reentry-likelihood bands (from reentry_risk.py) considered urgent enough
# to widen the notification beyond just the two directly involved operators.
URGENT_REENTRY_BANDS = {
    "Imminent (days-weeks)",
    "Very high (weeks-months)",
    "High (months-few years)",
}


def classify_reentry_concern(likelihood_a, likelihood_b):
    return likelihood_a in URGENT_REENTRY_BANDS or likelihood_b in URGENT_REENTRY_BANDS


def build_geopolitical_assessment(predictions_df, reentry_df=None):
    """
    Merge risk/country data (prediction.py) with reentry data (reentry_risk.py,
    optional) and flag which conjunctions require international coordination.

    If reentry_df is None, reentry concern is treated as unknown/False and
    coordination is flagged on cross-border risk alone.
    """
    merged = predictions_df.copy()

    if reentry_df is not None:
        merged = merged.merge(
            reentry_df[["NORAD_A", "NORAD_B", "OBJECT_A_TYPE", "OBJECT_B_TYPE",
                         "COLLISION_TYPE", "REENTRY_LIKELIHOOD_A", "REENTRY_LIKELIHOOD_B"]],
            on=["NORAD_A", "NORAD_B"], how="left",
        )
        merged["REENTRY_CONCERN"] = merged.apply(
            lambda r: classify_reentry_concern(r["REENTRY_LIKELIHOOD_A"], r["REENTRY_LIKELIHOOD_B"]),
            axis=1,
        )
    else:
        merged["OBJECT_A_TYPE"] = "Unknown"
        merged["OBJECT_B_TYPE"] = "Unknown"
        merged["COLLISION_TYPE"] = "Unknown"
        merged["REENTRY_LIKELIHOOD_A"] = "Not assessed"
        merged["REENTRY_LIKELIHOOD_B"] = "Not assessed"
        merged["REENTRY_CONCERN"] = False

    merged["CROSS_BORDER"] = merged["COUNTRY_A"] != merged["COUNTRY_B"]

    # Coordination is needed if the event is a real risk (MEDIUM/HIGH) AND
    # either crosses a border or has debris that could plausibly reach the ground.
    merged["COORDINATION_REQUIRED"] = (
        merged["RISK_LEVEL"].isin(["MEDIUM", "HIGH"])
        & (merged["CROSS_BORDER"] | merged["REENTRY_CONCERN"])
    )

    return merged


def affected_countries(row):
    """
    Countries that should receive a coordination warning for this event.
    Both operators' countries always; if reentry risk is urgent, note that
    the debris fall zone can't be precisely predicted and isn't limited to
    the operators' own countries.
    """
    countries = {row["COUNTRY_A"], row["COUNTRY_B"]} - {"Unknown"}
    return sorted(countries)


def format_coordination_message(row):
    countries = affected_countries(row)
    reentry_note = ""
    if row["REENTRY_CONCERN"]:
        reentry_note = (
            "\nNOTE: Predicted reentry likelihood is elevated for one or both "
            "objects. Debris fall location cannot be precisely predicted in "
            "advance and is not confined to the involved operators' own "
            "territory — broader international notification is recommended "
            "beyond the countries listed below."
        )

    return (
        "=== INTERNATIONAL COORDINATION WARNING ===\n"
        f"Object 1: {row['OBJECT_A']} (NORAD {row['NORAD_A']}) "
        f"- {row['OPERATOR_A']} ({row['COUNTRY_A']}) [{row['OBJECT_A_TYPE']}]\n"
        f"Object 2: {row['OBJECT_B']} (NORAD {row['NORAD_B']}) "
        f"- {row['OPERATOR_B']} ({row['COUNTRY_B']}) [{row['OBJECT_B_TYPE']}]\n"
        f"Collision type: {row['COLLISION_TYPE']}\n"
        f"Predicted TCA: {row['TCA']}\n"
        f"Miss distance: {row['MISS_DISTANCE_KM']:.3f} km\n"
        f"Risk level: {row['RISK_LEVEL']}\n"
        f"Reentry likelihood (A / B): {row['REENTRY_LIKELIHOOD_A']} / {row['REENTRY_LIKELIHOOD_B']}\n"
        f"Directly involved countries: {', '.join(countries) if countries else 'Unknown'}"
        f"{reentry_note}\n"
        "===========================================\n"
    )


def simulate_transmission(message, recipient_country):
    """
    Simulate securely sending one coordination message to one country,
    reusing the classical ECDH + AES-GCM channel from classical_security.py.
    Returns a transmission record for logging/audit purposes.
    """
    handshake = establish_classical_key()
    key = handshake["key"]

    nonce, ciphertext = encrypt_message(key, message)
    decrypted = decrypt_message(key, nonce, ciphertext)

    return {
        "recipient_country": recipient_country,
        "handshake_runtime_sec": handshake["runtime_sec"],
        "ciphertext_bytes": len(ciphertext),
        "delivery_verified": decrypted == message,
    }


def run_and_save():
    config.ensure_dirs()

    predictions_path = config.RESULTS_DIR / "predictions.csv"
    reentry_path = config.RESULTS_DIR / "reentry_analysis.csv"

    if not predictions_path.exists():
        print(f"No predictions file found at {predictions_path}. Run prediction.py first.")
        return None

    predictions = pd.read_csv(predictions_path, parse_dates=["TCA"])

    if reentry_path.exists():
        reentry = pd.read_csv(reentry_path)
    else:
        print(f"No reentry analysis found at {reentry_path} — proceeding with "
              f"cross-border coordination only (run reentry_risk.py to include "
              f"reentry-based coordination too).")
        reentry = None

    assessment = build_geopolitical_assessment(predictions, reentry)

    output_path = config.RESULTS_DIR / "geopolitical_coordination.csv"
    assessment.to_csv(output_path, index=False)

    flagged = assessment[assessment["COORDINATION_REQUIRED"]]

    print(f"Total events assessed: {len(assessment)}")
    print(f"Cross-border events: {assessment['CROSS_BORDER'].sum()}")
    print(f"Reentry-concern events: {assessment['REENTRY_CONCERN'].sum()}")
    print(f"Events requiring coordination: {len(flagged)}")
    print(f"Results written: {output_path}")

    transmission_log = []
    for _, row in flagged.iterrows():
        message = format_coordination_message(row)
        for country in affected_countries(row):
            record = simulate_transmission(message, country)
            record["event"] = f"{row['OBJECT_A']} vs {row['OBJECT_B']}"
            transmission_log.append(record)
            print(f"[SENT] {record['event']} -> {country} "
                  f"(verified={record['delivery_verified']}, "
                  f"{record['ciphertext_bytes']} bytes)")

    log_path = config.RESULTS_DIR / "coordination_transmission_log.json"
    with open(log_path, "w") as f:
        json.dump(transmission_log, f, indent=2, default=str)
    print(f"\nTransmission log written: {log_path} ({len(transmission_log)} message(s) sent)")

    if len(flagged) > 0:
        print("\nSample coordination message:")
        print(format_coordination_message(flagged.iloc[0]))

    return assessment


if __name__ == "__main__":
    run_and_save()