"""
Standalone benchmark (NOT part of the main pipeline): repeated-trial
comparison of QKD vs classical secure communication, producing the actual
RATES needed for objective 4 (delivery rate, communication overhead,
key-establishment performance, interception detection rate) and the data
needed to test H03. A single run of classical_security.py / qkd.py only
gives one delivered/not-delivered, one detected/not-detected outcome —
a rate requires many repeated trials, which is what this script adds.
"""

# Must be set before numpy/pandas/qiskit load anywhere in the process —
# see the matching comment in modules/qkd.py for why.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import json

import numpy as np
import pandas as pd

from modules.classical_security import establish_classical_key, encrypt_message, decrypt_message
from modules.qkd import simulate_bb84
import config


N_TRIALS = 50       # trials per condition (honest QKD, eavesdropped QKD, classical)
QKD_N_QUBITS = 256  # raw qubits sent per BB84 trial


def get_test_message():
    warnings_path = config.RESULTS_DIR / "warnings.txt"
    if warnings_path.exists():
        message = warnings_path.read_text().strip()
        if message and not message.startswith("No MEDIUM/HIGH"):
            return message
    return ("=== COLLISION WARNING (sample) ===\n"
            "No real MEDIUM/HIGH risk event on file — using a placeholder "
            "message for benchmarking.")


def run_classical_trials(message, n_trials=N_TRIALS):
    rows = []
    for i in range(n_trials):
        handshake = establish_classical_key()
        key = handshake["key"]
        nonce, ciphertext = encrypt_message(key, message)
        decrypted = decrypt_message(key, nonce, ciphertext)

        rows.append({
            "trial": i,
            "delivered": decrypted == message,
            "handshake_runtime_sec": handshake["runtime_sec"],
            "overhead_bytes": handshake["public_key_bytes_exchanged"],
            "ciphertext_bytes": len(ciphertext),
            # Classical ECDH has no built-in eavesdropping-detection mechanism —
            # an intercepted public-key exchange produces no observable signal
            # unless combined with a separate authentication/PKI scheme, which
            # is out of scope here. Reported as structurally undetectable,
            # not as a failed detection attempt.
            "eavesdropping_detected": None,
        })
    return pd.DataFrame(rows)


def run_qkd_trials(message, eavesdrop, n_trials=N_TRIALS, n_qubits=QKD_N_QUBITS):
    rows = []
    for i in range(n_trials):
        result = simulate_bb84(n_qubits, eavesdrop=eavesdrop, seed=None)

        # Only attempt message delivery if the key survived (enough sifted
        # bits left after the QBER-check reveal to derive a key at all)
        delivered = False
        if result["final_key_length_bits"] > 0:
            nonce, ciphertext = encrypt_message(result["final_key"], message)
            decrypted = decrypt_message(result["final_key"], nonce, ciphertext)
            delivered = decrypted == message

        rows.append({
            "trial": i,
            "delivered": delivered,
            "eavesdropping_detected": result["eavesdropping_detected"],
            # Real BB84 practice: abort and discard the key if eavesdropping
            # is detected. "delivered" alone can misleadingly show 100% even
            # under active attack, since decryption still technically works
            # with a compromised key — this reflects whether the message
            # was delivered on a key that was NOT flagged as compromised.
            "secure_delivery": delivered and not result["eavesdropping_detected"],
            "protocol_runtime_sec": result["runtime_sec"],
            "qubits_transmitted": result["qubits_transmitted"],
            "classical_bits_for_reconciliation": result["classical_bits_for_basis_reconciliation"],
            "classical_bits_for_qber_check": result["classical_bits_for_qber_check"],
            "final_key_length_bits": result["final_key_length_bits"],
            "qber": result["qber"],
        })
    return pd.DataFrame(rows)


def summarize(df, label):
    summary = {
        "condition": label,
        "n_trials": len(df),
        "delivery_rate": float(df["delivered"].mean()),
    }

    if "handshake_runtime_sec" in df.columns:
        summary["mean_runtime_sec"] = float(df["handshake_runtime_sec"].mean())
        summary["mean_overhead_bytes"] = float(df["overhead_bytes"].mean())
    else:
        summary["mean_runtime_sec"] = float(df["protocol_runtime_sec"].mean())
        summary["mean_qubits_transmitted"] = float(df["qubits_transmitted"].mean())
        summary["mean_classical_bits_total"] = float(
            (df["classical_bits_for_reconciliation"] + df["classical_bits_for_qber_check"]).mean()
        )
        summary["mean_final_key_length_bits"] = float(df["final_key_length_bits"].mean())
        summary["mean_qber"] = float(df["qber"].mean())
        summary["secure_delivery_rate"] = float(df["secure_delivery"].mean())

    if df["eavesdropping_detected"].notna().any():
        summary["detection_rate"] = float(df["eavesdropping_detected"].mean())
    else:
        summary["detection_rate"] = None  # classical: structurally undetectable

    return summary


def run_and_save():
    config.ensure_dirs()
    message = get_test_message()

    print(f"Running {N_TRIALS} trials per condition...")

    print("Classical (ECDH + AES-GCM)...")
    classical_df = run_classical_trials(message)

    print("QKD honest channel...")
    qkd_honest_df = run_qkd_trials(message, eavesdrop=False)

    print("QKD eavesdropped (intercept-resend)...")
    qkd_intercepted_df = run_qkd_trials(message, eavesdrop=True)

    summaries = [
        summarize(classical_df, "Classical (ECDH+AES-GCM)"),
        summarize(qkd_honest_df, "QKD - honest channel"),
        summarize(qkd_intercepted_df, "QKD - eavesdropped (intercept-resend)"),
    ]

    print("\n=== Benchmark Summary ===")
    for s in summaries:
        print(f"\n{s['condition']} (n={s['n_trials']}):")
        for k, v in s.items():
            if k in ("condition", "n_trials"):
                continue
            print(f"  {k}: {v}")

    classical_df.to_csv(config.RESULTS_DIR / "benchmark_classical_trials.csv", index=False)
    qkd_honest_df.to_csv(config.RESULTS_DIR / "benchmark_qkd_honest_trials.csv", index=False)
    qkd_intercepted_df.to_csv(config.RESULTS_DIR / "benchmark_qkd_eavesdropped_trials.csv", index=False)

    summary_path = config.RESULTS_DIR / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    print(f"\nPer-trial data and summary written to {config.RESULTS_DIR}")

    return classical_df, qkd_honest_df, qkd_intercepted_df, summaries


if __name__ == "__main__":
    run_and_save()