"""
Simulated BB84 Quantum Key Distribution, including an intercept-resend
eavesdropping scenario to test whether the disturbance it causes is
detectable via the quantum bit error rate (QBER).

The resulting shared key is used with the same AES-GCM encryption as
classical_security.py, so both methods protect the identical message
and are directly comparable.
"""

# MUST be set before numpy/qiskit are imported anywhere in the process.
# On Windows, numpy (via Intel MKL) and qiskit-aer each bundle their own
# OpenMP runtime; both initializing in the same process can crash with an
# access violation (0xC0000005) instead of a catchable Python error. This
# tells the runtime to tolerate the duplicate rather than crash.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import hashlib

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import config
from modules.classical_security import encrypt_message, decrypt_message


# max_parallel_threads=1 forces single-threaded execution, avoiding any
# further OpenMP thread-pool conflict on top of the env-var fix above.
SIMULATOR = AerSimulator(max_parallel_threads=1, max_parallel_experiments=1)


# ============================================================
# BB84 QUBIT ENCODING / DECODING
# ============================================================

def run_channel(bits, prep_bases, measure_bases):
    """
    Run n independent single-qubit BB84 transmissions, batched into ONE
    Aer call (a list of small circuits) rather than n separate .run() calls.
    Each circuit is still only 1 qubit — this avoids both the earlier
    exponential state-vector blowup (batching into ONE n-qubit circuit)
    and native-backend instability from too many rapid individual .run()
    invocations (observed as a Windows access-violation crash at n=256+).
    """
    n = len(bits)
    circuits = []
    for i in range(n):
        qc = QuantumCircuit(1, 1)
        if bits[i] == 1:
            qc.x(0)
        if prep_bases[i] == 1:
            qc.h(0)
        if measure_bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)
        circuits.append(qc)

    result = SIMULATOR.run(circuits, shots=1).result()

    measured_bits = []
    for i in range(n):
        counts = result.get_counts(i)
        measured_bits.append(int(list(counts.keys())[0]))
    return measured_bits


# ============================================================
# BB84 PROTOCOL
# ============================================================

def simulate_bb84(n_qubits, eavesdrop=False, reveal_fraction=0.5, seed=None):
    """
    Run one full BB84 round: Alice prepares n_qubits, optionally Eve
    intercepts and resends, Bob measures, both sift on matching bases,
    and a fraction of the sifted key is revealed to estimate QBER.
    """
    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(0, 2, n_qubits).tolist()
    alice_bases = rng.integers(0, 2, n_qubits).tolist()
    bob_bases = rng.integers(0, 2, n_qubits).tolist()

    start = time.time()

    if eavesdrop:
        # Stage 1: Alice -> Eve. Eve measures in her own random basis
        # (she doesn't know Alice's), collapsing the state.
        eve_bases = rng.integers(0, 2, n_qubits).tolist()
        eve_bits = run_channel(alice_bits, alice_bases, eve_bases)

        # Stage 2: Eve -> Bob. Eve has no quantum memory, so she must
        # re-prepare a fresh qubit from her (possibly wrong) measurement
        # and send that on to Bob.
        bob_bits = run_channel(eve_bits, eve_bases, bob_bases)
        qubits_transmitted = 2 * n_qubits  # Alice->Eve, then Eve->Bob
    else:
        # Direct Alice -> Bob channel, no interception
        bob_bits = run_channel(alice_bits, alice_bases, bob_bases)
        qubits_transmitted = n_qubits

    runtime = time.time() - start

    # --- Sifting: keep positions where Alice and Bob used the same basis ---
    sifted_indices = [i for i in range(n_qubits) if alice_bases[i] == bob_bases[i]]
    alice_sifted = [alice_bits[i] for i in sifted_indices]
    bob_sifted = [bob_bits[i] for i in sifted_indices]

    # --- QBER estimation: publicly reveal a fraction of the sifted key ---
    n_sifted = len(sifted_indices)
    n_reveal = max(1, int(n_sifted * reveal_fraction)) if n_sifted > 0 else 0

    reveal_idx = rng.choice(n_sifted, size=n_reveal, replace=False) if n_sifted > 0 else []
    mismatches = sum(1 for i in reveal_idx if alice_sifted[i] != bob_sifted[i])
    qber = mismatches / n_reveal if n_reveal > 0 else 0.0

    # Remaining (unrevealed) sifted bits become the final secret key
    keep_idx = [i for i in range(n_sifted) if i not in set(reveal_idx)]
    final_key_bits = [alice_sifted[i] for i in keep_idx]

    # Derive a fixed-length AES key from the raw key bits via SHA-256
    # (simple key-derivation / privacy amplification step)
    bit_string = "".join(str(b) for b in final_key_bits)
    final_key = hashlib.sha256(bit_string.encode("utf-8")).digest()

    # QBER threshold: standard BB84 security bound is ~11%; intercept-
    # resend attacks on a full basis mismatch induce ~25% QBER, so this
    # threshold reliably flags eavesdropping without false-flagging noise.
    eavesdropping_detected = qber > 0.11

    return {
        "n_qubits_sent": n_qubits,
        "qubits_transmitted": qubits_transmitted,  # includes Eve relay if present
        "classical_bits_for_basis_reconciliation": 2 * n_qubits,  # both parties announce bases
        "n_sifted": n_sifted,
        "n_revealed_for_qber_check": n_reveal,
        "classical_bits_for_qber_check": n_reveal,
        "qber": qber,
        "eavesdropping_detected": eavesdropping_detected,
        "final_key_length_bits": len(final_key_bits),
        "final_key": final_key,
        "runtime_sec": runtime,
        "eavesdrop_scenario": eavesdrop,
    }


# ============================================================
# PIPELINE
# ============================================================

def run_and_save(n_qubits=512):
    config.ensure_dirs()

    print(f"Running BB84 with {n_qubits} qubits (honest channel)...")
    honest = simulate_bb84(n_qubits, eavesdrop=False)

    print(f"Running BB84 with {n_qubits} qubits (eavesdropping scenario)...")
    intercepted = simulate_bb84(n_qubits, eavesdrop=True)

    print("\n=== BB84 Comparison ===")
    for label, result in [("No eavesdropper", honest), ("Eavesdropper (intercept-resend)", intercepted)]:
        print(f"\n{label}:")
        print(f"  Sifted key length: {result['n_sifted']} bits")
        print(f"  QBER: {result['qber']:.2%}")
        print(f"  Eavesdropping detected: {result['eavesdropping_detected']}")
        print(f"  Final key length: {result['final_key_length_bits']} bits")
        print(f"  Runtime: {result['runtime_sec']:.3f} sec")

    # Use the honest run's key to protect the actual warning message
    warnings_path = config.RESULTS_DIR / "warnings.txt"
    if warnings_path.exists():
        message = warnings_path.read_text().strip()
        if not message or message.startswith("No MEDIUM/HIGH"):
            message = ("=== COLLISION WARNING (sample) ===\n"
                       "No real MEDIUM/HIGH risk event on file — using a "
                       "placeholder message to demonstrate the security layer.")
    else:
        message = "=== COLLISION WARNING (sample) ===\nPlaceholder message."

    nonce, ciphertext = encrypt_message(honest["final_key"], message)
    decrypted = decrypt_message(honest["final_key"], nonce, ciphertext)
    assert decrypted == message, "QKD-keyed round-trip decryption mismatch"

    summary = {
        "honest_qber": honest["qber"],
        "honest_eavesdropping_detected": honest["eavesdropping_detected"],
        "intercepted_qber": intercepted["qber"],
        "intercepted_eavesdropping_detected": intercepted["eavesdropping_detected"],
        "final_key_length_bits_honest": honest["final_key_length_bits"],
        "qubits_transmitted_honest": honest["qubits_transmitted"],
        "qubits_transmitted_intercepted": intercepted["qubits_transmitted"],
        "message_round_trip_verified": decrypted == message,
        "ciphertext_bytes": len(ciphertext),
    }

    import json
    output_path = config.RESULTS_DIR / "qkd_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nMessage encrypted with QKD-derived key — round-trip verified: {decrypted == message}")
    print(f"Results written: {output_path}")

    return honest, intercepted, summary


if __name__ == "__main__":
    run_and_save()