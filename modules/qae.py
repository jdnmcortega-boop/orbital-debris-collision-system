"""
Quantum Amplitude Estimation (QAE) for collision-probability estimation,
compared against classical Monte Carlo.

IMPORTANT SCOPE NOTE (state this in your methodology/report):
A full quantum oracle that computes the physical collision condition from
superposed orbital position states would require quantum arithmetic on
continuous-valued data — well beyond the scope of this project. Instead,
the *analytic* collision probability (a standard closed-form Gaussian-
overlap formula used in real conjunction assessment) is computed
classically, then encoded into a single-qubit rotation as the amplitude
QAE is asked to estimate. This faithfully implements and benchmarks the
QAE *algorithm* (state preparation, Grover/Q operator, phase estimation,
measurement, and its quadratic query-complexity advantage over classical
sampling) but is NOT a physical quantum collision-detection oracle.
"""

import time

import numpy as np
from scipy.stats import ncx2

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFTGate, grover_operator
from qiskit_aer import AerSimulator

import config


# ============================================================
# ANALYTIC "GROUND TRUTH" COLLISION PROBABILITY
# ============================================================
# Standard 2D Gaussian-overlap conjunction-probability formula (Foster-
# style), simplified to the isotropic case already used in monte_carlo.py:
# both objects have independent isotropic position uncertainty sigma_km,
# so their relative position has isotropic uncertainty of sqrt(2)*sigma_km.
# The probability the relative position falls inside a circle of radius
# hard_body_radius_km, given nominal separation miss_distance_km, follows
# a non-central chi-square distribution with 2 degrees of freedom.

def analytic_collision_probability(miss_distance_km, sigma_km=None, hard_body_radius_km=None,
                                     sigma_a_km=None, sigma_b_km=None):
    """
    If sigma_a_km/sigma_b_km are given (per-object, e.g. age-scaled via
    uncertainty_model.py), the combined uncertainty uses them directly.
    Otherwise falls back to the original symmetric-sigma behavior
    (sigma_km applied to both objects, combined = sqrt(2)*sigma_km) for
    backward compatibility with existing callers.
    """
    hard_body_radius_km = hard_body_radius_km or config.HARD_BODY_RADIUS_KM

    if sigma_a_km is not None and sigma_b_km is not None:
        combined_sigma = np.sqrt(sigma_a_km ** 2 + sigma_b_km ** 2)
    else:
        sigma_km = sigma_km or getattr(config, "POSITION_UNCERTAINTY_KM", 1.0)
        combined_sigma = np.sqrt(2) * sigma_km

    if combined_sigma <= 0:
        return 0.0

    nc = (miss_distance_km / combined_sigma) ** 2
    x = (hard_body_radius_km / combined_sigma) ** 2

    return float(ncx2.cdf(x, df=2, nc=nc))


# ============================================================
# QAE CIRCUIT CONSTRUCTION
# ============================================================

def build_state_preparation(theta):
    """A: single-qubit state sqrt(1-a)|0> + sqrt(a)|1>, a = sin^2(theta/2)."""
    qc = QuantumCircuit(1, name="A")
    qc.ry(theta, 0)
    return qc


def build_oracle():
    """S_f: flips the sign of the 'good' (collision) state |1>."""
    qc = QuantumCircuit(1, name="S_f")
    qc.z(0)
    return qc


def build_grover_operator(theta):
    A = build_state_preparation(theta)
    oracle = build_oracle()
    return grover_operator(oracle=oracle, state_preparation=A, reflection_qubits=[0])


# ============================================================
# CANONICAL (QPE-BASED) AMPLITUDE ESTIMATION
# ============================================================

def run_qae(true_probability, num_eval_qubits=6, shots=200):
    """
    Encode `true_probability` as a quantum amplitude and estimate it via
    canonical QPE-based amplitude estimation.

    Returns dict with estimate, absolute error, oracle call count, runtime.
    """
    true_probability = max(0.0, min(1.0, true_probability))
    theta = 2 * np.arcsin(np.sqrt(true_probability))

    A = build_state_preparation(theta)
    grover_op = build_grover_operator(theta)
    Q_gate = grover_op.to_gate()
    Q_gate.name = "Q"

    m = num_eval_qubits
    eval_reg = QuantumRegister(m, name="eval")
    state_reg = QuantumRegister(1, name="state")
    creg = ClassicalRegister(m, name="c")
    qc = QuantumCircuit(eval_reg, state_reg, creg)

    # Prepare A|0> on the state register
    qc.append(A.to_gate(), [state_reg[0]])

    # Hadamards on evaluation register
    qc.h(eval_reg)

    # Controlled Q^(2^k) for each evaluation qubit
    for k in range(m):
        controlled_Qk = Q_gate.power(2 ** k).control(1)
        qc.append(controlled_Qk, [eval_reg[k], state_reg[0]])

    # Inverse QFT on evaluation register, then measure
    qc.append(QFTGate(m).inverse(), eval_reg)
    qc.measure(eval_reg, creg)

    simulator = AerSimulator()
    transpiled = transpile(qc, simulator)

    start = time.time()
    result = simulator.run(transpiled, shots=shots).result()
    runtime = time.time() - start

    counts = result.get_counts()

    # Weighted average estimate across all measured outcomes
    total_shots = sum(counts.values())
    weighted_estimate = 0.0
    for bitstring, count in counts.items():
        y = int(bitstring, 2)
        a_est = np.sin(np.pi * y / (2 ** m)) ** 2
        weighted_estimate += a_est * (count / total_shots)

    error = abs(weighted_estimate - true_probability)
    oracle_calls = shots * (2 ** m - 1)  # total Q applications across all shots

    return {
        "true_probability": true_probability,
        "qae_estimate": weighted_estimate,
        "qae_error": error,
        "oracle_calls": oracle_calls,
        "runtime_sec": runtime,
        "eval_qubits": m,
        "shots": shots,
    }


# ============================================================
# MATCHED-BUDGET CLASSICAL MONTE CARLO (FOR FAIR COMPARISON)
# ============================================================

def run_classical_mc(true_probability, n_samples):
    """Bernoulli sampling at a known true probability, for benchmarking only."""
    start = time.time()
    hits = np.random.binomial(1, true_probability, size=n_samples).sum()
    runtime = time.time() - start

    estimate = hits / n_samples
    error = abs(estimate - true_probability)

    return {
        "true_probability": true_probability,
        "mc_estimate": estimate,
        "mc_error": error,
        "n_samples": n_samples,
        "runtime_sec": runtime,
    }


# ============================================================
# COMPARISON PIPELINE
# ============================================================

def compare_methods(miss_distance_km, sigma_km=None, hard_body_radius_km=None,
                     num_eval_qubits=6, shots=200):
    true_p = analytic_collision_probability(miss_distance_km, sigma_km, hard_body_radius_km)

    qae_result = run_qae(true_p, num_eval_qubits=num_eval_qubits, shots=shots)
    mc_result = run_classical_mc(true_p, n_samples=qae_result["oracle_calls"])

    return {
        "MISS_DISTANCE_KM": miss_distance_km,
        "ANALYTIC_PC": true_p,
        "QAE_ESTIMATE": qae_result["qae_estimate"],
        "QAE_ERROR": qae_result["qae_error"],
        "QAE_ORACLE_CALLS": qae_result["oracle_calls"],
        "QAE_RUNTIME_SEC": qae_result["runtime_sec"],
        "MC_ESTIMATE": mc_result["mc_estimate"],
        "MC_ERROR": mc_result["mc_error"],
        "MC_SAMPLES": mc_result["n_samples"],
        "MC_RUNTIME_SEC": mc_result["runtime_sec"],
    }


def run_and_save(num_eval_qubits=6, shots=200):
    import pandas as pd

    config.ensure_dirs()

    conj_path = config.CONJUNCTIONS_FILE
    if not conj_path.exists():
        print(f"No conjunctions file found at {conj_path}. Run conjunction_detection first.")
        return None

    conjunctions = pd.read_csv(conj_path)
    if len(conjunctions) == 0:
        print("Conjunctions file is empty — nothing to compare.")
        return None

    rows = []
    for _, row in conjunctions.iterrows():
        print(f"[QAE] {row['OBJECT_A']} vs {row['OBJECT_B']} "
              f"(miss={row['MISS_DISTANCE_KM']:.3f} km)...")
        result = compare_methods(
            row["MISS_DISTANCE_KM"],
            num_eval_qubits=num_eval_qubits,
            shots=shots,
        )
        result["OBJECT_A"] = row["OBJECT_A"]
        result["OBJECT_B"] = row["OBJECT_B"]
        rows.append(result)

    output = pd.DataFrame(rows)
    output_path = config.RESULTS_DIR / "qae_comparison.csv"
    output.to_csv(output_path, index=False)

    print(f"\nResults written: {output_path}")
    print(output[["OBJECT_A", "OBJECT_B", "ANALYTIC_PC",
                   "QAE_ESTIMATE", "QAE_ERROR", "MC_ESTIMATE", "MC_ERROR"]
                  ].to_string(index=False))

    return output


if __name__ == "__main__":
    run_and_save()