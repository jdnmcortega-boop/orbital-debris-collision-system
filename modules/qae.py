"""
Quantum Amplitude Estimation (QAE) for collision-probability estimation.

Provides:
    1. Analytic collision-probability calculation
    2. QPE-based Quantum Amplitude Estimation
    3. Matched-budget classical Monte Carlo
    4. QAE vs Monte Carlo comparison
    5. Query-budget accuracy sweep

The physical collision probability is calculated classically and then
encoded into a quantum amplitude. QAE estimates that amplitude using
a simulated quantum circuit.

The QAE readout uses a maximum-likelihood estimator over the exact
quantum-phase-estimation measurement model. This replaces the previous
"weighted average of encoded amplitudes" readout, which is biased for
very small probabilities.
"""

import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import ncx2

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit.circuit.library import QFTGate, grover_operator
from qiskit_aer import AerSimulator

import config


# ============================================================
# ANALYTIC COLLISION PROBABILITY
# ============================================================

def analytic_collision_probability(
    miss_distance_km,
    sigma_km=None,
    hard_body_radius_km=None,
    sigma_a_km=None,
    sigma_b_km=None,
):
    if hard_body_radius_km is None:
        hard_body_radius_km = getattr(
            config,
            "HARD_BODY_RADIUS_KM",
            0.02,
        )

    if sigma_a_km is not None and sigma_b_km is not None:
        combined_sigma = np.sqrt(
            sigma_a_km ** 2 +
            sigma_b_km ** 2
        )
    else:
        if sigma_km is None:
            sigma_km = getattr(
                config,
                "POSITION_UNCERTAINTY_KM",
                1.0,
            )

        combined_sigma = np.sqrt(2.0) * sigma_km

    if combined_sigma <= 0:
        return 0.0

    miss_distance_km = abs(float(miss_distance_km))
    hard_body_radius_km = float(hard_body_radius_km)

    if hard_body_radius_km <= 0:
        return 0.0

    nc = (
        miss_distance_km /
        combined_sigma
    ) ** 2

    x = (
        hard_body_radius_km /
        combined_sigma
    ) ** 2

    probability = ncx2.cdf(
        x,
        df=2,
        nc=nc,
    )

    return float(
        np.clip(probability, 0.0, 1.0)
    )


# ============================================================
# STATE PREPARATION
# ============================================================

def build_state_preparation(theta):
    qc = QuantumCircuit(1, name="A")
    qc.ry(theta, 0)
    return qc


# ============================================================
# ORACLE
# ============================================================

def build_oracle():
    qc = QuantumCircuit(1, name="S_f")
    qc.z(0)
    return qc


# ============================================================
# GROVER OPERATOR
# ============================================================

def build_grover_operator(theta):
    A = build_state_preparation(theta)
    oracle = build_oracle()

    return grover_operator(
        oracle=oracle,
        state_preparation=A,
        reflection_qubits=[0],
    )


# ============================================================
# EXACT QPE MEASUREMENT MODEL
# ============================================================

def _qpe_phase_distribution(phi, num_eval_qubits):
    """Return the ideal QPE outcome probabilities for phase ``phi``.

    For the Grover operator used by amplitude estimation, the initial state
    is an equal superposition of the two eigenphases +2*theta and -2*theta.
    With phi = theta/pi, the two corresponding normalized QPE phases are
    ``phi`` and ``1 - phi``.

    The probability for a single QPE outcome y is the exact finite-register
    Dirichlet-kernel expression. ``np.sinc`` is used for numerical stability
    near zero.
    """
    m = int(num_eval_qubits)
    if m < 1:
        raise ValueError("num_eval_qubits must be >= 1")

    M = 2 ** m
    y = np.arange(M, dtype=float)

    def one_phase_probability(phase):
        delta = phase - y / M
        # sin(pi*M*delta)/(M*sin(pi*delta))
        # = sinc(M*delta) / sinc(delta)
        numerator = np.sinc(M * delta)
        denominator = np.sinc(delta)
        values = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=np.abs(denominator) > 1e-15,
        )
        return values ** 2

    distribution = 0.5 * (
        one_phase_probability(phi)
        + one_phase_probability(1.0 - phi)
    )

    total = float(np.sum(distribution))
    if total <= 0.0:
        raise RuntimeError("Invalid QPE probability distribution")

    return distribution / total


def _qae_counts_log_likelihood(phi, counts, num_eval_qubits):
    """Compute the multinomial log-likelihood for a QPE phase candidate."""
    probabilities = _qpe_phase_distribution(
        phi,
        num_eval_qubits,
    )

    # Counts are stored by the measured binary string.
    likelihood = 0.0
    floor = 1e-300

    for bitstring, count in counts.items():
        y = int(bitstring, 2)
        if y >= len(probabilities):
            return -np.inf
        likelihood += int(count) * math.log(
            max(float(probabilities[y]), floor)
        )

    return float(likelihood)


def _qae_mle_phase(counts, num_eval_qubits):
    """Estimate the Grover phase using maximum likelihood.

    A logarithmic low-phase search is included because orbital collision
    probabilities often live around 1e-12 to 1e-6, where a purely uniform
    coarse grid would waste almost all of its resolution.
    """
    m = int(num_eval_qubits)

    # Build a search grid concentrated near zero while retaining full-range
    # coverage. The finite QPE register makes this cheap enough for the small
    # evaluation registers used in this project.
    low = np.geomspace(
        1e-10,
        0.01,
        600,
    )
    linear = np.linspace(
        0.0,
        0.5,
        1200,
    )
    candidates = np.unique(
        np.clip(
            np.concatenate(([0.0], low, linear)),
            0.0,
            0.5,
        )
    )

    log_likelihoods = np.array([
        _qae_counts_log_likelihood(phi, counts, m)
        for phi in candidates
    ])

    best_index = int(np.argmax(log_likelihoods))
    best_phi = float(candidates[best_index])
    best_log_likelihood = float(log_likelihoods[best_index])

    # Refine the best candidate locally when it is not already at a boundary.
    left_index = max(0, best_index - 1)
    right_index = min(len(candidates) - 1, best_index + 1)

    left = float(candidates[left_index])
    right = float(candidates[right_index])

    if right > left:
        result = minimize_scalar(
            lambda value: -_qae_counts_log_likelihood(
                float(value),
                counts,
                m,
            ),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-12},
        )

        if result.success:
            refined_phi = float(
                np.clip(result.x, 0.0, 0.5)
            )
            refined_log_likelihood = float(-result.fun)

            if refined_log_likelihood >= best_log_likelihood:
                best_phi = refined_phi

    return float(
        np.clip(best_phi, 0.0, 0.5)
    )


# ============================================================
# QAE
# ============================================================

def run_qae(
    true_probability,
    num_eval_qubits=6,
    shots=200,
):
    """
    Run QPE-based QAE.

    The query budget is:

        shots * (2^m - 1)

    where m is the number of evaluation qubits.
    """
    true_probability = float(
        np.clip(
            true_probability,
            0.0,
            1.0,
        )
    )

    m = int(num_eval_qubits)
    shots = int(shots)

    if m < 1:
        raise ValueError("num_eval_qubits must be >= 1.")
    if shots < 1:
        raise ValueError("shots must be >= 1.")

    # p = sin²(theta / 2)
    theta = (
        2.0 *
        np.arcsin(
            np.sqrt(true_probability)
        )
    )

    A = build_state_preparation(theta)
    grover_op = build_grover_operator(theta)
    Q_gate = grover_op.to_gate()
    Q_gate.name = "Q"

    eval_reg = QuantumRegister(m, name="eval")
    state_reg = QuantumRegister(1, name="state")
    creg = ClassicalRegister(m, name="c")

    qc = QuantumCircuit(
        eval_reg,
        state_reg,
        creg,
    )

    qc.append(
        A.to_gate(),
        [state_reg[0]],
    )

    qc.h(eval_reg)

    for k in range(m):
        controlled_Q = (
            Q_gate
            .power(2 ** k)
            .control(1)
        )

        qc.append(
            controlled_Q,
            [
                eval_reg[k],
                state_reg[0],
            ],
        )

    qc.append(
        QFTGate(m).inverse(),
        eval_reg,
    )

    qc.measure(
        eval_reg,
        creg,
    )

    simulator = AerSimulator()
    transpiled = transpile(qc, simulator)

    start = time.time()

    result = simulator.run(
        transpiled,
        shots=shots,
    ).result()

    runtime = time.time() - start
    counts = result.get_counts()
    total_shots = sum(counts.values())

    if total_shots <= 0:
        raise RuntimeError("QAE circuit returned zero measurement shots")

    mle_phi = _qae_mle_phase(
        counts,
        num_eval_qubits=m,
    )

    # phi = theta/pi and p = sin²(theta/2)
    mle_theta = np.pi * mle_phi
    estimate = np.sin(mle_theta / 2.0) ** 2

    estimate = float(
        np.clip(
            estimate,
            0.0,
            1.0,
        )
    )

    error = abs(
        estimate -
        true_probability
    )

    oracle_calls = (
        shots *
        ((2 ** m) - 1)
    )

    return {
        "true_probability": true_probability,
        "qae_estimate": estimate,
        "qae_error": float(error),
        "oracle_calls": int(oracle_calls),
        "runtime_sec": float(runtime),
        "eval_qubits": int(m),
        "shots": int(shots),
        "estimator": "QPE_MLE",
        "measurement_outcomes": int(len(counts)),
    }


# ============================================================
# MONTE CARLO
# ============================================================

def _wilson_interval(hits, n_samples, z=1.96):
    """Wilson confidence interval for a Bernoulli probability estimate."""
    n = int(n_samples)
    x = int(hits)

    if n <= 0:
        raise ValueError("n_samples must be positive")

    phat = x / n
    z2 = z ** 2
    denominator = 1.0 + z2 / n
    center = (
        phat +
        z2 / (2.0 * n)
    ) / denominator
    margin = (
        z *
        np.sqrt(
            (
                phat * (1.0 - phat) /
                n
            )
            +
            z2 / (4.0 * n ** 2)
        ) /
        denominator
    )

    return (
        float(max(0.0, center - margin)),
        float(min(1.0, center + margin)),
    )


def run_classical_mc(
    true_probability,
    n_samples,
    seed=None,
):
    true_probability = float(
        np.clip(
            true_probability,
            0.0,
            1.0,
        )
    )

    n_samples = int(n_samples)

    if n_samples < 1:
        raise ValueError(
            "n_samples must be >= 1."
        )

    rng = np.random.default_rng(seed)

    start = time.time()

    hits = int(
        rng.binomial(
            n_samples,
            true_probability,
        )
    )

    runtime = time.time() - start

    estimate = hits / n_samples

    error = abs(
        estimate -
        true_probability
    )

    ci_low, ci_high = _wilson_interval(
        hits,
        n_samples,
    )

    return {
        "true_probability": true_probability,
        "mc_estimate": float(estimate),
        "mc_error": float(error),
        "n_samples": int(n_samples),
        "runtime_sec": float(runtime),
        "hits": int(hits),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ============================================================
# QAE VS MONTE CARLO
# ============================================================

def compare_methods(
    miss_distance_km,
    sigma_km=None,
    hard_body_radius_km=None,
    num_eval_qubits=6,
    shots=200,
):
    true_p = analytic_collision_probability(
        miss_distance_km,
        sigma_km,
        hard_body_radius_km,
    )

    qae_result = run_qae(
        true_p,
        num_eval_qubits=num_eval_qubits,
        shots=shots,
    )

    mc_result = run_classical_mc(
        true_p,
        n_samples=qae_result["oracle_calls"],
    )

    return {
        "MISS_DISTANCE_KM": miss_distance_km,
        "ANALYTIC_PC": true_p,
        "QAE_ESTIMATE": qae_result["qae_estimate"],
        "QAE_ERROR": qae_result["qae_error"],
        "QAE_ORACLE_CALLS": qae_result["oracle_calls"],
        "QAE_RUNTIME_SEC": qae_result["runtime_sec"],
        "QAE_EVAL_QUBITS": qae_result["eval_qubits"],
        "QAE_ESTIMATOR": qae_result["estimator"],
        "MC_ESTIMATE": mc_result["mc_estimate"],
        "MC_ERROR": mc_result["mc_error"],
        "MC_SAMPLES": mc_result["n_samples"],
        "MC_RUNTIME_SEC": mc_result["runtime_sec"],
        "MC_HITS": mc_result["hits"],
        "MC_CI_LOW": mc_result["ci_low"],
        "MC_CI_HIGH": mc_result["ci_high"],
    }


# ============================================================
# QAE VS MC PIPELINE
# ============================================================

def run_and_save(
    num_eval_qubits=6,
    shots=200,
):
    config.ensure_dirs()

    conj_path = config.CONJUNCTIONS_FILE

    if not conj_path.exists():
        print(
            f"No conjunctions file found at "
            f"{conj_path}. Run conjunction_detection first."
        )
        return None

    conjunctions = pd.read_csv(conj_path)

    if conjunctions.empty:
        print("Conjunctions file is empty.")
        return None

    rows = []

    for _, row in conjunctions.iterrows():
        result = compare_methods(
            row["MISS_DISTANCE_KM"],
            num_eval_qubits=num_eval_qubits,
            shots=shots,
        )

        result["OBJECT_A"] = row["OBJECT_A"]
        result["OBJECT_B"] = row["OBJECT_B"]

        rows.append(result)

    output = pd.DataFrame(rows)

    output_path = (
        config.RESULTS_DIR /
        "qae_comparison.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults written: "
        f"{output_path}"
    )

    print(
        output[
            [
                "OBJECT_A",
                "OBJECT_B",
                "ANALYTIC_PC",
                "QAE_ESTIMATE",
                "QAE_ERROR",
                "MC_ESTIMATE",
                "MC_ERROR",
                "MC_HITS",
                "MC_CI_LOW",
                "MC_CI_HIGH",
                "QAE_ORACLE_CALLS",
                "MC_SAMPLES",
            ]
        ].to_string(index=False)
    )

    return output


# ============================================================
# ACCURACY SWEEP
# ============================================================

def run_accuracy_sweep(
    probabilities=None,
    eval_qubits=None,
    shots=100,
    n_trials=10,
):
    if probabilities is None:
        probabilities = [
            0.500,
            0.100,
            0.050,
            0.010,
            0.005,
            0.001,
        ]

    if eval_qubits is None:
        eval_qubits = [
            3,
            5,
            7,
            9,
            11,
            13,
            15,
            17,
            19,
        ]

    rows = []

    for probability in probabilities:
        for m in eval_qubits:
            qae_errors = []
            mc_errors = []

            oracle_calls = (
                shots *
                ((2 ** m) - 1)
            )

            for trial in range(n_trials):
                qae = run_qae(
                    probability,
                    num_eval_qubits=m,
                    shots=shots,
                )

                mc = run_classical_mc(
                    probability,
                    n_samples=oracle_calls,
                    seed=trial,
                )

                qae_errors.append(
                    qae["qae_error"]
                )

                mc_errors.append(
                    mc["mc_error"]
                )

            qae_error_mean = float(
                np.mean(qae_errors)
            )

            mc_error_mean = float(
                np.mean(mc_errors)
            )

            rows.append(
                {
                    "TRUE_PROBABILITY": probability,
                    "EVAL_QUBITS": m,
                    "ORACLE_CALLS": oracle_calls,
                    "N_TRIALS": n_trials,
                    "QAE_ERROR_MEAN": qae_error_mean,
                    "MC_ERROR_MEAN": mc_error_mean,
                    "QAE_WINS": qae_error_mean < mc_error_mean,
                }
            )

    output = pd.DataFrame(rows)

    output_path = (
        config.RESULTS_DIR /
        "qae_accuracy_sweep.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nAccuracy sweep written: "
        f"{output_path}"
    )

    return output


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_and_save()
    run_accuracy_sweep()
