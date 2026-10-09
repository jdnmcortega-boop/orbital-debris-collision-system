"""
Deterministic sanity checks for the encounter-plane Monte Carlo estimator.
Run from the repository root:
    python -m modules.test_monte_carlo_sanity

These are model-level checks, not operational CARA validation.
"""
import numpy as np

from modules import monte_carlo
import config


def run_case(label, separation_km, combined_sigma_km, n_samples=100_000):
    # Set equal independent object sigmas so sqrt(sigma_a^2 + sigma_b^2)
    # equals the requested combined relative-position sigma.
    sigma_each = combined_sigma_km / np.sqrt(2.0)
    result = monte_carlo.estimate_collision_probability(
        pos_a=[7000.0, 0.0, 0.0],
        pos_b=[7000.0 + separation_km, 0.0, 0.0],
        sigma_a_km=sigma_each,
        sigma_b_km=sigma_each,
        n_samples=n_samples,
        hard_body_radius_km=config.HARD_BODY_RADIUS_KM,
        random_seed=12345,
    )
    probability, _hits, n, upper, lower, ess, log10p = result
    print(
        f"{label:40s} d={separation_km:8.4f} km "
        f"sigma={combined_sigma_km:7.4f} km "
        f"P={probability:.6e} log10(P)={log10p:9.3f} "
        f"CI=[{lower:.3e}, {upper:.3e}] ESS={ess:.0f}/{n}"
    )
    return probability, log10p


if __name__ == "__main__":
    radius = float(config.HARD_BODY_RADIUS_KM)
    print("Hard-body radius:", radius, "km")
    print("=" * 110)

    # A point well inside the collision disk with negligible uncertainty.
    p_inside, _ = run_case("Inside hard body, tiny uncertainty", 0.5 * radius, 1e-5)
    assert p_inside > 0.99, "Inside-disk limiting case should approach probability 1."

    # With zero nominal miss distance and a combined sigma equal to the
    # hard-body radius, probability must be positive and less than one.
    p_center, _ = run_case("Zero miss distance", 0.0, radius)
    assert 0.0 < p_center < 1.0, "Centered Gaussian disk probability should be between 0 and 1."

    # Increasing separation with fixed sigma must not increase probability.
    p_near, _ = run_case("Near conjunction", 0.05, 1.0)
    p_far, log_far = run_case("Farther conjunction", 9.07, 1.0)
    assert p_near > p_far, "Probability should decrease with increasing miss distance."

    # Far events may underflow as an ordinary float; log10(P) should still
    # expose the scale when it is finite.
    p_very_far, log_very_far = run_case("Rare-event diagnostic", 68.0, 1.0)
    assert p_very_far == 0.0 and np.isfinite(log_very_far), (
        "Expected ordinary probability underflow but a finite log-probability diagnostic."
    )

    print("=" * 110)
    print("All Monte Carlo sanity checks passed.")
