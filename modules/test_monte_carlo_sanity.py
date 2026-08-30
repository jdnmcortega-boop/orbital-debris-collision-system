"""
Sanity check for monte_carlo.py's core math — NOT part of the main pipeline.
Run this standalone to confirm estimate_collision_probability() actually
produces nonzero probabilities when objects are genuinely close, before
trusting the all-zero results on your real 9-67 km conjunction data.
"""

import numpy as np

from modules import monte_carlo
import config


def run_case(label, pos_a, pos_b, sigma_km, n_samples=100_000):
    prob, hits, n = monte_carlo.estimate_collision_probability(
        pos_a, pos_b,
        sigma_km=sigma_km,
        n_samples=n_samples,
        hard_body_radius_km=config.HARD_BODY_RADIUS_KM,
    )
    print(f"{label:45s} sigma={sigma_km:>5.3f} km  "
          f"P={prob:.6f}  ({hits}/{n} samples)")


if __name__ == "__main__":
    print("Hard body radius:", config.HARD_BODY_RADIUS_KM, "km")
    print("=" * 80)

    # Case 1: objects at the SAME point (nominal miss distance = 0).
    # With zero separation and any nonzero uncertainty, roughly half the
    # random noise combinations should land within the hard-body radius
    # for small enough sigma. This is the strongest possible sanity check —
    # if this comes back 0, something is actually broken in the code.
    run_case(
        "Same position (0 km apart)",
        pos_a=[7000.0, 0.0, 0.0],
        pos_b=[7000.0, 0.0, 0.0],
        sigma_km=0.02,
    )

    # Case 2: objects 15 m apart — well inside a 20 m hard-body radius
    # even with zero noise. Should return a probability close to 1.0
    # at very small sigma, and still clearly nonzero at larger sigma.
    run_case(
        "15 m apart (inside hard-body radius)",
        pos_a=[7000.0, 0.0, 0.0],
        pos_b=[7000.015, 0.0, 0.0],
        sigma_km=0.02,
    )

    # Case 3: objects 9.07 km apart — YOUR closest real conjunction
    # (COSMOS 2251 DEB vs LUSAT), at your actual uncertainty setting.
    run_case(
        "9.07 km apart (your real closest pair)",
        pos_a=[7000.0, 0.0, 0.0],
        pos_b=[7009.074414, 0.0, 0.0],
        sigma_km=getattr(config, "POSITION_UNCERTAINTY_KM", 1.0),
    )

    # Case 4: same 9.07 km separation, but with a much larger uncertainty
    # (5 km) — shows the probability rising as position uncertainty grows,
    # which is the expected physical behavior (sloppier tracking = more
    # spread out samples = higher chance some land close together).
    run_case(
        "9.07 km apart, larger uncertainty",
        pos_a=[7000.0, 0.0, 0.0],
        pos_b=[7009.074414, 0.0, 0.0],
        sigma_km=5.0,
    )

    print("=" * 80)
    print("If Cases 1 and 2 return ~0 as well, the bug is in the code.")
    print("If Cases 1 and 2 return nonzero but Cases 3 and 4 stay near 0,")
    print("the code is correct and your real conjunctions genuinely have")
    print("probabilities far below what 10,000-100,000 samples can resolve.")