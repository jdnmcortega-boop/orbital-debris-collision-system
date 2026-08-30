import numpy as np

from .orbital_mechanics import (
    position_vector,
    velocity_vector,
)


def collision_condition(
    relative_position,
    collision_radius_km
):
    """
    Return True when the relative position falls
    within the defined collision radius.
    """

    distance = np.linalg.norm(
        relative_position
    )

    return distance <= collision_radius_km


def generate_uncertain_state(
    row,
    rng,
    position_sigma_km=0.1,
    velocity_sigma_km_s=0.001
):
    """
    Generate one state from a simplified Gaussian
    uncertainty model.
    """

    position = position_vector(row)
    velocity = velocity_vector(row)

    noisy_position = (
        position
        + rng.normal(
            0,
            position_sigma_km,
            size=3
        )
    )

    noisy_velocity = (
        velocity
        + rng.normal(
            0,
            velocity_sigma_km_s,
            size=3
        )
    )

    return noisy_position, noisy_velocity


def sample_relative_state(
    row_a,
    row_b,
    rng,
    position_sigma_km=0.1,
    velocity_sigma_km_s=0.001
):
    """
    Generate uncertain relative position and velocity.
    """

    pos_a, vel_a = generate_uncertain_state(
        row_a,
        rng,
        position_sigma_km,
        velocity_sigma_km_s,
    )

    pos_b, vel_b = generate_uncertain_state(
        row_b,
        rng,
        position_sigma_km,
        velocity_sigma_km_s,
    )

    return (
        pos_a - pos_b,
        vel_a - vel_b,
    )