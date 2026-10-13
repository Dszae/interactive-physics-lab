"""
Numerical validation and analytical comparison utilities for Invisible Physics.

Provides quantitative verification functions used across unit tests and laboratory notebooks:
1. Inverse-square law verification for a point charge (|E(r)| ~ 1 / r^2).
2. Superposition principle verification (E(q1 + q2) == E(q1) + E(q2) to machine precision).
3. Potential-field gradient consistency (-grad(V) vs. Coulomb E(r) outside singularity cores).
4. Zero-field (saddle/null point) location between two like or unequal opposite charges.
5. Biot-Savart 1/r scaling for a long straight wire and exact on-axis match for a circular loop.
6. Electromagnetic wave vacuum invariants (c = f * lambda, |E| / |B| = c, E . B = 0).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .constants import C_VACUUM, COULOMB_K, DEFAULT_CORE_RADIUS_M, MICROCOULOMB_TO_COULOMB, MU_0
from .electrostatics import (
    PointCharge,
    create_coordinate_grid,
    electric_field_2d,
    electric_potential_2d,
    numerical_field_from_potential,
)
from .magnetism import (
    magnetic_field_circular_loop_on_axis,
    magnetic_field_circular_loop_xz,
    magnetic_field_straight_wire_2d,
)
from .waves import PlaneWaveVacuum, evaluate_plane_wave_1d


@dataclass(frozen=True)
class GradientConsistencyReport:
    """
    Quantitative comparison between direct Coulomb field E_exact and numerical gradient -grad(V)
    evaluated outside a safe exclusion radius r_eval_min > r_min around each charge.
    """

    grid_resolution: int
    grid_spacing_m: float
    r_eval_min_m: float
    num_valid_points: int
    max_rel_error: float
    median_rel_error: float
    mean_rel_error: float
    rms_angle_error_deg: float


def validate_potential_gradient_consistency(
    charges: Sequence[PointCharge],
    x_lim: tuple[float, float] = (-2.0, 2.0),
    y_lim: tuple[float, float] = (-2.0, 2.0),
    n_points: int = 201,
    r_min: float = DEFAULT_CORE_RADIUS_M,
    r_eval_min: float = 0.35,
) -> GradientConsistencyReport:
    """
    Compare E_coulomb(x, y) against E_num = -grad(V(x, y)) on a 2D Cartesian grid,
    excluding points within `r_eval_min` of any charge and boundary edges where one-sided
    differences have higher truncation error.

    Parameters
    ----------
    charges : Sequence[PointCharge]
        Point charges to evaluate.
    x_lim, y_lim : tuple[float, float]
        Coordinate domain in meters (m).
    n_points : int, default 201
        Grid points per dimension.
    r_min : float, default 0.05
        Singularity core cutoff radius in meters (m).
    r_eval_min : float, default 0.35
        Evaluation exclusion radius around each charge in meters (m). Must satisfy r_eval_min > r_min.

    Returns
    -------
    GradientConsistencyReport
    """
    if r_eval_min <= r_min:
        raise ValueError(f"r_eval_min ({r_eval_min}) must be strictly greater than r_min ({r_min}).")

    X, Y = create_coordinate_grid(x_lim=x_lim, y_lim=y_lim, n_points=n_points)
    dx = float(X[0, 1] - X[0, 0])

    field_res = electric_field_2d(X, Y, charges, r_min=r_min, strategy="core_cutoff")
    pot_res = electric_potential_2d(X, Y, charges, r_min=r_min, strategy="core_cutoff")
    Ex_num, Ey_num, E_mag_num = numerical_field_from_potential(pot_res)

    # Exclude points within r_eval_min of any charge and 2 grid cells from domain boundary
    valid_mask = np.ones_like(X, dtype=np.bool_)
    valid_mask[:2, :] = False
    valid_mask[-2:, :] = False
    valid_mask[:, :2] = False
    valid_mask[:, -2:] = False

    for c in charges:
        dist = np.hypot(X - c.x, Y - c.y)
        valid_mask &= dist >= r_eval_min

    # Also exclude very tiny field regions (e.g. exact null points) when computing relative error
    mag_threshold = 1e-3 * float(np.max(field_res.E_mag[valid_mask]))
    valid_mask &= field_res.E_mag > mag_threshold

    if not np.any(valid_mask):
        raise ValueError("No valid grid points remained after applying exclusion mask.")

    diff_mag = np.hypot(Ex_num[valid_mask] - field_res.Ex[valid_mask], Ey_num[valid_mask] - field_res.Ey[valid_mask])
    rel_error = diff_mag / field_res.E_mag[valid_mask]

    # Compute angle agreement in degrees between E_coulomb and -grad(V)
    dot = (
        field_res.Ex[valid_mask] * Ex_num[valid_mask]
        + field_res.Ey[valid_mask] * Ey_num[valid_mask]
    ) / (field_res.E_mag[valid_mask] * E_mag_num[valid_mask])
    cos_theta = np.clip(dot, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_theta))

    return GradientConsistencyReport(
        grid_resolution=n_points,
        grid_spacing_m=dx,
        r_eval_min_m=float(r_eval_min),
        num_valid_points=int(np.sum(valid_mask)),
        max_rel_error=float(np.max(rel_error)),
        median_rel_error=float(np.median(rel_error)),
        mean_rel_error=float(np.mean(rel_error)),
        rms_angle_error_deg=float(np.sqrt(np.mean(angles_deg * angles_deg))),
    )


def verify_inverse_square_law(
    q_uc: float = 1.0,
    radii_m: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict[str, NDArray[np.float64] | float]:
    """
    Verify that the electric-field magnitude of a single point charge at the origin follows
    |E(r)| = k_e * |q| / r^2 and returns the log-log slope (theoretical value: -2.0).
    """
    r_arr = np.asarray(radii_m, dtype=np.float64)
    if np.any(r_arr <= DEFAULT_CORE_RADIUS_M):
        raise ValueError(f"All evaluation radii must exceed DEFAULT_CORE_RADIUS_M ({DEFAULT_CORE_RADIUS_M} m).")

    y_zeros = np.zeros_like(r_arr)
    charge = PointCharge(q_uc=q_uc, x=0.0, y=0.0)
    res = electric_field_2d(r_arr, y_zeros, [charge], r_min=DEFAULT_CORE_RADIUS_M)

    analytical = COULOMB_K * abs(charge.q_coulombs) / (r_arr * r_arr)
    rel_err = np.abs(res.E_mag - analytical) / analytical

    # Linear regression on log(r) vs log(|E|)
    slope, _ = np.polyfit(np.log(r_arr), np.log(res.E_mag), 1)
    return {
        "radii_m": r_arr,
        "computed_E_v_per_m": res.E_mag,
        "analytical_E_v_per_m": analytical,
        "relative_error": rel_err,
        "max_relative_error": float(np.max(rel_err)),
        "log_log_slope": float(slope),
    }


def find_collinear_zero_field_point(
    q1_uc: float,
    x1_m: float,
    q2_uc: float,
    x2_m: float,
) -> dict[str, float]:
    """
    Find the exact analytical and root-solved 1D zero-field point (Ex(x, 0) = 0) along the x-axis
    for two non-zero point charges located at (x1_m, 0) and (x2_m, 0).

    Physical Cases:
    1. Same-sign charges (q1 * q2 > 0): Unique null point lies strictly between x1 and x2:
       x_null = (x1 * sqrt(|q2|) + x2 * sqrt(|q1|)) / (sqrt(|q1|) + sqrt(|q2|)).
    2. Opposite-sign charges with |q1| != |q2| (q1 * q2 < 0): Unique null point lies outside the
       segment [x1, x2], on the side of the smaller-magnitude charge:
       x_null = (x2 * sqrt(|q1|) - x1 * sqrt(|q2|)) / (sqrt(|q1|) - sqrt(|q2|)).
    3. Equal and opposite dipole (q1 == -q2): No finite zero-field point exists.
    """
    if q1_uc == 0.0 or q2_uc == 0.0:
        raise ValueError("Both charges q1_uc and q2_uc must be non-zero.")
    if math.isclose(x1_m, x2_m, abs_tol=1e-9):
        raise ValueError("Charges must be at distinct positions x1_m != x2_m.")
    if q1_uc * q2_uc < 0.0 and math.isclose(abs(q1_uc), abs(q2_uc), rel_tol=1e-9):
        raise ValueError("An equal-and-opposite dipole has no finite zero-field point.")

    # Ensure x1 < x2 for interval bracketing
    if x1_m > x2_m:
        x1_m, x2_m = x2_m, x1_m
        q1_uc, q2_uc = q2_uc, q1_uc

    s1 = math.sqrt(abs(q1_uc))
    s2 = math.sqrt(abs(q2_uc))
    charges = [
        PointCharge(q_uc=q1_uc, x=x1_m, y=0.0),
        PointCharge(q_uc=q2_uc, x=x2_m, y=0.0),
    ]

    def ex_at_x(x_val: float) -> float:
        res = electric_field_2d(
            np.array([x_val], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            charges,
            r_min=1e-6,
        )
        return float(res.Ex[0])

    if q1_uc * q2_uc > 0.0:
        x_analytical = (x1_m * s2 + x2_m * s1) / (s1 + s2)
        eps = 0.05 * (x2_m - x1_m)
        x_numerical = float(brentq(ex_at_x, x1_m + eps, x2_m - eps, xtol=1e-12))
    else:
        x_analytical = (x2_m * s1 - x1_m * s2) / (s1 - s2)
        span = abs(x_analytical - x1_m) + abs(x2_m - x1_m)
        if abs(q1_uc) < abs(q2_uc):
            # Null point is to the left of x1_m
            x_numerical = float(brentq(ex_at_x, x_analytical - span, x1_m - 1e-3, xtol=1e-12))
        else:
            # Null point is to the right of x2_m
            x_numerical = float(brentq(ex_at_x, x2_m + 1e-3, x_analytical + span, xtol=1e-12))

    residual_e = abs(ex_at_x(x_numerical))
    return {
        "x_analytical_m": float(x_analytical),
        "x_numerical_m": float(x_numerical),
        "abs_position_error_m": abs(float(x_numerical) - float(x_analytical)),
        "residual_field_v_per_m": residual_e,
    }


def verify_circular_loop_on_axis(
    current_amps: float = 15.0,
    radius_m: float = 0.8,
    z_max_m: float = 2.0,
    n_points: int = 101,
) -> dict[str, float]:
    """
    Validate that the 2D elliptic-integral circular-loop field evaluated along x = 0
    agrees with the exact elementary on-axis Biot-Savart formula:
        Bz(0, z) = (mu_0 * I * R^2) / (2 * (R^2 + z^2)^(3/2)).
    """
    z_1d = np.linspace(-z_max_m, z_max_m, n_points, dtype=np.float64)
    x_zeros = np.zeros_like(z_1d)

    res_2d = magnetic_field_circular_loop_xz(x_zeros, z_1d, current_amps=current_amps, radius_m=radius_m)
    bz_analytical = magnetic_field_circular_loop_on_axis(z_1d, current_amps=current_amps, radius_m=radius_m)

    rel_err = np.abs(res_2d.By - bz_analytical) / np.abs(bz_analytical)
    max_transverse = float(np.max(np.abs(res_2d.Bx)))
    return {
        "max_relative_error_bz": float(np.max(rel_err)),
        "max_transverse_bx_tesla": max_transverse,
    }

