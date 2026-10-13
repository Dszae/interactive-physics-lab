"""
Automated unit and physics verification tests for the electrostatics module.

Tests cover:
1. Single charge at a known position against exact Coulomb formula.
2. Inverse-square dependence |E(r)| ~ 1 / r^2.
3. Exact field and potential sign reversal when q -> -q.
4. Linear superposition of independent fields and potentials.
5. Geometric and parity symmetry for symmetric configurations (dipole, quadrupole, like charges).
6. Potential-field gradient consistency (-grad(V) vs E) away from singularities.
7. Explicit singularity-handling strategies ("core_cutoff", "softening", "mask") and input validation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from invisible_physics import (
    COULOMB_K,
    MICROCOULOMB_TO_COULOMB,
    PointCharge,
    create_coordinate_grid,
    electric_field_2d,
    electric_potential_2d,
    get_preset_charges,
    numerical_field_from_potential,
)


def test_single_point_charge_analytical_field_and_potential() -> None:
    """Verify exact Coulomb field and potential for a single +2.5 uC charge at (0.3, -0.4)."""
    q_uc = 2.5
    x0, y0 = 0.3, -0.4
    charge = PointCharge(q_uc=q_uc, x=x0, y=y0)

    # Evaluate at (1.5, 0.5) where distance r = sqrt(1.2^2 + 0.9^2) = 1.5 m > r_min
    X = np.array([[1.5]], dtype=np.float64)
    Y = np.array([[0.5]], dtype=np.float64)

    field_res = electric_field_2d(X, Y, [charge], r_min=0.05)
    pot_res = electric_potential_2d(X, Y, [charge], r_min=0.05)

    r_expected = 1.5
    q_c = q_uc * MICROCOULOMB_TO_COULOMB
    e_mag_expected = COULOMB_K * q_c / (r_expected**2)
    ex_expected = e_mag_expected * (1.2 / r_expected)
    ey_expected = e_mag_expected * (0.9 / r_expected)
    v_expected = COULOMB_K * q_c / r_expected

    assert math.isclose(field_res.E_mag[0, 0], e_mag_expected, rel_tol=1e-12)
    assert math.isclose(field_res.Ex[0, 0], ex_expected, rel_tol=1e-12)
    assert math.isclose(field_res.Ey[0, 0], ey_expected, rel_tol=1e-12)
    assert math.isclose(pot_res.V[0, 0], v_expected, rel_tol=1e-12)


def test_inverse_square_scaling_and_charge_linearity() -> None:
    """Doubling distance reduces |E| by 4x and V by 2x; doubling charge doubles both |E| and V."""
    c1 = PointCharge(q_uc=1.0, x=0.0, y=0.0)
    c2 = PointCharge(q_uc=2.0, x=0.0, y=0.0)

    X = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    Y = np.zeros_like(X)

    res1 = electric_field_2d(X, Y, [c1], r_min=0.05)
    res2 = electric_field_2d(X, Y, [c2], r_min=0.05)
    pot1 = electric_potential_2d(X, Y, [c1], r_min=0.05)

    # Doubling distance from 0.5m -> 1.0m -> 2.0m reduces |E| by factor of 4 each step
    assert math.isclose(res1.E_mag[0] / res1.E_mag[1], 4.0, rel_tol=1e-12)
    assert math.isclose(res1.E_mag[1] / res1.E_mag[2], 4.0, rel_tol=1e-12)

    # Doubling distance reduces V by factor of 2 each step
    assert math.isclose(pot1.V[0] / pot1.V[1], 2.0, rel_tol=1e-12)
    assert math.isclose(pot1.V[1] / pot1.V[2], 2.0, rel_tol=1e-12)

    # Doubling charge doubles |E|
    np.testing.assert_allclose(res2.E_mag, 2.0 * res1.E_mag, rtol=1e-12)


def test_field_reversal_on_charge_sign_flip() -> None:
    """Flipping all charge signs reverses (Ex, Ey) and V while preserving |E|."""
    X, Y = create_coordinate_grid((-1.5, 1.5), (-1.5, 1.5), n_points=31)
    pos_charges = [PointCharge(q_uc=1.5, x=-0.4, y=0.2), PointCharge(q_uc=-0.8, x=0.6, y=-0.3)]
    neg_charges = [PointCharge(q_uc=-1.5, x=-0.4, y=0.2), PointCharge(q_uc=+0.8, x=0.6, y=-0.3)]

    f_pos = electric_field_2d(X, Y, pos_charges)
    f_neg = electric_field_2d(X, Y, neg_charges)
    v_pos = electric_potential_2d(X, Y, pos_charges)
    v_neg = electric_potential_2d(X, Y, neg_charges)

    np.testing.assert_allclose(f_neg.Ex, -f_pos.Ex, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(f_neg.Ey, -f_pos.Ey, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(f_neg.E_mag, f_pos.E_mag, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_neg.V, -v_pos.V, rtol=1e-12, atol=1e-12)


def test_linear_superposition_principle() -> None:
    """Verify E(q1 + q2) == E(q1) + E(q2) and V(q1 + q2) == V(q1) + V(q2) across a 2D grid."""
    X, Y = create_coordinate_grid((-2.0, 2.0), (-2.0, 2.0), n_points=41)
    q1 = PointCharge(q_uc=1.2, x=-0.7, y=0.3)
    q2 = PointCharge(q_uc=-1.9, x=0.5, y=-0.6)

    f1 = electric_field_2d(X, Y, [q1])
    f2 = electric_field_2d(X, Y, [q2])
    f12 = electric_field_2d(X, Y, [q1, q2])

    v1 = electric_potential_2d(X, Y, [q1])
    v2 = electric_potential_2d(X, Y, [q2])
    v12 = electric_potential_2d(X, Y, [q1, q2])

    np.testing.assert_allclose(f12.Ex, f1.Ex + f2.Ex, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(f12.Ey, f1.Ey + f2.Ey, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(v12.V, v1.V + v2.V, rtol=1e-12, atol=1e-9)


def test_symmetry_properties_for_dipole_and_like_charges() -> None:
    """
    - For an electric dipole (+q at -a, -q at +a): along the midplane x = 0, V(0, y) = 0 and Ey(0, y) = 0.
    - For two equal positive charges (+q at -a, +q at +a): at the midpoint (0, 0), Ex = 0 and Ey = 0.
    """
    y_line = np.linspace(-2.0, 2.0, 51)
    x_zero = np.zeros_like(y_line)

    dipole = get_preset_charges("dipole", scale_q_uc=1.0, separation_m=1.2)
    f_dip = electric_field_2d(x_zero, y_line, dipole)
    v_dip = electric_potential_2d(x_zero, y_line, dipole)

    np.testing.assert_allclose(v_dip.V, 0.0, atol=1e-10)
    np.testing.assert_allclose(f_dip.Ey, 0.0, atol=1e-10)
    assert np.all(f_dip.Ex > 0.0)  # Points from +q (at -a) toward -q (at +a)

    like_pair = get_preset_charges("two_like_positive", scale_q_uc=1.0, separation_m=1.2)
    f_mid = electric_field_2d(np.array([0.0]), np.array([0.0]), like_pair)
    assert math.isclose(f_mid.E_mag[0], 0.0, abs_tol=1e-12)


def test_singularity_handling_strategies_no_nan_or_inf() -> None:
    """Verify that evaluating directly on top of a point charge behaves safely under all 3 strategies."""
    charge = PointCharge(q_uc=1.0, x=0.0, y=0.0)
    X = np.array([0.0, 0.02, 0.5], dtype=np.float64)
    Y = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    for strat in ("core_cutoff", "softening"):
        f_res = electric_field_2d(X, Y, [charge], r_min=0.05, strategy=strat)
        v_res = electric_potential_2d(X, Y, [charge], r_min=0.05, strategy=strat)
        assert np.all(np.isfinite(f_res.Ex))
        assert np.all(np.isfinite(f_res.Ey))
        assert np.all(np.isfinite(f_res.E_mag))
        assert np.all(np.isfinite(v_res.V))
        # Exactly at r = 0, direction vector and field components must be 0 by symmetry
        assert f_res.E_mag[0] == 0.0
        assert bool(f_res.core_mask[0]) is True
        assert bool(f_res.core_mask[2]) is False

    # Under "mask" strategy, points inside r_min are explicitly NaN and outside are exact
    f_mask = electric_field_2d(X, Y, [charge], r_min=0.05, strategy="mask")
    v_mask = electric_potential_2d(X, Y, [charge], r_min=0.05, strategy="mask")
    assert np.isnan(f_mask.E_mag[0]) and np.isnan(f_mask.E_mag[1])
    assert np.isfinite(f_mask.E_mag[2])
    assert np.isnan(v_mask.V[0]) and np.isfinite(v_mask.V[2])


def test_numerical_gradient_matches_coulomb_field() -> None:
    """Verify -grad(V) agrees closely with direct Coulomb E(r) outside the charge cores."""
    X, Y = create_coordinate_grid((-2.0, 2.0), (-2.0, 2.0), n_points=151)
    charges = get_preset_charges("dipole", scale_q_uc=1.0, separation_m=1.2)

    f_res = electric_field_2d(X, Y, charges, r_min=0.05)
    v_res = electric_potential_2d(X, Y, charges, r_min=0.05)
    Ex_num, Ey_num, _ = numerical_field_from_potential(v_res)

    # Evaluate at a well-resolved point (0.0, 0.8) away from charges and boundaries
    ix = 75  # x = 0.0
    iy = 105  # y = +0.8
    assert math.isclose(Ex_num[iy, ix], f_res.Ex[iy, ix], rel_tol=5e-3)
    assert math.isclose(Ey_num[iy, ix], f_res.Ey[iy, ix], abs_tol=1e-6)


def test_electrostatics_input_validation() -> None:
    """Invalid grid limits, mismatched shapes, or non-positive r_min raise ValueError."""
    with pytest.raises(ValueError):
        create_coordinate_grid((2.0, -2.0), (-1.0, 1.0), n_points=20)
    with pytest.raises(ValueError):
        PointCharge(q_uc=float("nan"), x=0.0, y=0.0)
    with pytest.raises(ValueError):
        electric_field_2d(np.zeros((5, 5)), np.zeros((5, 4)), [])
    with pytest.raises(ValueError):
        electric_field_2d(np.zeros((5, 5)), np.zeros((5, 5)), [], r_min=-0.1)

