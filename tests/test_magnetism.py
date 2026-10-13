"""
Automated unit and physics verification tests for the magnetostatics module.

Tests cover:
1. Straight wire Biot-Savart magnitude B = mu_0 * I / (2 * pi * r) and 1/r scaling.
2. Right-hand rule azimuthal vector orientation (B . r = 0 and z_hat . (r x B) > 0 for I > 0).
3. Circular current loop exact agreement on the z-axis with Bz = mu_0 * I * R^2 / (2 * (R^2 + z^2)^(3/2)).
4. Point magnetic dipole axial vs. equatorial field ratio (|B_axial(r)| == 2 * |B_equatorial(r)|).
5. Singularity protection and input validation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from invisible_physics import (
    MU_0,
     create_coordinate_grid,
    magnetic_field_circular_loop_on_axis,
    magnetic_field_circular_loop_xz,
    magnetic_field_dipole_2d,
    magnetic_field_straight_wire_1d,
    magnetic_field_straight_wire_2d,
)


def test_straight_wire_magnitude_and_inverse_r_scaling() -> None:
    """Verify B = mu_0 * I / (2 * pi * r) and 1/r scaling for a long straight wire."""
    I = 15.0
    radii = np.array([0.2, 0.4, 0.8, 1.6], dtype=np.float64)
    B_1d = magnetic_field_straight_wire_1d(radii, current_amps=I, r_min=0.05)

    expected = (MU_0 * I) / (2.0 * math.pi * radii)
    np.testing.assert_allclose(B_1d, expected, rtol=1e-12)

    # Doubling distance halves B
    assert math.isclose(B_1d[0] / B_1d[1], 2.0, rel_tol=1e-12)
    assert math.isclose(B_1d[1] / B_1d[2], 2.0, rel_tol=1e-12)


def test_straight_wire_right_hand_rule_and_orthogonality() -> None:
    """
    For I > 0 (+z out of page):
    - At (+r, 0), B points in +y (Bx = 0, By > 0).
    - At (0, +r), B points in -x (Bx < 0, By = 0).
    - Everywhere outside core, r . B == 0 (azimuthal tangent).
    """
    X, Y = create_coordinate_grid((-1.5, 1.5), (-1.5, 1.5), n_points=31)
    res_pos = magnetic_field_straight_wire_2d(X, Y, current_amps=10.0, r_min=0.05)
    res_neg = magnetic_field_straight_wire_2d(X, Y, current_amps=-10.0, r_min=0.05)

    # Reversing current reverses B vector components
    np.testing.assert_allclose(res_neg.Bx, -res_pos.Bx, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(res_neg.By, -res_pos.By, rtol=1e-12, atol=1e-15)

    # Radial dot product X * Bx + Y * By == 0 everywhere
    radial_dot = X * res_pos.Bx + Y * res_pos.By
    np.testing.assert_allclose(radial_dot, 0.0, atol=1e-18)

    # Specific cardinal points (1.0, 0.0) and (0.0, 1.0)
    pt_res = magnetic_field_straight_wire_2d(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        current_amps=10.0,
    )
    assert math.isclose(pt_res.Bx[0], 0.0, abs_tol=1e-18) and pt_res.By[0] > 0.0
    assert pt_res.Bx[1] < 0.0 and math.isclose(pt_res.By[1], 0.0, abs_tol=1e-18)


def test_circular_loop_on_axis_matches_analytical_formula() -> None:
    """
    At the center of a circular loop (x = 0, z = 0), Bz = mu_0 * I / (2 * R),
    and along the entire z-axis (x = 0, z), elliptic-integral Bz matches elementary Biot-Savart.
    """
    I = 12.0
    R = 0.75
    z = np.linspace(-2.0, 2.0, 41)
    x = np.zeros_like(z)

    loop_2d = magnetic_field_circular_loop_xz(x, z, current_amps=I, radius_m=R)
    bz_exact = magnetic_field_circular_loop_on_axis(z, current_amps=I, radius_m=R)

    np.testing.assert_allclose(loop_2d.By, bz_exact, rtol=1e-12)
    np.testing.assert_allclose(loop_2d.Bx, 0.0, atol=1e-18)

    # Check center value at z = 0 (index 20)
    expected_center = (MU_0 * I) / (2.0 * R)
    assert math.isclose(loop_2d.By[20], expected_center, rel_tol=1e-12)


def test_point_magnetic_dipole_axial_to_equatorial_ratio() -> None:
    """
    For a point magnetic dipole oriented along +y (90 deg), the field magnitude at axial distance r
    along the y-axis is exactly twice the field magnitude at equatorial distance r along the x-axis,
    and both decay as 1 / r^3 outside r_min.
    """
    r_val = 1.0
    X = np.array([0.0, r_val, 0.0], dtype=np.float64)
    Y = np.array([r_val, 0.0, 2.0 * r_val], dtype=np.float64)

    res = magnetic_field_dipole_2d(X, Y, dipole_moment_am2=5.0, orientation_deg=90.0, r_min=0.1)
    b_axial_r = res.B_mag[0]
    b_equatorial_r = res.B_mag[1]
    b_axial_2r = res.B_mag[2]

    assert math.isclose(b_axial_r / b_equatorial_r, 2.0, rel_tol=1e-12)
    assert math.isclose(b_axial_r / b_axial_2r, 8.0, rel_tol=1e-12)  # 1/r^3 scaling -> 2^3 = 8


def test_magnetism_input_validation() -> None:
    """Invalid radii or non-finite currents raise ValueError."""
    with pytest.raises(ValueError):
        magnetic_field_straight_wire_1d(np.array([0.0, 1.0]), current_amps=5.0)
    with pytest.raises(ValueError):
        magnetic_field_circular_loop_on_axis(0.0, current_amps=5.0, radius_m=-0.5)

