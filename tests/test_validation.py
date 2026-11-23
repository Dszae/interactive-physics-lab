"""
Automated unit tests for the validation and visualization modules.
"""

from __future__ import annotations

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from invisible_physics import (
    PlaneWaveVacuum,
    create_coordinate_grid,
    create_electromagnetic_wave_animation,
    electric_field_2d,
    electric_potential_2d,
    evaluate_plane_wave_1d,
    find_collinear_zero_field_point,
    get_preset_charges,
    magnetic_field_circular_loop_xz,
    magnetic_field_dipole_2d,
    magnetic_field_straight_wire_2d,
    plot_combined_field_and_potential,
    plot_electric_field_heatmap,
    plot_electric_field_streamlines,
    plot_electric_field_vectors,
    plot_electric_potential_landscape,
    plot_electromagnetic_wave_2d_panels,
    plot_electromagnetic_wave_3d,
    plot_magnetic_field_2d,
    plot_potential_surface_3d,
    validate_potential_gradient_consistency,
    verify_circular_loop_on_axis,
    verify_inverse_square_law,
)

matplotlib.use("Agg")


def test_verify_inverse_square_law_slope() -> None:
    """Log-log slope of |E(r)| vs r outside the core cutoff must equal -2.0 to machine precision."""
    res = verify_inverse_square_law(q_uc=2.0, radii_m=(0.2, 0.5, 1.0, 2.0, 5.0))
    assert math.isclose(float(res["log_log_slope"]), -2.0, rel_tol=1e-12)
    assert float(res["max_relative_error"]) < 1e-12


def test_find_collinear_zero_field_point_same_and_opposite_signs() -> None:
    """
    Test analytical vs root-solved zero-field points:
    1. Equal like charges (+1 uC at -1 m, +1 uC at +1 m) -> null point at x = 0.0 m.
    2. Unequal like charges (+1 uC at 0 m, +4 uC at 3 m) -> null point at x = 1.0 m.
    3. Unequal opposite charges (+4 uC at 0 m, -1 uC at 2 m) -> null point at x = 4.0 m.
    """
    res_sym = find_collinear_zero_field_point(1.0, -1.0, 1.0, 1.0)
    assert math.isclose(res_sym["x_analytical_m"], 0.0, abs_tol=1e-12)
    assert math.isclose(res_sym["x_numerical_m"], 0.0, abs_tol=1e-10)

    res_like = find_collinear_zero_field_point(1.0, 0.0, 4.0, 3.0)
    assert math.isclose(res_like["x_analytical_m"], 1.0, rel_tol=1e-12)
    assert math.isclose(res_like["x_numerical_m"], 1.0, rel_tol=1e-9)

    res_opp = find_collinear_zero_field_point(4.0, 0.0, -1.0, 2.0)
    assert math.isclose(res_opp["x_analytical_m"], 4.0, rel_tol=1e-12)
    assert math.isclose(res_opp["x_numerical_m"], 4.0, rel_tol=1e-9)

    # Equal and opposite dipole has no finite zero-field point
    with pytest.raises(ValueError, match="no finite zero-field point"):
        find_collinear_zero_field_point(1.0, -1.0, -1.0, 1.0)


@pytest.mark.parametrize(
    ("q1_uc", "q2_uc"),
    [
        (1.0, 400.0),  # like charges: null within 5% of the weaker charge
        (1.0, 1.0e4),
        (-1.0, -1.0e4),
        (1.0e4, 1.0),  # weaker charge on the right
        (-1.0, 1.0e7),  # opposite charges: null within 1e-3 m of the weaker charge
        (1.0e7, -1.0),
    ],
)
def test_find_collinear_zero_field_point_for_lopsided_charges(q1_uc: float, q2_uc: float) -> None:
    """
    The null point of a lopsided pair sits very close to the weaker charge. The root
    search must still bracket it: x_numerical has to agree with the analytical value.
    """
    res = find_collinear_zero_field_point(q1_uc, 0.0, q2_uc, 1.0)
    assert math.isclose(res["x_numerical_m"], res["x_analytical_m"], rel_tol=1e-9, abs_tol=1e-12)


def test_validate_potential_gradient_consistency_convergence() -> None:
    """
    Verify that increasing grid resolution reduces the median relative error between
    direct Coulomb field E(r) and numerical gradient -grad(V(r)) (2nd-order finite differences).
    """
    charges = get_preset_charges("dipole", scale_q_uc=1.0, separation_m=1.2)
    rep_coarse = validate_potential_gradient_consistency(charges, n_points=81, r_eval_min=0.4)
    rep_fine = validate_potential_gradient_consistency(charges, n_points=201, r_eval_min=0.4)

    assert rep_fine.median_rel_error < rep_coarse.median_rel_error
    assert rep_fine.median_rel_error < 2e-3
    assert rep_fine.rms_angle_error_deg < 0.25


def test_verify_circular_loop_on_axis_agreement() -> None:
    """Elliptic-integral loop field on-axis matches exact elementary formula to machine precision."""
    rep = verify_circular_loop_on_axis(current_amps=10.0, radius_m=0.6)
    assert rep["max_relative_error_bz"] < 1e-12
    assert rep["max_transverse_bx_tesla"] == 0.0


def test_all_visualization_functions_execute_cleanly() -> None:
    """Verify that every plotting function in visualization.py produces valid axes with labels."""
    X, Y = create_coordinate_grid((-1.5, 1.5), (-1.5, 1.5), n_points=31)
    charges = get_preset_charges("dipole", scale_q_uc=1.0, separation_m=1.0)
    f_res = electric_field_2d(X, Y, charges)
    v_res = electric_potential_2d(X, Y, charges)

    fig1, ax1 = plot_electric_field_vectors(f_res, charges, stride=3)
    assert ax1.get_xlabel() != "" and ax1.get_ylabel() != ""
    plt.close(fig1)

    fig2, ax2 = plot_electric_field_streamlines(f_res, charges)
    assert ax2.get_xlabel() != ""
    plt.close(fig2)

    fig_hm, ax_hm = plot_electric_field_heatmap(f_res, charges)
    assert ax_hm.get_xlabel() != ""
    plt.close(fig_hm)

    fig3, ax3 = plot_electric_potential_landscape(v_res, charges)
    assert ax3.get_xlabel() != ""
    plt.close(fig3)

    fig4, ax4 = plot_combined_field_and_potential(f_res, v_res, charges)
    assert ax4.get_xlabel() != ""
    plt.close(fig4)

    fig5, ax5 = plot_potential_surface_3d(v_res)
    assert ax5.get_xlabel() != ""
    plt.close(fig5)

    for mag_res in [
        magnetic_field_straight_wire_2d(X, Y, current_amps=10.0),
        magnetic_field_circular_loop_xz(X, Y, current_amps=10.0, radius_m=0.6),
        magnetic_field_dipole_2d(X, Y, dipole_moment_am2=5.0),
    ]:
        fig_m, ax_m = plot_magnetic_field_2d(mag_res, current_amps=10.0, loop_radius_m=0.6)
        assert ax_m.get_xlabel() != ""
        plt.close(fig_m)

    wave = PlaneWaveVacuum.from_frequency(100.0e6, e0_v_per_m=50.0)
    prof = evaluate_plane_wave_1d(wave, np.linspace(0.0, 3.0, 50), t_s=0.0)
    fig_w3d, ax_w3d = plot_electromagnetic_wave_3d(prof)
    assert ax_w3d.get_xlabel() != ""
    plt.close(fig_w3d)

    fig_w2d, axes_w2d = plot_electromagnetic_wave_2d_panels(prof)
    assert len(axes_w2d) == 3
    plt.close(fig_w2d)

    fig_anim, anim = create_electromagnetic_wave_animation(wave, n_frames=4, n_points=40)
    assert anim is not None
    plt.close(fig_anim)
