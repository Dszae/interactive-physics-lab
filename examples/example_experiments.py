"""
Command-line script executing reproducible physics experiments and generating
visual figures for documentation.

Usage:
    python examples/example_experiments.py [--no-show] [--save-dir docs/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from invisible_physics import (
    PlaneWaveVacuum,
    create_coordinate_grid,
    electric_field_2d,
    electric_potential_2d,
    evaluate_plane_wave_1d,
    find_collinear_zero_field_point,
    get_preset_charges,
    magnetic_field_circular_loop_xz,
    magnetic_field_straight_wire_2d,
    plot_combined_field_and_potential,
    plot_electric_field_streamlines,
    plot_electromagnetic_wave_3d,
    plot_magnetic_field_2d,
    plot_potential_surface_3d,
    validate_potential_gradient_consistency,
    verify_circular_loop_on_axis,
    verify_inverse_square_law,
)


def run_experiments(save_dir: Path | None = None, show: bool = False) -> None:
    if not show:
        matplotlib.use("Agg")

    print("=" * 74)
    print("INVISIBLE PHYSICS — REPRODUCIBLE NUMERICAL EXPERIMENTS & VALIDATION")
    print("=" * 74)

    # 1. Inverse-square law verification
    inv_sq = verify_inverse_square_law(q_uc=1.0, radii_m=(0.25, 0.5, 1.0, 2.0, 4.0))
    print("\n[Experiment 1] Single Point Charge Inverse-Square Law:")
    print(f"  - Computed log-log slope d(ln|E|)/d(ln r) : {inv_sq['log_log_slope']:.12f} (exact: -2.0)")
    print(f"  - Max relative error vs Coulomb formula   : {inv_sq['max_relative_error']:.3e}")

    # 2. Zero-field (null point) discovery
    null_like = find_collinear_zero_field_point(q1_uc=1.0, x1_m=-1.0, q2_uc=4.0, x2_m=2.0)
    null_opp = find_collinear_zero_field_point(q1_uc=4.0, x1_m=-0.6, q2_uc=-1.0, x2_m=0.6)
    print("\n[Experiment 2] Collinear Zero-Field Null Points:")
    print(
        f"  - Unequal like charges (+1 uC at -1m, +4 uC at +2m): "
        f"x_null = {null_like['x_numerical_m']:+.6f} m (analytical: {null_like['x_analytical_m']:+.6f} m)"
    )
    print(
        f"  - Unequal opposite charges (+4 uC at -0.6m, -1 uC at +0.6m): "
        f"x_null = {null_opp['x_numerical_m']:+.6f} m (analytical: {null_opp['x_analytical_m']:+.6f} m)"
    )

    # 3. Potential-field gradient consistency
    dipole_charges = get_preset_charges("dipole", scale_q_uc=1.0, separation_m=1.2)
    grad_rep = validate_potential_gradient_consistency(dipole_charges, n_points=201, r_eval_min=0.35)
    print("\n[Experiment 3] Electrostatic Potential Gradient Consistency (E = -grad V):")
    print(f"  - Grid resolution                         : {grad_rep.grid_resolution} x {grad_rep.grid_resolution}")
    print(f"  - Median relative error (r >= 0.35 m)     : {grad_rep.median_rel_error * 100:.4f}%")
    print(f"  - RMS directional angle error             : {grad_rep.rms_angle_error_deg:.4f} deg")

    # 4. Circular current loop axial validation
    loop_rep = verify_circular_loop_on_axis(current_amps=15.0, radius_m=0.8)
    print("\n[Experiment 4] Circular Current Loop Biot-Savart Axial Verification:")
    print(f"  - Max relative error in Bz(0, z)          : {loop_rep['max_relative_error_bz']:.3e}")
    print(f"  - Max transverse Bx along symmetry axis   : {loop_rep['max_transverse_bx_tesla']:.3e} T")

    # 5. Vacuum electromagnetic wave verification
    wave = PlaneWaveVacuum.from_frequency(frequency_hz=100.0e6, e0_v_per_m=100.0)
    x_wave = np.linspace(0.0, 2.0 * wave.wavelength_m, 300)
    wave_prof = evaluate_plane_wave_1d(wave, x_wave, t_s=0.0)
    print("\n[Experiment 5] Vacuum Plane Electromagnetic Wave (100 MHz):")
    print(f"  - Derived wavelength lambda = c / f       : {wave.wavelength_m:.6f} m")
    print(f"  - Peak magnetic field B0 = E0 / c         : {wave.b0_tesla * 1e9:.4f} nT")
    print(f"  - Time-averaged intensity <S>             : {wave.time_averaged_intensity_w_per_m2:.4f} W/m^2")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        X, Y = create_coordinate_grid((-2.0, 2.0), (-2.0, 2.0), n_points=121)

        # Figure 1: Electric dipole streamlines & magnitude
        f_dip = electric_field_2d(X, Y, dipole_charges)
        v_dip = electric_potential_2d(X, Y, dipole_charges)
        fig1, _ = plot_electric_field_streamlines(
            f_dip, dipole_charges, title="Electric Dipole Field Streamlines & Magnitude |E|"
        )
        fig1.savefig(save_dir / "01_dipole_electric_field.png", bbox_inches="tight")
        plt.close(fig1)

        # Figure 2: Combined potential & field orthogonality for quadrupole
        quad_charges = get_preset_charges("symmetric_quadrupole", scale_q_uc=1.0, separation_m=1.4)
        f_quad = electric_field_2d(X, Y, quad_charges)
        v_quad = electric_potential_2d(X, Y, quad_charges)
        fig2, _ = plot_combined_field_and_potential(
            f_quad,
            v_quad,
            quad_charges,
            title="Symmetric Quadrupole: Potential V(r), Equipotentials & E-Field Lines",
        )
        fig2.savefig(save_dir / "02_quadrupole_field_and_potential.png", bbox_inches="tight")
        plt.close(fig2)

        # Figure 3: 3D potential landscape of dipole
        fig3, _ = plot_potential_surface_3d(
            v_dip, title="3D Electrostatic Potential Landscape V(x, y) — Electric Dipole"
        )
        fig3.savefig(save_dir / "03_dipole_potential_3d.png", bbox_inches="tight")
        plt.close(fig3)

        # Figure 4: Circular current loop magnetic field
        mag_loop = magnetic_field_circular_loop_xz(X, Y, current_amps=15.0, radius_m=0.8)
        fig4, _ = plot_magnetic_field_2d(
            mag_loop,
            title="Circular Current Loop (R = 0.8 m, I = 15 A): |B| & Field Lines",
            current_amps=15.0,
            loop_radius_m=0.8,
        )
        fig4.savefig(save_dir / "04_circular_loop_magnetic_field.png", bbox_inches="tight")
        plt.close(fig4)

        # Figure 5: 3D Plane EM wave
        fig5, _ = plot_electromagnetic_wave_3d(wave_prof)
        fig5.savefig(save_dir / "05_electromagnetic_plane_wave_3d.png", bbox_inches="tight")
        plt.close(fig5)
        print(f"\nSaved 5 publication-quality figures to: {save_dir.resolve()}")

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Invisible Physics validation experiments.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run headlessly without opening interactive Matplotlib windows.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="docs/figures",
        help="Directory to save generated plot images (default: docs/figures).",
    )
    args = parser.parse_args()
    save_path = Path(args.save_dir) if args.save_dir else None
    run_experiments(save_dir=save_path, show=not args.no_show)


if __name__ == "__main__":
    main()

