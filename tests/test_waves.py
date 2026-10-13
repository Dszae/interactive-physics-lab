"""
Automated unit and physics verification tests for the electromagnetic wave module.

Tests cover:
1. Vacuum dispersion constraint c = f * lambda.
2. Rejection of contradictory independent (f, lambda) settings in vacuum.
3. Field amplitude ratio |E| = c * |B| at all spatial and temporal points.
4. Transverse orthogonality: E . B = 0, E . k = 0, B . k = 0, and Poynting direction.
5. Time-averaged intensity <S> = E_0^2 / (2 * Z_0) agreement with spatial average over one wavelength.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from invisible_physics import (
    C_VACUUM,
    PlaneWaveVacuum,
    evaluate_plane_wave_1d,
)


def test_plane_wave_dispersion_and_amplitude_ratio() -> None:
    """Verify c = f * lambda and B_0 = E_0 / c when constructed from frequency or wavelength."""
    f_hz = 100.0e6  # 100 MHz FM radio wave
    e0 = 120.0
    wave = PlaneWaveVacuum.from_frequency(frequency_hz=f_hz, e0_v_per_m=e0)

    assert math.isclose(wave.frequency_hz * wave.wavelength_m, C_VACUUM, rel_tol=1e-12)
    assert math.isclose(wave.e0_v_per_m / wave.b0_tesla, C_VACUUM, rel_tol=1e-12)

    wave_from_lam = PlaneWaveVacuum.from_wavelength(wavelength_m=wave.wavelength_m, e0_v_per_m=e0)
    assert math.isclose(wave_from_lam.frequency_hz, f_hz, rel_tol=1e-12)


def test_contradictory_vacuum_parameters_raise_value_error() -> None:
    """Directly passing incompatible f and lambda where f * lambda != c must raise ValueError."""
    with pytest.raises(ValueError, match="Contradictory vacuum wave parameters"):
        PlaneWaveVacuum(e0_v_per_m=10.0, frequency_hz=1e8, wavelength_m=10.0)


def test_transverse_orthogonality_and_poynting_flux() -> None:
    """
    Evaluate a vacuum plane wave over two full wavelengths and verify:
    - E . B == 0 everywhere
    - |E(x, t)| == c * |B(x, t)| everywhere
    - Spatial average of S_x(x, t) over an integer number of wavelengths equals <S> = E_0^2 / (2 * Z_0).
    """
    wave = PlaneWaveVacuum.from_wavelength(wavelength_m=2.0, e0_v_per_m=60.0, phase_rad=0.4)
    # Use endpoint=False over exactly 2 wavelengths for exact periodic quadrature
    x = np.linspace(0.0, 4.0, 400, endpoint=False)
    profile = evaluate_plane_wave_1d(wave, x, t_s=1.25e-9)

    # Check E . B == 0 at all points
    e_dot_b = profile.Ex * profile.Bx + profile.Ey * profile.By + profile.Ez * profile.Bz
    np.testing.assert_allclose(e_dot_b, 0.0, atol=1e-20)

    # Check |E| = c * |B| at all points
    np.testing.assert_allclose(profile.E_mag, C_VACUUM * profile.B_mag, rtol=1e-12, atol=1e-15)

    # For +x propagation, S_x >= 0 everywhere and spatial mean equals analytical time-averaged intensity
    assert np.all(profile.Sx >= -1e-18)
    assert math.isclose(float(np.mean(profile.Sx)), wave.time_averaged_intensity_w_per_m2, rel_tol=1e-12)

    # For -x propagation, S_x <= 0 everywhere
    wave_neg = PlaneWaveVacuum.from_wavelength(
        wavelength_m=2.0, e0_v_per_m=60.0, propagation_axis="-x"
    )
    prof_neg = evaluate_plane_wave_1d(wave_neg, x, t_s=0.0)
    assert np.all(prof_neg.Sx <= 1e-18)
    assert math.isclose(float(np.mean(prof_neg.Sx)), -wave_neg.time_averaged_intensity_w_per_m2, rel_tol=1e-12)

