"""
Electromagnetic wave propagation module for idealized plane waves in vacuum.

Physical Model & Maxwell's Equations in Vacuum:
For a monochromatic, linearly polarized plane electromagnetic wave propagating in vacuum
along the +x direction (k = k * x_hat):
- Electric field oscillates along the y-axis:
  E(x, t) = E_0 * cos(k * x - omega * t + phi) * y_hat
- Magnetic field oscillates along the z-axis:
  B(x, t) = B_0 * cos(k * x - omega * t + phi) * z_hat

Enforced Physical Constraints in Vacuum:
1. Dispersion relation: c = f * lambda  (omega / k = c)
2. Transverse orthogonality: E . B = 0, E . k = 0, B . k = 0, and (E x B) points along +k.
3. Field amplitude ratio: |E| = c * |B|, so B_0 = E_0 / c at every point in space and time.
4. Instantaneous Poynting vector (energy flux density, W/m^2):
   S(x, t) = (1 / mu_0) * (E x B) = (E_y * B_z / mu_0) * x_hat.

To prevent contradictory independent settings of frequency f and wavelength lambda in vacuum,
`PlaneWaveVacuum` accepts EITHER `frequency_hz` OR `wavelength_m` (or verifies that both satisfy
c = f * lambda within floating-point tolerance).

Note on Materials vs. Vacuum:
In a lossless dielectric medium with refractive index n = sqrt(epsilon_r * mu_r) > 1, the phase
speed is v_p = c / n, lambda_medium = v_p / f, and |E| = v_p * |B|. This module models vacuum
plane waves rigorously and provides an explicit helper for comparing dielectric phase speed when requested.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .constants import C_VACUUM, EPSILON_0, MU_0, Z_0_VACUUM


@dataclass(frozen=True)
class PlaneWaveVacuum:
    """
    Represents a physically consistent monochromatic plane electromagnetic wave in vacuum.

    Enforces:
        c = f * lambda
        B_0 = E_0 / c

    Parameters
    ----------
    e0_v_per_m : float
        Peak electric-field amplitude E_0 in volts per meter (V/m). Must be non-negative.
    frequency_hz : float
        Wave frequency f in hertz (Hz). Must be strictly positive.
    wavelength_m : float
        Wavelength lambda in meters (m). Automatically linked to frequency_hz via c = f * lambda.
    phase_rad : float
        Initial phase phi in radians (rad).
    propagation_axis : str
        Propagation direction, default "+x" (with E along y and B along z).
    """

    e0_v_per_m: float
    frequency_hz: float
    wavelength_m: float
    phase_rad: float = 0.0
    propagation_axis: Literal["+x", "-x"] = "+x"

    def __post_init__(self) -> None:
        if self.e0_v_per_m < 0.0 or not np.isfinite(self.e0_v_per_m):
            raise ValueError(f"Electric field amplitude e0_v_per_m must be >= 0 and finite; got {self.e0_v_per_m}.")
        if self.frequency_hz <= 0.0 or not np.isfinite(self.frequency_hz):
            raise ValueError(f"frequency_hz must be > 0 and finite; got {self.frequency_hz}.")
        if self.wavelength_m <= 0.0 or not np.isfinite(self.wavelength_m):
            raise ValueError(f"wavelength_m must be > 0 and finite; got {self.wavelength_m}.")
        if not np.isfinite(self.phase_rad):
            raise ValueError(f"phase_rad must be finite; got {self.phase_rad}.")
        if self.propagation_axis not in ("+x", "-x"):
            raise ValueError(f"propagation_axis must be '+x' or '-x'; got {self.propagation_axis}.")

        # Enforce vacuum dispersion relation c = f * lambda
        product = self.frequency_hz * self.wavelength_m
        if not math.isclose(product, C_VACUUM, rel_tol=1e-9):
            raise ValueError(
                f"Contradictory vacuum wave parameters: f * lambda = {product:.6e} m/s != c = {C_VACUUM:.6e} m/s. "
                "Use PlaneWaveVacuum.from_frequency() or PlaneWaveVacuum.from_wavelength() to keep c = f * lambda consistent."
            )

    @classmethod
    def from_frequency(
        cls,
        frequency_hz: float,
        e0_v_per_m: float = 100.0,
        phase_rad: float = 0.0,
        propagation_axis: Literal["+x", "-x"] = "+x",
    ) -> "PlaneWaveVacuum":
        """Construct a vacuum plane wave from frequency f (Hz), deriving lambda = c / f."""
        if frequency_hz <= 0.0 or not np.isfinite(frequency_hz):
            raise ValueError(f"frequency_hz must be positive and finite; got {frequency_hz}.")
        wavelength_m = C_VACUUM / float(frequency_hz)
        return cls(
            e0_v_per_m=float(e0_v_per_m),
            frequency_hz=float(frequency_hz),
            wavelength_m=wavelength_m,
            phase_rad=float(phase_rad),
            propagation_axis=propagation_axis,
        )

    @classmethod
    def from_wavelength(
        cls,
        wavelength_m: float,
        e0_v_per_m: float = 100.0,
        phase_rad: float = 0.0,
        propagation_axis: Literal["+x", "-x"] = "+x",
    ) -> "PlaneWaveVacuum":
        """Construct a vacuum plane wave from wavelength lambda (m), deriving f = c / lambda."""
        if wavelength_m <= 0.0 or not np.isfinite(wavelength_m):
            raise ValueError(f"wavelength_m must be positive and finite; got {wavelength_m}.")
        frequency_hz = C_VACUUM / float(wavelength_m)
        return cls(
            e0_v_per_m=float(e0_v_per_m),
            frequency_hz=frequency_hz,
            wavelength_m=float(wavelength_m),
            phase_rad=float(phase_rad),
            propagation_axis=propagation_axis,
        )

    @property
    def b0_tesla(self) -> float:
        """Peak magnetic-field amplitude B_0 = E_0 / c in teslas (T)."""
        return self.e0_v_per_m / C_VACUUM

    @property
    def wave_speed_m_per_s(self) -> float:
        """Phase and group speed in vacuum c (m/s)."""
        return C_VACUUM

    @property
    def wavenumber_rad_per_m(self) -> float:
        """Angular wavenumber k = 2 * pi / lambda in rad/m."""
        return (2.0 * math.pi) / self.wavelength_m

    @property
    def angular_frequency_rad_per_s(self) -> float:
        """Angular frequency omega = 2 * pi * f in rad/s."""
        return 2.0 * math.pi * self.frequency_hz

    @property
    def period_s(self) -> float:
        """Oscillation period T = 1 / f in seconds (s)."""
        return 1.0 / self.frequency_hz

    @property
    def time_averaged_intensity_w_per_m2(self) -> float:
        """Time-averaged Poynting flux (irradiance) <S> = E_0^2 / (2 * mu_0 * c) in W/m^2."""
        return (self.e0_v_per_m * self.e0_v_per_m) / (2.0 * Z_0_VACUUM)


@dataclass(frozen=True)
class PlaneWaveProfileResult:
    """
    Container for evaluated 3D vectors of a plane electromagnetic wave along the propagation axis x.

    For propagation along +x:
    - E(x, t) = (0, Ey, 0) in V/m
    - B(x, t) = (0, 0, Bz) in T (for "-x" propagation, Bz reverses sign so E x B points along -x)
    - S(x, t) = Sx * x_hat in W/m^2
    """

    x: NDArray[np.float64]
    t_s: float
    Ex: NDArray[np.float64]
    Ey: NDArray[np.float64]
    Ez: NDArray[np.float64]
    Bx: NDArray[np.float64]
    By: NDArray[np.float64]
    Bz: NDArray[np.float64]
    E_mag: NDArray[np.float64]
    B_mag: NDArray[np.float64]
    Sx: NDArray[np.float64]
    energy_density_j_per_m3: NDArray[np.float64]
    wave: PlaneWaveVacuum


def evaluate_plane_wave_1d(
    wave: PlaneWaveVacuum,
    x: NDArray[np.floating],
    t_s: float = 0.0,
) -> PlaneWaveProfileResult:
    """
    Evaluate the spatial waveform E(x, t), B(x, t), Poynting vector S(x, t), and electromagnetic
    energy density u(x, t) along propagation coordinate x at time t_s.

    Parameters
    ----------
    wave : PlaneWaveVacuum
        Physically consistent vacuum plane wave specification.
    x : numpy.ndarray
        1D array of positions along the x-axis in meters (m).
    t_s : float, default 0.0
        Observation time t in seconds (s).

    Returns
    -------
    PlaneWaveProfileResult
    """
    if not isinstance(wave, PlaneWaveVacuum):
        raise TypeError("wave must be an instance of PlaneWaveVacuum.")
    if not np.isfinite(t_s):
        raise ValueError(f"t_s must be finite; got {t_s}.")

    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError("x must be a non-empty 1D coordinate array.")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must contain only finite values.")

    k = wave.wavenumber_rad_per_m
    omega = wave.angular_frequency_rad_per_s
    direction_sign = 1.0 if wave.propagation_axis == "+x" else -1.0

    # Phase argument: k * x - omega * t + phi for +x propagation, or -k * x - omega * t + phi for -x
    phase = direction_sign * k * x_arr - omega * float(t_s) + wave.phase_rad
    cos_phase = np.cos(phase)

    zeros = np.zeros_like(x_arr, dtype=np.float64)
    Ex = zeros.copy()
    Ey = wave.e0_v_per_m * cos_phase
    Ez = zeros.copy()

    Bx = zeros.copy()
    By = zeros.copy()
    # By Maxwell-Faraday law (curl E = -dB/dt), for +x propagation Ey > 0 pairs with Bz > 0;
    # for -x propagation, Ey > 0 pairs with Bz < 0 so that E x B points along -x.
    Bz = direction_sign * wave.b0_tesla * cos_phase

    E_mag = np.abs(Ey)
    B_mag = np.abs(Bz)

    # Poynting vector x-component: S_x = (Ey * Bz) / mu_0
    Sx = (Ey * Bz) / MU_0

    # Total electromagnetic energy density: u = 0.5 * epsilon_0 * E^2 + 0.5 * B^2 / mu_0
    u_em = 0.5 * EPSILON_0 * (Ey * Ey) + 0.5 * (Bz * Bz) / MU_0

    return PlaneWaveProfileResult(
        x=x_arr,
        t_s=float(t_s),
        Ex=Ex,
        Ey=Ey,
        Ez=Ez,
        Bx=Bx,
        By=By,
        Bz=Bz,
        E_mag=E_mag,
        B_mag=B_mag,
        Sx=Sx,
        energy_density_j_per_m3=u_em,
        wave=wave,
    )

