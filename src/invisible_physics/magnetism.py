"""
Magnetostatics calculation engine for steady currents and idealized magnetic dipoles.

Physical Models & Assumptions:
1. Infinitely Long Straight Wire (Ampere's / Biot-Savart Law):
   - Assumes steady direct current I flowing along the z-axis (perpendicular to the x-y plane).
   - Current I > 0 points in +z (out of the x-y plane toward the observer); I < 0 points in -z (into the page).
   - Right-hand rule: thumb along current (+z) -> fingers curl counterclockwise in the x-y plane.
   - Magnitude: B(r) = (mu_0 * |I|) / (2 * pi * r).
   - Vector components at (x, y) with r = sqrt(x^2 + y^2):
     Bx = -(mu_0 * I / (2 * pi * r)) * (y / r)
     By = +(mu_0 * I / (2 * pi * r)) * (x / r)

2. Circular Current Loop of Radius R in the x-y Plane (z = 0):
   - Evaluated in the meridional x-z plane (y = 0) where x is radial distance rho = |x| and z is axial coordinate.
   - Uses the exact analytical Biot-Savart solution in terms of complete elliptic integrals K(m) and E(m).
   - On the symmetry axis (x = 0), reduces exactly to:
     Bz(0, z) = (mu_0 * I * R^2) / (2 * (R^2 + z^2)^(3/2)).

3. Ideal Point Magnetic Dipole (Far-Field Approximation):
   - B(r) = (mu_0 / (4 * pi)) * [3 (m . r_hat) r_hat - m] / r^3.
   - IMPORTANT DISTINCTION: A point magnetic dipole is an asymptotic far-field idealization (r >> source size).
     A real physical bar magnet has finite length and width; inside and near a real bar magnet, the field
     differs substantially from a point dipole (which diverges at r -> 0). We enforce an explicit core
     exclusion/cutoff radius r_min and document this distinction clearly.

Note: All models here describe magnetostatic fields (dB/dt = 0, steady currents), which do not radiate
electromagnetic waves. Time-varying propagating electromagnetic waves are modeled separately in waves.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ellipe, ellipk

from .constants import DEFAULT_CORE_RADIUS_M, MU_0


@dataclass(frozen=True)
class MagneticField2DResult:
    """
    Container for 2D magnetic flux density B(r) calculation outputs.

    All field components (Bx, By, B_mag) are in SI units: teslas (T).
    Note: For meridional x-z plane calculations (such as the circular loop),
    X represents the transverse coordinate x and Y represents the axial coordinate z,
    while Bx is the transverse component and By is the axial component Bz.
    """

    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    Bx: NDArray[np.float64]
    By: NDArray[np.float64]
    B_mag: NDArray[np.float64]
    B_dir_x: NDArray[np.float64]
    B_dir_y: NDArray[np.float64]
    core_mask: NDArray[np.bool_]
    r_min: float
    model_name: str


def magnetic_field_straight_wire_1d(
    r: NDArray[np.floating] | float,
    current_amps: float,
    r_min: float = DEFAULT_CORE_RADIUS_M,
) -> NDArray[np.float64]:
    """
    Compute magnetic field magnitude B(r) = mu_0 * |I| / (2 * pi * r) at radial distance r
    from an idealized infinitely long straight wire.

    Parameters
    ----------
    r : array-like or float
        Radial distance(s) from the wire axis in meters (m). Must be positive.
    current_amps : float
        Steady electric current I in amperes (A).
    r_min : float, default 0.05
        Minimum wire core radius in meters (m) for singularity protection.

    Returns
    -------
    B_mag : numpy.ndarray
        Magnetic field magnitude in teslas (T).
    """
    if not np.isfinite(current_amps):
        raise ValueError(f"current_amps must be finite; got {current_amps}.")
    if r_min <= 0.0 or not np.isfinite(r_min):
        raise ValueError(f"r_min must be positive and finite; got {r_min}.")

    r_arr = np.asarray(r, dtype=np.float64)
    if np.any(r_arr <= 0.0) or not np.all(np.isfinite(r_arr)):
        raise ValueError("Radial distances r must be strictly positive and finite.")

    r_eff = np.maximum(r_arr, r_min)
    return (MU_0 * abs(float(current_amps))) / (2.0 * math.pi * r_eff)


def magnetic_field_straight_wire_2d(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    current_amps: float = 10.0,
    wire_x: float = 0.0,
    wire_y: float = 0.0,
    r_min: float = DEFAULT_CORE_RADIUS_M,
) -> MagneticField2DResult:
    """
    Compute the 2D magnetic vector field B(x, y) around an infinitely long straight wire
    oriented along the z-axis at position (wire_x, wire_y).

    Right-Hand Rule Convention:
    - current_amps > 0: Current flows in +z (out of the page). Field lines circulate counterclockwise.
    - current_amps < 0: Current flows in -z (into the page). Field lines circulate clockwise.

    Parameters
    ----------
    X, Y : numpy.ndarray
        2D coordinate arrays in meters (m).
    current_amps : float, default 10.0
        Electric current I in amperes (A).
    wire_x, wire_y : float, default (0.0, 0.0)
        Wire cross-section coordinates in the x-y plane in meters (m).
    r_min : float, default 0.05
        Core cutoff radius in meters (m) representing finite wire radius / singularity cutoff.

    Returns
    -------
    MagneticField2DResult
        Field components Bx, By, and magnitude B_mag in teslas (T).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    if X_arr.shape != Y_arr.shape or X_arr.size == 0:
        raise ValueError("X and Y must be non-empty arrays with the same shape.")
    if not (np.isfinite(current_amps) and np.isfinite(wire_x) and np.isfinite(wire_y)):
        raise ValueError("current_amps, wire_x, and wire_y must be finite numbers.")
    if r_min <= 0.0 or not np.isfinite(r_min):
        raise ValueError(f"r_min must be positive and finite; got {r_min}.")

    dx = X_arr - float(wire_x)
    dy = Y_arr - float(wire_y)
    r = np.hypot(dx, dy)
    core_mask = r < r_min
    r_eff = np.maximum(r, r_min)

    prefactor = (MU_0 * float(current_amps)) / (2.0 * math.pi * r_eff * r_eff)
    Bx = -prefactor * dy
    By = +prefactor * dx
    B_mag = np.hypot(Bx, By)

    with np.errstate(divide="ignore", invalid="ignore"):
        B_dir_x = np.where(B_mag > 0.0, Bx / B_mag, 0.0)
        B_dir_y = np.where(B_mag > 0.0, By / B_mag, 0.0)

    return MagneticField2DResult(
        X=X_arr,
        Y=Y_arr,
        Bx=Bx,
        By=By,
        B_mag=B_mag,
        B_dir_x=B_dir_x,
        B_dir_y=B_dir_y,
        core_mask=core_mask,
        r_min=float(r_min),
        model_name="straight_wire",
    )


def magnetic_field_circular_loop_on_axis(
    z: NDArray[np.floating] | float,
    current_amps: float,
    radius_m: float,
) -> NDArray[np.float64]:
    """
    Exact analytical magnetic field Bz(0, z) along the central symmetry axis (z-axis)
    of a circular current loop of radius R lying in the x-y plane (z = 0):

        Bz(0, z) = (mu_0 * I * R^2) / (2 * (R^2 + z^2)^(3/2))

    Parameters
    ----------
    z : array-like or float
        Axial position(s) along the loop axis in meters (m).
    current_amps : float
        Loop current I in amperes (A). Positive current flows counterclockwise when viewed from +z.
    radius_m : float
        Loop radius R in meters (m). Must be strictly positive.

    Returns
    -------
    Bz : numpy.ndarray
        Axial magnetic field in teslas (T).
    """
    if radius_m <= 0.0 or not np.isfinite(radius_m):
        raise ValueError(f"Loop radius_m must be positive and finite; got {radius_m}.")
    if not np.isfinite(current_amps):
        raise ValueError(f"current_amps must be finite; got {current_amps}.")

    z_arr = np.asarray(z, dtype=np.float64)
    R = float(radius_m)
    I = float(current_amps)
    denom = 2.0 * np.power(R * R + z_arr * z_arr, 1.5)
    return (MU_0 * I * R * R) / denom


def magnetic_field_circular_loop_xz(
    X: NDArray[np.floating],
    Z: NDArray[np.floating],
    current_amps: float = 10.0,
    radius_m: float = 0.8,
    r_min: float = DEFAULT_CORE_RADIUS_M,
) -> MagneticField2DResult:
    """
    Compute the exact 2D magnetic vector field (Bx, Bz) in the meridional x-z plane (y = 0)
    for a circular current loop of radius R lying in the x-y plane (z = 0).

    Uses the analytical Biot-Savart formulation in cylindrical coordinates (rho = |x|, z)
    via complete elliptic integrals K(m) and E(m):

        alpha^2 = (R - rho)^2 + z^2
        beta^2  = (R + rho)^2 + z^2
        m       = 1 - alpha^2 / beta^2 = 4 R rho / ((R + rho)^2 + z^2)

        Bz   = (mu_0 * I) / (2 * pi * beta) * [ K(m) + ((R^2 - rho^2 - z^2) / alpha^2) * E(m) ]
        Brho = (mu_0 * I * z) / (2 * pi * rho * beta) * [ -K(m) + ((R^2 + rho^2 + z^2) / alpha^2) * E(m) ]

    Near the wire cross-sections at (x, z) = (+R, 0) and (-R, 0), alpha -> 0; we regularize
    alpha^2 with r_min^2 to prevent division by zero at the ideal filament singularity.

    Parameters
    ----------
    X, Z : numpy.ndarray
        2D meshgrid coordinates in the x-z plane in meters (m).
    current_amps : float, default 10.0
        Current I in amperes (A). Positive I produces +Bz along the loop axis.
    radius_m : float, default 0.8
        Radius R of the circular loop in meters (m).
    r_min : float, default 0.05
        Filament core regularization radius in meters (m) around (x = +/-R, z = 0).

    Returns
    -------
    MagneticField2DResult
        Note: In the returned object, X is X, Y is Z, Bx is Bx, and By represents Bz.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    Z_arr = np.asarray(Z, dtype=np.float64)
    if X_arr.shape != Z_arr.shape or X_arr.size == 0:
        raise ValueError("X and Z must be non-empty arrays with the same shape.")
    if radius_m <= 0.0 or not np.isfinite(radius_m):
        raise ValueError(f"radius_m must be positive and finite; got {radius_m}.")
    if not np.isfinite(current_amps):
        raise ValueError(f"current_amps must be finite; got {current_amps}.")
    if r_min <= 0.0 or not np.isfinite(r_min):
        raise ValueError(f"r_min must be positive and finite; got {r_min}.")

    R = float(radius_m)
    I = float(current_amps)

    rho = np.abs(X_arr)
    dist_plus = np.hypot(X_arr - R, Z_arr)
    dist_minus = np.hypot(X_arr + R, Z_arr)
    core_mask = (dist_plus < r_min) | (dist_minus < r_min)

    # Regularize alpha_sq using r_min^2 near the wire filaments
    alpha_sq_raw = (R - rho) ** 2 + Z_arr**2
    alpha_sq = np.maximum(alpha_sq_raw, r_min * r_min)
    beta_sq = (R + rho) ** 2 + Z_arr**2
    beta = np.sqrt(beta_sq)

    m = (4.0 * R * rho) / beta_sq
    # Clamp m strictly below 1 - eps to keep K(m) finite inside filament core
    m_safe = np.clip(m, 0.0, 1.0 - (r_min / (2.0 * R + r_min)) ** 2)

    K_m = ellipk(m_safe)
    E_m = ellipe(m_safe)

    c0 = (MU_0 * I) / (2.0 * math.pi * beta)
    Bz = c0 * (K_m + ((R * R - rho * rho - Z_arr * Z_arr) / alpha_sq) * E_m)

    # Handle rho -> 0 carefully: on axis (rho = 0), Brho = 0 by symmetry
    on_axis = rho < 1e-12
    rho_safe = np.where(on_axis, 1.0, rho)
    Brho = (c0 * Z_arr / rho_safe) * (-K_m + ((R * R + rho * rho + Z_arr * Z_arr) / alpha_sq) * E_m)
    Brho = np.where(on_axis, 0.0, Brho)
    Bx = np.sign(X_arr) * Brho

    B_mag = np.hypot(Bx, Bz)
    with np.errstate(divide="ignore", invalid="ignore"):
        B_dir_x = np.where(B_mag > 0.0, Bx / B_mag, 0.0)
        B_dir_y = np.where(B_mag > 0.0, Bz / B_mag, 0.0)

    return MagneticField2DResult(
        X=X_arr,
        Y=Z_arr,
        Bx=Bx,
        By=Bz,
        B_mag=B_mag,
        B_dir_x=B_dir_x,
        B_dir_y=B_dir_y,
        core_mask=core_mask,
        r_min=float(r_min),
        model_name="circular_loop",
    )


def magnetic_field_dipole_2d(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    dipole_moment_am2: float = 10.0,
    orientation_deg: float = 90.0,
    dipole_x: float = 0.0,
    dipole_y: float = 0.0,
    r_min: float = 0.15,
) -> MagneticField2DResult:
    """
    Compute the 2D magnetic vector field of an idealized point magnetic dipole:

        B(r) = (mu_0 / (4 * pi)) * [3 (m . r_hat) r_hat - m] / r^3

    IMPORTANT PHYSICAL NOTE:
    This is an idealized point magnetic dipole model valid in the far field (r >> source dimensions).
    It is NOT an exact model of the interior or near-surface field of a finite physical bar magnet.
    Points inside r < r_min are flagged in `core_mask` and clamped at r_min to prevent the 1/r^3
    point-dipole singularity from dominating the visualization.

    Parameters
    ----------
    X, Y : numpy.ndarray
        2D coordinate arrays in meters (m).
    dipole_moment_am2 : float, default 10.0
        Magnitude of the magnetic dipole moment |m| in A * m^2 (or J/T).
    orientation_deg : float, default 90.0
        Angle of the magnetic dipole vector m in degrees counterclockwise from +x axis
        (90.0 deg points North along +y).
    dipole_x, dipole_y : float, default (0.0, 0.0)
        Dipole position in meters (m).
    r_min : float, default 0.15
        Core exclusion/cutoff radius in meters (m).

    Returns
    -------
    MagneticField2DResult
    """
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    if X_arr.shape != Y_arr.shape or X_arr.size == 0:
        raise ValueError("X and Y must be non-empty arrays with the same shape.")
    if not (
        np.isfinite(dipole_moment_am2)
        and np.isfinite(orientation_deg)
        and np.isfinite(dipole_x)
        and np.isfinite(dipole_y)
    ):
        raise ValueError("Dipole parameters must be finite real numbers.")
    if r_min <= 0.0 or not np.isfinite(r_min):
        raise ValueError(f"r_min must be positive and finite; got {r_min}.")

    theta = math.radians(float(orientation_deg))
    mx = float(dipole_moment_am2) * math.cos(theta)
    my = float(dipole_moment_am2) * math.sin(theta)

    dx = X_arr - float(dipole_x)
    dy = Y_arr - float(dipole_y)
    r = np.hypot(dx, dy)
    core_mask = r < r_min
    r_eff = np.maximum(r, r_min)

    # Unit radial vector r_hat = (dx, dy) / r_eff
    rx_hat = np.where(r > 0.0, dx / r_eff, 0.0)
    ry_hat = np.where(r > 0.0, dy / r_eff, 0.0)

    m_dot_rhat = mx * rx_hat + my * ry_hat
    prefactor = (MU_0 / (4.0 * math.pi)) / (r_eff * r_eff * r_eff)

    Bx = prefactor * (3.0 * m_dot_rhat * rx_hat - mx)
    By = prefactor * (3.0 * m_dot_rhat * ry_hat - my)
    B_mag = np.hypot(Bx, By)

    with np.errstate(divide="ignore", invalid="ignore"):
        B_dir_x = np.where(B_mag > 0.0, Bx / B_mag, 0.0)
        B_dir_y = np.where(B_mag > 0.0, By / B_mag, 0.0)

    return MagneticField2DResult(
        X=X_arr,
        Y=Y_arr,
        Bx=Bx,
        By=By,
        B_mag=B_mag,
        B_dir_x=B_dir_x,
        B_dir_y=B_dir_y,
        core_mask=core_mask,
        r_min=float(r_min),
        model_name="point_magnetic_dipole",
    )

