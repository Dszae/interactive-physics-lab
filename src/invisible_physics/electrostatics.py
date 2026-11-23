"""
Electrostatics engine for stationary point charges in two and three dimensions.

Implements Coulomb's law for the electric field E(r) and electrostatic potential V(r)
using fully vectorized NumPy array operations over coordinate grids.

Physical Model & Assumptions:
- Stationary point charges in vacuum (electrostatic regime, curl(E) = 0).
- Linear superposition of fields and potentials from N discrete point charges.
- Potential reference V(infinity) = 0.

Singularity Handling Strategy:
Ideal classical point charges exhibit an unphysical mathematical singularity as |r - r_i| -> 0,
where V(r) -> infinity and |E(r)| -> infinity. Rather than silently returning NaN/Inf or applying
undocumented arbitrary clipping, this module provides two explicit, documented strategies:
1. "core_cutoff" (default): Distances r < r_min are clamped to r_min (equivalent to treating the
   charge outside r_min as a point charge and capping the radial denominator at r_min inside the core),
   while setting the vector direction precisely at r = 0 to zero to avoid 0/0 division.
2. "softening": Plumber softening where |r - r_i| is replaced by sqrt(|r - r_i|^2 + r_min^2),
   creating a smooth, infinitely differentiable potential and field profile inside r_min.
3. "mask": Evaluates the exact point-charge expression and masks points with |r - r_i| < r_min
   as np.nan (plus returning a boolean singularity mask) so downstream analysis can exclude cores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .constants import (
    COULOMB_K,
    DEFAULT_CORE_RADIUS_M,
    MICROCOULOMB_TO_COULOMB,
)

SingularityStrategy = Literal["core_cutoff", "softening", "mask"]


@dataclass(frozen=True)
class PointCharge:
    """
    Represents a stationary point charge in 2D space (z = 0 plane).

    Parameters
    ----------
    q_uc : float
        Charge value in microcoulombs (uC). Can be positive, negative, or zero.
    x : float
        Horizontal position in meters (m).
    y : float
        Vertical position in meters (m).
    label : str, optional
        Optional descriptive label for plots and tables.
    """

    q_uc: float
    x: float
    y: float
    label: str = ""

    def __post_init__(self) -> None:
        if not (np.isfinite(self.q_uc) and np.isfinite(self.x) and np.isfinite(self.y)):
            raise ValueError(
                f"PointCharge parameters must be finite real numbers; got q_uc={self.q_uc}, x={self.x}, y={self.y}."
            )

    @property
    def q_coulombs(self) -> float:
        """Return charge value in SI units (coulombs, C)."""
        return float(self.q_uc) * MICROCOULOMB_TO_COULOMB

    @property
    def position(self) -> NDArray[np.float64]:
        """Return 2D position vector [x, y] in meters (m)."""
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass(frozen=True)
class ElectricFieldResult:
    """
    Container for vectorized 2D electric-field calculation outputs.

    All field components and magnitudes are in SI units: V/m (equivalent to N/C).
    """

    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    Ex: NDArray[np.float64]
    Ey: NDArray[np.float64]
    E_mag: NDArray[np.float64]
    E_dir_x: NDArray[np.float64]
    E_dir_y: NDArray[np.float64]
    core_mask: NDArray[np.bool_]
    r_min: float
    strategy: SingularityStrategy


@dataclass(frozen=True)
class ElectricPotentialResult:
    """
    Container for vectorized 2D electric-potential calculation outputs.

    Potential values V are in SI units: volts (V).
    """

    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    V: NDArray[np.float64]
    core_mask: NDArray[np.bool_]
    r_min: float
    strategy: SingularityStrategy


def _is_whole_number(value: object) -> bool:
    """
    True for a grid resolution that is unambiguously an integer: a Python or NumPy
    integer (np.int64 comes out of array arithmetic and .size), or a float with no
    fractional part. A bool is not a resolution, and 11.5 must not become 11 silently.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, (float, np.floating)) and float(value).is_integer()


def create_coordinate_grid(
    x_lim: tuple[float, float] = (-2.0, 2.0),
    y_lim: tuple[float, float] = (-2.0, 2.0),
    n_points: int | tuple[int, int] = 101,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Create a 2D Cartesian coordinate meshgrid (X, Y) in meters.

    Parameters
    ----------
    x_lim : tuple[float, float]
        (x_min, x_max) domain bounds in meters.
    y_lim : tuple[float, float]
        (y_min, y_max) domain bounds in meters.
    n_points : int or tuple[int, int]
        Grid resolution (nx, ny). Must be at least 5 along each axis.

    Returns
    -------
    X, Y : tuple of 2D numpy.ndarray
        Coordinate arrays of shape (ny, nx) following 'xy' indexing.
    """
    if len(x_lim) != 2 or x_lim[0] >= x_lim[1]:
        raise ValueError(f"x_lim must be (x_min, x_max) with x_min < x_max; got {x_lim}.")
    if len(y_lim) != 2 or y_lim[0] >= y_lim[1]:
        raise ValueError(f"y_lim must be (y_min, y_max) with y_min < y_max; got {y_lim}.")
    if not (np.all(np.isfinite(x_lim)) and np.all(np.isfinite(y_lim))):
        raise ValueError("Grid limits must be finite real numbers.")

    if _is_whole_number(n_points):
        nx = ny = int(n_points)
    elif isinstance(n_points, (tuple, list)) and len(n_points) == 2 and all(map(_is_whole_number, n_points)):
        nx, ny = int(n_points[0]), int(n_points[1])
    else:
        raise ValueError(f"n_points must be an integer or a 2-tuple of integers; got {n_points}.")

    if nx < 5 or ny < 5:
        raise ValueError(f"Grid resolution must be at least 5 in each dimension; got ({nx}, {ny}).")
    if nx > 2000 or ny > 2000:
        raise ValueError(
            f"Grid resolution ({nx}, {ny}) is excessively large; keep dimensions <= 2000 for interactive performance."
        )

    x = np.linspace(float(x_lim[0]), float(x_lim[1]), nx, dtype=np.float64)
    y = np.linspace(float(y_lim[0]), float(y_lim[1]), ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y


def _validate_grid_and_charges(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    charges: Sequence[PointCharge],
    r_min: float,
    strategy: SingularityStrategy,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[PointCharge]]:
    """Validate coordinate arrays, charge sequence, and singularity parameters."""
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)

    if X_arr.shape != Y_arr.shape:
        raise ValueError(f"Coordinate arrays X and Y must have the same shape; got {X_arr.shape} vs {Y_arr.shape}.")
    if X_arr.ndim == 0 or X_arr.size == 0:
        raise ValueError("Coordinate arrays X and Y must be non-empty.")
    if not (np.all(np.isfinite(X_arr)) and np.all(np.isfinite(Y_arr))):
        raise ValueError("Coordinate arrays X and Y must contain only finite values.")

    if r_min <= 0.0 or not np.isfinite(r_min):
        raise ValueError(f"Singularity core radius r_min must be positive and finite; got {r_min}.")

    if strategy not in ("core_cutoff", "softening", "mask"):
        raise ValueError(
            f"Invalid singularity strategy '{strategy}'. Must be 'core_cutoff', 'softening', or 'mask'."
        )

    charge_list = list(charges)
    for idx, c in enumerate(charge_list):
        if not isinstance(c, PointCharge):
            raise TypeError(f"Element {idx} of charges is not a PointCharge instance: {type(c)}.")

    return X_arr, Y_arr, charge_list


def electric_field_2d(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    charges: Sequence[PointCharge],
    r_min: float = DEFAULT_CORE_RADIUS_M,
    strategy: SingularityStrategy = "core_cutoff",
) -> ElectricFieldResult:
    """
    Compute the 2D electrostatic vector field E(r) for multiple point charges using Coulomb's law:

        E(r) = (1 / (4 * pi * epsilon_0)) * sum_i q_i (r - r_i) / |r - r_i|^3

    Calculation is fully vectorized across all grid points and all point charges simultaneously
    using NumPy broadcasting without Python loops over grid coordinates.

    Parameters
    ----------
    X, Y : numpy.ndarray
        Coordinate arrays in meters (m) of identical shape, e.g. (ny, nx).
    charges : Sequence[PointCharge]
        Collection of stationary point charges (q_uc in microcoulombs, positions in meters).
    r_min : float, default 0.05
        Singularity radius in meters (m) defining the core region around each point charge.
    strategy : {"core_cutoff", "softening", "mask"}, default "core_cutoff"
        Documented singularity-handling strategy near |r - r_i| < r_min.

    Returns
    -------
    ElectricFieldResult
        Dataclass containing Ex, Ey, E_mag, unit direction components (E_dir_x, E_dir_y),
        and boolean core_mask indicating points within r_min of any charge.
    """
    X_arr, Y_arr, charge_list = _validate_grid_and_charges(X, Y, charges, r_min, strategy)

    if len(charge_list) == 0:
        zeros = np.zeros_like(X_arr, dtype=np.float64)
        mask = np.zeros_like(X_arr, dtype=np.bool_)
        return ElectricFieldResult(
            X=X_arr,
            Y=Y_arr,
            Ex=zeros,
            Ey=zeros.copy(),
            E_mag=zeros.copy(),
            E_dir_x=zeros.copy(),
            E_dir_y=zeros.copy(),
            core_mask=mask,
            r_min=float(r_min),
            strategy=strategy,
        )

    # Extract charge arrays of shape (N_charges,) and broadcast against grid shape (...,)
    q_c = np.array([c.q_coulombs for c in charge_list], dtype=np.float64)
    qx = np.array([c.x for c in charge_list], dtype=np.float64)
    qy = np.array([c.y for c in charge_list], dtype=np.float64)

    # Shape: (N_charges, *X_arr.shape)
    dx = X_arr[np.newaxis, ...] - qx[(slice(None),) + (np.newaxis,) * X_arr.ndim]
    dy = Y_arr[np.newaxis, ...] - qy[(slice(None),) + (np.newaxis,) * Y_arr.ndim]
    r_sq = dx * dx + dy * dy
    r = np.sqrt(r_sq)

    core_mask = np.any(r < r_min, axis=0)
    q_broad = q_c[(slice(None),) + (np.newaxis,) * X_arr.ndim]

    if strategy == "core_cutoff":
        r_eff = np.maximum(r, r_min)
        inv_r3 = 1.0 / (r_eff * r_eff * r_eff)
        Ex = COULOMB_K * np.sum(q_broad * dx * inv_r3, axis=0)
        Ey = COULOMB_K * np.sum(q_broad * dy * inv_r3, axis=0)
    elif strategy == "softening":
        r_soft = np.sqrt(r_sq + r_min * r_min)
        inv_r3 = 1.0 / (r_soft * r_soft * r_soft)
        Ex = COULOMB_K * np.sum(q_broad * dx * inv_r3, axis=0)
        Ey = COULOMB_K * np.sum(q_broad * dy * inv_r3, axis=0)
    else:  # strategy == "mask"
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_r3 = np.where(r >= r_min, 1.0 / (r * r * r), np.nan)
            Ex = COULOMB_K * np.sum(q_broad * dx * inv_r3, axis=0)
            Ey = COULOMB_K * np.sum(q_broad * dy * inv_r3, axis=0)
        Ex = np.where(core_mask, np.nan, Ex)
        Ey = np.where(core_mask, np.nan, Ey)

    E_mag = np.hypot(Ex, Ey)

    with np.errstate(divide="ignore", invalid="ignore"):
        E_dir_x = np.where(E_mag > 0.0, Ex / E_mag, 0.0)
        E_dir_y = np.where(E_mag > 0.0, Ey / E_mag, 0.0)

    if strategy == "mask":
        E_dir_x = np.where(core_mask, np.nan, E_dir_x)
        E_dir_y = np.where(core_mask, np.nan, E_dir_y)

    return ElectricFieldResult(
        X=X_arr,
        Y=Y_arr,
        Ex=Ex,
        Ey=Ey,
        E_mag=E_mag,
        E_dir_x=E_dir_x,
        E_dir_y=E_dir_y,
        core_mask=core_mask,
        r_min=float(r_min),
        strategy=strategy,
    )


def electric_potential_2d(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    charges: Sequence[PointCharge],
    r_min: float = DEFAULT_CORE_RADIUS_M,
    strategy: SingularityStrategy = "core_cutoff",
) -> ElectricPotentialResult:
    """
    Compute the 2D electrostatic scalar potential V(r) for multiple point charges:

        V(r) = (1 / (4 * pi * epsilon_0)) * sum_i q_i / |r - r_i|

    with reference V(infinity) = 0.

    Parameters
    ----------
    X, Y : numpy.ndarray
        Coordinate arrays in meters (m) of identical shape.
    charges : Sequence[PointCharge]
        Collection of stationary point charges (q_uc in microcoulombs, positions in meters).
    r_min : float, default 0.05
        Singularity radius in meters (m).
    strategy : {"core_cutoff", "softening", "mask"}, default "core_cutoff"
        Documented singularity-handling strategy near |r - r_i| < r_min.

    Returns
    -------
    ElectricPotentialResult
        Dataclass containing scalar potential V in volts (V) and core_mask.
    """
    X_arr, Y_arr, charge_list = _validate_grid_and_charges(X, Y, charges, r_min, strategy)

    if len(charge_list) == 0:
        return ElectricPotentialResult(
            X=X_arr,
            Y=Y_arr,
            V=np.zeros_like(X_arr, dtype=np.float64),
            core_mask=np.zeros_like(X_arr, dtype=np.bool_),
            r_min=float(r_min),
            strategy=strategy,
        )

    q_c = np.array([c.q_coulombs for c in charge_list], dtype=np.float64)
    qx = np.array([c.x for c in charge_list], dtype=np.float64)
    qy = np.array([c.y for c in charge_list], dtype=np.float64)

    dx = X_arr[np.newaxis, ...] - qx[(slice(None),) + (np.newaxis,) * X_arr.ndim]
    dy = Y_arr[np.newaxis, ...] - qy[(slice(None),) + (np.newaxis,) * Y_arr.ndim]
    r_sq = dx * dx + dy * dy
    r = np.sqrt(r_sq)

    core_mask = np.any(r < r_min, axis=0)
    q_broad = q_c[(slice(None),) + (np.newaxis,) * X_arr.ndim]

    if strategy == "core_cutoff":
        r_eff = np.maximum(r, r_min)
        V = COULOMB_K * np.sum(q_broad / r_eff, axis=0)
    elif strategy == "softening":
        r_soft = np.sqrt(r_sq + r_min * r_min)
        V = COULOMB_K * np.sum(q_broad / r_soft, axis=0)
    else:  # strategy == "mask"
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_r = np.where(r >= r_min, 1.0 / r, np.nan)
            V = COULOMB_K * np.sum(q_broad * inv_r, axis=0)
        V = np.where(core_mask, np.nan, V)

    return ElectricPotentialResult(
        X=X_arr,
        Y=Y_arr,
        V=V,
        core_mask=core_mask,
        r_min=float(r_min),
        strategy=strategy,
    )


def numerical_field_from_potential(
    potential_result: ElectricPotentialResult,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the electric field numerically from the negative gradient of scalar potential:

        E_num = -grad(V) = (-dV/dx, -dV/dy)

    Uses second-order central differences via numpy.gradient on the 2D grid.

    Parameters
    ----------
    potential_result : ElectricPotentialResult
        Result of electric_potential_2d evaluated on a 2D regular Cartesian meshgrid.

    Returns
    -------
    Ex_num, Ey_num, E_mag_num : tuple of numpy.ndarray
        Horizontal, vertical, and magnitude of the numerical electric field in V/m.
    """
    X = potential_result.X
    Y = potential_result.Y
    V = potential_result.V

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("numerical_field_from_potential requires 2D meshgrid arrays.")

    x_1d = X[0, :]
    y_1d = Y[:, 0]

    # In 'xy' meshgrid indexing, axis 0 is y and axis 1 is x
    dV_dy, dV_dx = np.gradient(V, y_1d, x_1d, edge_order=2)
    Ex_num = -dV_dx
    Ey_num = -dV_dy
    E_mag_num = np.hypot(Ex_num, Ey_num)
    return Ex_num, Ey_num, E_mag_num


# ---------------------------------------------------------------------------
# Standard Preset Charge Configurations for Reproducible Experiments
# ---------------------------------------------------------------------------

PresetName = Literal[
    "single_positive",
    "single_negative",
    "dipole",
    "two_like_positive",
    "unequal_opposite",
    "symmetric_quadrupole",
    "three_charge_triangle",
]


def get_preset_charges(
    preset: PresetName,
    scale_q_uc: float = 1.0,
    separation_m: float = 1.2,
) -> list[PointCharge]:
    """
    Return a standard preset charge configuration for exploration and validation.

    Parameters
    ----------
    preset : str
        One of:
        - "single_positive": Single +q charge at the origin (0, 0).
        - "single_negative": Single -q charge at the origin (0, 0).
        - "dipole": Equal and opposite charges (+q, -q) separated along the x-axis.
        - "two_like_positive": Two equal positive charges (+q, +q) separated along the x-axis.
        - "unequal_opposite": Asymmetric pair (+2q, -q) separated along the x-axis.
        - "symmetric_quadrupole": Four alternating charges (+q, -q, +q, -q) at the corners of a square.
        - "three_charge_triangle": Equilateral triangle with two +q charges at the base and -2q at apex.
    scale_q_uc : float, default 1.0
        Base charge magnitude in microcoulombs (uC), must be positive.
    separation_m : float, default 1.2
        Characteristic distance between charges in meters (m), must be positive.

    Returns
    -------
    list[PointCharge]
    """
    if scale_q_uc <= 0 or not np.isfinite(scale_q_uc):
        raise ValueError(f"scale_q_uc must be positive and finite; got {scale_q_uc}.")
    if separation_m <= 0 or not np.isfinite(separation_m):
        raise ValueError(f"separation_m must be positive and finite; got {separation_m}.")

    q = float(scale_q_uc)
    a = 0.5 * float(separation_m)

    if preset == "single_positive":
        return [PointCharge(q_uc=+q, x=0.0, y=0.0, label="+q")]
    if preset == "single_negative":
        return [PointCharge(q_uc=-q, x=0.0, y=0.0, label="-q")]
    if preset == "dipole":
        return [
            PointCharge(q_uc=+q, x=-a, y=0.0, label="+q"),
            PointCharge(q_uc=-q, x=+a, y=0.0, label="-q"),
        ]
    if preset == "two_like_positive":
        return [
            PointCharge(q_uc=+q, x=-a, y=0.0, label="+q₁"),
            PointCharge(q_uc=+q, x=+a, y=0.0, label="+q₂"),
        ]
    if preset == "unequal_opposite":
        return [
            PointCharge(q_uc=+2.0 * q, x=-a, y=0.0, label="+2q"),
            PointCharge(q_uc=-1.0 * q, x=+a, y=0.0, label="-q"),
        ]
    if preset == "symmetric_quadrupole":
        return [
            PointCharge(q_uc=+q, x=-a, y=-a, label="+q₁"),
            PointCharge(q_uc=-q, x=+a, y=-a, label="-q₂"),
            PointCharge(q_uc=+q, x=+a, y=+a, label="+q₃"),
            PointCharge(q_uc=-q, x=-a, y=+a, label="-q₄"),
        ]
    if preset == "three_charge_triangle":
        h = a * np.sqrt(3.0)
        return [
            PointCharge(q_uc=+q, x=-a, y=-h / 3.0, label="+q₁"),
            PointCharge(q_uc=+q, x=+a, y=-h / 3.0, label="+q₂"),
            PointCharge(q_uc=-2.0 * q, x=0.0, y=2.0 * h / 3.0, label="-2q"),
        ]

    raise ValueError(f"Unknown preset '{preset}'.")

