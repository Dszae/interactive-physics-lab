"""
Scientific visualization design system for Invisible Physics.

Design Principles:
- Consistent typographic hierarchy and restrained, high-contrast scientific colormaps.
- Diverging colormaps ("RdBu_r") centered at zero for signed scalar potential V(r).
- Sequential perceptually uniform colormaps ("viridis" / "magma") for non-negative magnitudes |E| and |B|.
- Accessible charge and current markers that never rely on color alone:
  * Positive charges (+q): Red circle ('o') with embedded '+' symbol and explicit charge label (in uC).
  * Negative charges (-q): Blue square ('s') with embedded '-' symbol and explicit charge label (in uC).
  * Out-of-page current (+I): Circle with central dot (⊙).
  * Into-page current (-I): Circle with cross (⊗).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize, SymLogNorm, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from .electrostatics import (
    ElectricFieldResult,
    ElectricPotentialResult,
    PointCharge,
)
from .magnetism import MagneticField2DResult
from .waves import PlaneWaveProfileResult

# Color palette constants
COLOR_POS_CHARGE = "#c0392b"  # High-contrast crimson red
COLOR_NEG_CHARGE = "#1f618d"  # High-contrast deep sapphire blue
COLOR_NEUTRAL = "#5d6d7e"
CMAP_MAGNITUDE = "viridis"
CMAP_MAGNETIC = "magma"
CMAP_POTENTIAL = "RdBu_r"


def apply_scientific_style() -> None:
    """
    Configure Matplotlib global rcParams for a clean, publication-grade scientific laboratory layout.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfd",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 10.5,
            "axes.labelweight": "normal",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": "#dce1e7",
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.fontsize": 9.0,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cccccc",
            "font.family": "sans-serif",
        }
    )


def plot_charge_markers(
    ax: Axes,
    charges: Sequence[PointCharge],
    show_labels: bool = True,
    show_core_radius: bool = False,
    r_min: float = 0.05,
    add_legend_entries: bool = True,
) -> None:
    """
    Plot accessible point-charge markers using distinct shapes, colors, internal +/- symbols,
    and explicit text annotations so positive and negative charges are distinguishable even in grayscale.

    - Positive charge (+q): Circle ('o') + '+' glyph + label
    - Negative charge (-q): Square ('s') + '−' glyph + label
    """
    has_pos = False
    has_neg = False

    for idx, c in enumerate(charges):
        if show_core_radius and r_min > 0:
            core_circle = Circle(
                (c.x, c.y),
                radius=r_min,
                fill=False,
                linestyle="--",
                linewidth=1.0,
                edgecolor="#555555",
                alpha=0.75,
                zorder=5,
            )
            ax.add_patch(core_circle)

        if c.q_uc > 0:
            marker = "o"
            color = COLOR_POS_CHARGE
            glyph = "+"
            has_pos = True
        elif c.q_uc < 0:
            marker = "s"
            color = COLOR_NEG_CHARGE
            glyph = "−"
            has_neg = True
        else:
            marker = "^"
            color = COLOR_NEUTRAL
            glyph = "0"

        ax.plot(
            c.x,
            c.y,
            marker=marker,
            markersize=13,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.6,
            linestyle="None",
            zorder=7,
        )
        ax.text(
            c.x,
            c.y,
            glyph,
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )

        if show_labels:
            tag = c.label if c.label else f"q{idx + 1}"
            label_str = f"{tag} ({c.q_uc:+.1f} μC)"
            ax.annotate(
                label_str,
                xy=(c.x, c.y),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color="#1a1a1a",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.88,
                    "lw": 0.9,
                },
                zorder=9,
            )

    if add_legend_entries and (has_pos or has_neg):
        handles = []
        if has_pos:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Positive charge (+q, circle)",
                    markerfacecolor=COLOR_POS_CHARGE,
                    markeredgecolor="white",
                    markersize=9,
                )
            )
        if has_neg:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    label="Negative charge (−q, square)",
                    markerfacecolor=COLOR_NEG_CHARGE,
                    markeredgecolor="white",
                    markersize=9,
                )
            )
        if handles:
            ax.legend(handles=handles, loc="upper right")


def _safe_log_norm(values: np.ndarray, dynamic_range: float = 1e4) -> LogNorm:
    """Create a well-behaved LogNorm for strictly positive field magnitudes."""
    finite_pos = values[np.isfinite(values) & (values > 0.0)]
    if finite_pos.size == 0:
        return LogNorm(vmin=1e-1, vmax=1e3)
    vmax = float(np.percentile(finite_pos, 99.0))
    vmin = max(float(np.percentile(finite_pos, 2.0)), vmax / dynamic_range)
    if vmin >= vmax:
        vmin = vmax * 0.1 if vmax > 0 else 1.0
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_electric_field_vectors(
    field_result: ElectricFieldResult,
    charges: Sequence[PointCharge],
    ax: Axes | None = None,
    stride: int = 5,
    title: str = "Electric Field Vector Map (Unit Direction Colored by |E|)",
    show_labels: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a 2D electric-field vector (quiver) diagram.

    Because Coulomb's 1/r^2 field spans several orders of magnitude across a grid, raw arrow
    lengths either explode near charges or vanish elsewhere. We plot unit direction vectors
    (E_dir_x, E_dir_y) colored by |E| (V/m) on a logarithmic colorbar so both direction and
    magnitude are accurately and legibly communicated.
    """
    apply_scientific_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
    else:
        fig = ax.figure  # type: ignore[assignment]

    s = max(1, int(stride))
    Xs = field_result.X[::s, ::s]
    Ys = field_result.Y[::s, ::s]
    Ux = field_result.E_dir_x[::s, ::s]
    Uy = field_result.E_dir_y[::s, ::s]
    Ms = field_result.E_mag[::s, ::s]

    norm = _safe_log_norm(field_result.E_mag)
    q_plot = ax.quiver(
        Xs,
        Ys,
        Ux,
        Uy,
        Ms,
        cmap=CMAP_MAGNITUDE,
        norm=norm,
        pivot="mid",
        scale=26,
        width=0.0042,
        headwidth=3.8,
        headlength=4.5,
        zorder=3,
    )
    cbar = fig.colorbar(q_plot, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Electric Field Magnitude |E| (V/m, log scale)")

    plot_charge_markers(
        ax,
        charges,
        show_labels=show_labels,
        show_core_radius=True,
        r_min=field_result.r_min,
    )

    ax.set_title(title)
    ax.set_xlabel("Horizontal Position x (m)")
    ax.set_ylabel("Vertical Position y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(field_result.X)), float(np.max(field_result.X)))
    ax.set_ylim(float(np.min(field_result.Y)), float(np.max(field_result.Y)))
    return fig, ax


def plot_electric_field_streamlines(
    field_result: ElectricFieldResult,
    charges: Sequence[PointCharge],
    ax: Axes | None = None,
    density: float = 1.35,
    show_heatmap_background: bool = True,
    title: str = "Electric Field Streamlines & Magnitude |E|",
    show_labels: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot continuous electric-field streamlines overlaid on an electric-field magnitude heatmap.
    """
    apply_scientific_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
    else:
        fig = ax.figure  # type: ignore[assignment]

    norm = _safe_log_norm(field_result.E_mag)
    if show_heatmap_background:
        mesh = ax.pcolormesh(
            field_result.X,
            field_result.Y,
            np.maximum(field_result.E_mag, norm.vmin),
            cmap=CMAP_MAGNITUDE,
            norm=norm,
            shading="auto",
            alpha=0.82,
            zorder=1,
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Electric Field Magnitude |E| (V/m, log scale)")

        stream_color = "#ffffff"
    else:
        stream_color = "#1f3a52"

    # Replace any NaNs (if strategy="mask") with 0 for streamplot integrator safety
    Ex_clean = np.nan_to_num(field_result.Ex, nan=0.0)
    Ey_clean = np.nan_to_num(field_result.Ey, nan=0.0)

    if np.any( np.hypot(Ex_clean, Ey_clean) > 0.0 ):
        ax.streamplot(
            field_result.X,
            field_result.Y,
            Ex_clean,
            Ey_clean,
            color=stream_color,
            linewidth=1.1,
            density=density,
            arrowsize=1.15,
            arrowstyle="->",
            zorder=3,
        )

    plot_charge_markers(
        ax,
        charges,
        show_labels=show_labels,
        show_core_radius=True,
        r_min=field_result.r_min,
    )

    ax.set_title(title)
    ax.set_xlabel("Horizontal Position x (m)")
    ax.set_ylabel("Vertical Position y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(field_result.X)), float(np.max(field_result.X)))
    ax.set_ylim(float(np.min(field_result.Y)), float(np.max(field_result.Y)))
    return fig, ax


def plot_electric_field_heatmap(
    field_result: ElectricFieldResult,
    charges: Sequence[PointCharge],
    ax: Axes | None = None,
    show_iso_contours: bool = True,
    title: str = "Electric Field Magnitude Heatmap |E(r)|",
    show_labels: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a dedicated 2D heatmap of electric-field magnitude |E(r)| (in V/m on a logarithmic scale)
    with optional iso-magnitude contour lines and accessible charge-position markers.
    """
    apply_scientific_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
    else:
        fig = ax.figure  # type: ignore[assignment]

    norm = _safe_log_norm(field_result.E_mag)
    e_clamped = np.maximum(np.nan_to_num(field_result.E_mag, nan=norm.vmin), norm.vmin)
    mesh = ax.pcolormesh(
        field_result.X,
        field_result.Y,
        e_clamped,
        cmap=CMAP_MAGNITUDE,
        norm=norm,
        shading="auto",
        zorder=1,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Electric Field Magnitude |E| (V/m, log scale)")

    if show_iso_contours and norm.vmin is not None and norm.vmax is not None and norm.vmax > norm.vmin:
        log_levels = np.logspace(np.log10(norm.vmin * 2.0), np.log10(norm.vmax * 0.5), 6)
        ax.contour(
            field_result.X,
            field_result.Y,
            e_clamped,
            levels=log_levels,
            colors="white",
            linewidths=0.75,
            linestyles=":",
            alpha=0.65,
            zorder=2,
        )

    plot_charge_markers(
        ax,
        charges,
        show_labels=show_labels,
        show_core_radius=True,
        r_min=field_result.r_min,
    )

    ax.set_title(title)
    ax.set_xlabel("Horizontal Position x (m)")
    ax.set_ylabel("Vertical Position y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(field_result.X)), float(np.max(field_result.X)))
    ax.set_ylim(float(np.min(field_result.Y)), float(np.max(field_result.Y)))
    return fig, ax


def plot_electric_potential_landscape(
    potential_result: ElectricPotentialResult,
    charges: Sequence[PointCharge],
    ax: Axes | None = None,
    n_contours: int = 15,
    title: str = "Electric Potential V(r) & Equipotential Contours",
    show_labels: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot the 2D scalar electric potential V(r) in kilovolts (kV) using a symmetric diverging
    colormap centered at V = 0 V, overlaid with labeled equipotential contour lines.
    """
    apply_scientific_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
    else:
        fig = ax.figure  # type: ignore[assignment]

    # Convert Volts to kilovolts (kV) for clean axis/colorbar typography
    V_kv = potential_result.V * 1e-3
    finite_v = V_kv[np.isfinite(V_kv)]
    max_abs = float(np.percentile(np.abs(finite_v), 97.5)) if finite_v.size > 0 else 10.0
    if max_abs <= 0.0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    mesh = ax.pcolormesh(
        potential_result.X,
        potential_result.Y,
        np.clip(V_kv, -max_abs, max_abs),
        cmap=CMAP_POTENTIAL,
        norm=norm,
        shading="auto",
        zorder=1,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Electric Potential V (kV, 1 kV = 10³ V)")

    # Overlay equipotential contours
    if np.ptp(finite_v) > 1e-9:
        levels = np.linspace(-0.85 * max_abs, 0.85 * max_abs, n_contours)
        cs = ax.contour(
            potential_result.X,
            potential_result.Y,
            V_kv,
            levels=levels,
            colors="#222222",
            linewidths=0.85,
            linestyles="solid",
            alpha=0.75,
            zorder=3,
        )
        ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.1f kV")

    plot_charge_markers(
        ax,
        charges,
        show_labels=show_labels,
        show_core_radius=True,
        r_min=potential_result.r_min,
    )

    ax.set_title(title)
    ax.set_xlabel("Horizontal Position x (m)")
    ax.set_ylabel("Vertical Position y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(potential_result.X)), float(np.max(potential_result.X)))
    ax.set_ylim(float(np.min(potential_result.Y)), float(np.max(potential_result.Y)))
    return fig, ax


def plot_combined_field_and_potential(
    field_result: ElectricFieldResult,
    potential_result: ElectricPotentialResult,
    charges: Sequence[PointCharge],
    ax: Axes | None = None,
    n_contours: int = 13,
    stream_density: float = 1.2,
    title: str = "Orthogonality of Electric Field Lines and Equipotential Contours",
    show_labels: bool = True,
) -> tuple[Figure, Axes]:
    """
    Simultaneously display the scalar electric potential V(r), equipotential contours (dashed),
    and electric-field streamlines (solid arrows) to demonstrate that E = -grad(V) is everywhere
    perpendicular to equipotential surfaces and points toward decreasing potential.
    """
    apply_scientific_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
    else:
        fig = ax.figure  # type: ignore[assignment]

    V_kv = potential_result.V * 1e-3
    finite_v = V_kv[np.isfinite(V_kv)]
    max_abs = float(np.percentile(np.abs(finite_v), 97.0)) if finite_v.size > 0 else 10.0
    if max_abs <= 0.0:
        max_abs = 1.0

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    mesh = ax.pcolormesh(
        potential_result.X,
        potential_result.Y,
        np.clip(V_kv, -max_abs, max_abs),
        cmap=CMAP_POTENTIAL,
        norm=norm,
        shading="auto",
        alpha=0.80,
        zorder=1,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Electric Potential V (kV)")

    if np.ptp(finite_v) > 1e-9:
        levels = np.linspace(-0.85 * max_abs, 0.85 * max_abs, n_contours)
        ax.contour(
            potential_result.X,
            potential_result.Y,
            V_kv,
            levels=levels,
            colors="#111111",
            linewidths=1.0,
            linestyles="--",
            alpha=0.85,
            zorder=2,
        )

    Ex_clean = np.nan_to_num(field_result.Ex, nan=0.0)
    Ey_clean = np.nan_to_num(field_result.Ey, nan=0.0)
    if np.any(np.hypot(Ex_clean, Ey_clean) > 0.0):
        ax.streamplot(
            field_result.X,
            field_result.Y,
            Ex_clean,
            Ey_clean,
            color="#1b2631",
            linewidth=1.15,
            density=stream_density,
            arrowsize=1.15,
            arrowstyle="->",
            zorder=4,
        )

    plot_charge_markers(
        ax,
        charges,
        show_labels=show_labels,
        show_core_radius=True,
        r_min=field_result.r_min,
    )

    ax.set_title(title)
    ax.set_xlabel("Horizontal Position x (m)")
    ax.set_ylabel("Vertical Position y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(field_result.X)), float(np.max(field_result.X)))
    ax.set_ylim(float(np.min(field_result.Y)), float(np.max(field_result.Y)))
    return fig, ax


def plot_potential_surface_3d(
    potential_result: ElectricPotentialResult,
    title: str = "3D Electrostatic Potential Landscape V(x, y)",
    elev: float = 28.0,
    azim: float = -55.0,
) -> tuple[Figure, Axes]:
    """
    Render a 3D surface plot of the electric potential landscape V(x, y) in kilovolts (kV),
    illustrating potential hills (+q) and potential wells (-q).
    """
    apply_scientific_style()
    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    V_kv = potential_result.V * 1e-3
    finite_v = V_kv[np.isfinite(V_kv)]
    max_abs = float(np.percentile(np.abs(finite_v), 98.0)) if finite_v.size > 0 else 10.0
    if max_abs <= 0.0:
        max_abs = 1.0

    V_clipped = np.clip(np.nan_to_num(V_kv, nan=0.0), -max_abs, max_abs)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    surf = ax.plot_surface(
        potential_result.X,
        potential_result.Y,
        V_clipped,
        cmap=CMAP_POTENTIAL,
        norm=norm,
        rcount=75,
        ccount=75,
        linewidth=0.15,
        edgecolor="#444444",
        antialiased=True,
        alpha=0.92,
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.68, pad=0.08)
    cbar.set_label("Electric Potential V (kV)")

    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("V (kV)")
    ax.set_zlim(-max_abs * 1.05, max_abs * 1.05)
    ax.view_init(elev=elev, azim=azim)
    return fig, ax


def plot_magnetic_field_2d(
    mag_result: MagneticField2DResult,
    ax: Axes | None = None,
    title: str = "Magnetic Flux Density |B| and Field Lines",
    current_amps: float | None = None,
    loop_radius_m: float | None = None,
    dipole_orientation_deg: float | None = None,
    stream_density: float = 1.25,
) -> tuple[Figure, Axes]:
    """
    Plot 2D magnetic flux density magnitude |B| (in microteslas, uT) alongside magnetic field lines
    and source geometry annotations (straight wire ⊙/⊗, circular loop cross-sections, or magnetic dipole).
    """
    apply_scientific_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
    else:
        fig = ax.figure  # type: ignore[assignment]

    # Convert Tesla to microtesla (uT) for intuitive laboratory readout
    B_ut = mag_result.B_mag * 1e6
    norm = _safe_log_norm(B_ut, dynamic_range=1e3)

    mesh = ax.pcolormesh(
        mag_result.X,
        mag_result.Y,
        np.maximum(B_ut, norm.vmin),
        cmap=CMAP_MAGNETIC,
        norm=norm,
        shading="auto",
        alpha=0.85,
        zorder=1,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Magnetic Field Magnitude |B| (μT, 1 μT = 10⁻⁶ T)")

    Bx_clean = np.nan_to_num(mag_result.Bx, nan=0.0)
    By_clean = np.nan_to_num(mag_result.By, nan=0.0)
    if np.any(np.hypot(Bx_clean, By_clean) > 0.0):
        ax.streamplot(
            mag_result.X,
            mag_result.Y,
            Bx_clean,
            By_clean,
            color="#ffffff",
            linewidth=1.15,
            density=stream_density,
            arrowsize=1.2,
            arrowstyle="->",
            zorder=3,
        )

    # Draw model-specific physical source indicators
    if mag_result.model_name == "straight_wire":
        symbol = "⊙" if (current_amps is None or current_amps >= 0) else "⊗"
        dir_label = "Out of page (+z)" if (current_amps is None or current_amps >= 0) else "Into page (−z)"
        core = Circle((0.0, 0.0), radius=mag_result.r_min, facecolor="white", edgecolor="#111111", lw=1.5, zorder=6)
        ax.add_patch(core)
        ax.text(0.0, 0.0, symbol, fontsize=14, fontweight="bold", ha="center", va="center", color="#111111", zorder=7)
        ax.annotate(
            f"Wire I = {current_amps if current_amps is not None else 0:+.1f} A ({dir_label})",
            xy=(0.0, 0.0),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#333333", "alpha": 0.9},
            zorder=8,
        )
        ax.set_xlabel("Horizontal Position x (m)")
        ax.set_ylabel("Vertical Position y (m)")

    elif mag_result.model_name == "circular_loop" and loop_radius_m is not None:
        R = float(loop_radius_m)
        # For I > 0 (CCW in x-y plane viewed from +z): at x = -R, current exits page (⊙); at x = +R, enters page (⊗)
        left_sym = "⊙" if (current_amps is None or current_amps >= 0) else "⊗"
        right_sym = "⊗" if (current_amps is None or current_amps >= 0) else "⊙"
        for x_pos, sym, tag in [(-R, left_sym, "Wire (−R, 0)"), (+R, right_sym, "Wire (+R, 0)")]:
            c_patch = Circle(
                (x_pos, 0.0), radius=mag_result.r_min, facecolor="white", edgecolor="#111111", lw=1.5, zorder=6
            )
            ax.add_patch(c_patch)
            ax.text(x_pos, 0.0, sym, fontsize=12, fontweight="bold", ha="center", va="center", zorder=7)
            ax.annotate(
                f"{tag} {sym}",
                xy=(x_pos, 0.0),
                xytext=(0, -18),
                textcoords="offset points",
                ha="center",
                fontsize=8.0,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#333333", "alpha": 0.9},
                zorder=8,
            )
        ax.set_xlabel("Transverse Coordinate x (m)")
        ax.set_ylabel("Axial Coordinate z (m)")

    elif mag_result.model_name == "point_magnetic_dipole":
        core = Circle(
            (0.0, 0.0),
            radius=mag_result.r_min,
            facecolor="#2c3e50",
            edgecolor="white",
            linestyle="--",
            lw=1.4,
            alpha=0.85,
            zorder=6,
        )
        ax.add_patch(core)
        ax.text(
            0.0,
            0.0,
            "m",
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
        )
        ax.annotate(
            f"Point Dipole Core (r_min = {mag_result.r_min:.2f} m)\nFar-field idealization only",
            xy=(0.0, 0.0),
            xytext=(12, -18),
            textcoords="offset points",
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#333333", "alpha": 0.9},
            zorder=8,
        )
        ax.set_xlabel("Horizontal Position x (m)")
        ax.set_ylabel("Vertical Position y (m)")

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(mag_result.X)), float(np.max(mag_result.X)))
    ax.set_ylim(float(np.min(mag_result.Y)), float(np.max(mag_result.Y)))
    return fig, ax


def plot_electromagnetic_wave_3d(
    profile: PlaneWaveProfileResult,
    title: str | None = None,
    n_vectors: int = 36,
) -> tuple[Figure, Axes]:
    """
    Render a 3D perspective plot of a transverse plane electromagnetic wave propagating along x,
    highlighting the mutually perpendicular oscillations of E_y(x, t) (red) and B_z(x, t) (blue)
    so both fields can be visually compared on normalized geometric axes while preserving exact
    SI amplitudes in the legend and axis labels.
    """
    apply_scientific_style()
    fig = plt.figure(figsize=(9.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    x = profile.x
    e0 = profile.wave.e0_v_per_m if profile.wave.e0_v_per_m > 0 else 1.0
    b0 = profile.wave.b0_tesla if profile.wave.b0_tesla > 0 else 1.0

    # Normalize to [-1, 1] for joint 3D geometric display while reporting exact SI amplitudes
    ey_norm = profile.Ey / e0
    bz_norm = profile.Bz / b0
    zeros = np.zeros_like(x)

    # Propagation axis line
    ax.plot(x, zeros, zeros, color="#222222", lw=1.6, label=f"Propagation Axis ({profile.wave.propagation_axis})")

    # E-field envelope in x-y plane (z = 0)
    ax.plot(
        x,
        ey_norm,
        zeros,
        color=COLOR_POS_CHARGE,
        lw=2.2,
        label=f"Electric Field E_y (E₀ = {profile.wave.e0_v_per_m:.2g} V/m)",
    )
    # B-field envelope in x-z plane (y = 0)
    ax.plot(
        x,
        zeros,
        bz_norm,
        color=COLOR_NEG_CHARGE,
        lw=2.2,
        label=f"Magnetic Field B_z (B₀ = {profile.wave.b0_tesla * 1e9:.3g} nT)",
    )

    # Draw discrete field vectors from the x-axis to the wave envelopes
    step = max(1, len(x) // n_vectors)
    for xi, eyi, bzi in zip(x[::step], ey_norm[::step], bz_norm[::step]):
        ax.plot([xi, xi], [0.0, eyi], [0.0, 0.0], color=COLOR_POS_CHARGE, alpha=0.55, lw=1.1)
        ax.plot([xi, xi], [0.0, 0.0], [0.0, bzi], color=COLOR_NEG_CHARGE, alpha=0.55, lw=1.1)

    if title is None:
        title = (
            f"Transverse Plane Electromagnetic Wave in Vacuum\n"
            f"f = {profile.wave.frequency_hz:.3g} Hz, λ = {profile.wave.wavelength_m:.3g} m, "
            f"|E|/|B| = c = {profile.wave.wave_speed_m_per_s:.6g} m/s"
        )
    ax.set_title(title)
    ax.set_xlabel("Propagation Coordinate x (m)")
    ax.set_ylabel("Normalized E_y / E₀ (y-axis)")
    ax.set_zlabel("Normalized B_z / B₀ (z-axis)")
    ax.set_ylim(-1.25, 1.25)
    ax.set_zlim(-1.25, 1.25)
    ax.view_init(elev=22.0, azim=-58.0)
    ax.legend(loc="upper right")
    return fig, ax


def plot_electromagnetic_wave_2d_panels(
    profile: PlaneWaveProfileResult,
) -> tuple[Figure, Sequence[Axes]]:
    """
    Plot synchronized 2D scientific panels of a vacuum plane wave along propagation coordinate x:
    1. Electric field E_y(x, t) in V/m.
    2. Magnetic field B_z(x, t) in nanoteslas (nT).
    3. Instantaneous Poynting energy flux S_x(x, t) in W/m^2 and time-averaged intensity <S>.
    """
    apply_scientific_style()
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.8), sharex=True, constrained_layout=True)

    x = profile.x
    # Panel 1: Electric field Ey
    axes[0].plot(x, profile.Ey, color=COLOR_POS_CHARGE, lw=2.0, label="E_y(x, t)")
    axes[0].axhline(0.0, color="#555555", lw=0.8, ls="--")
    axes[0].set_ylabel("E_y (V/m)")
    axes[0].set_title(
        f"Plane Wave Spatial Profile at t = {profile.t_s:.2e} s "
        f"(f = {profile.wave.frequency_hz:.3g} Hz, λ = {profile.wave.wavelength_m:.3g} m)"
    )
    axes[0].legend(loc="upper right")

    # Panel 2: Magnetic field Bz in nT
    axes[1].plot(x, profile.Bz * 1e9, color=COLOR_NEG_CHARGE, lw=2.0, label="B_z(x, t) = ±E_y / c")
    axes[1].axhline(0.0, color="#555555", lw=0.8, ls="--")
    axes[1].set_ylabel("B_z (nT)")
    axes[1].legend(loc="upper right")

    # Panel 3: Poynting flux Sx
    axes[2].plot(x, profile.Sx, color="#1e8449", lw=2.0, label="Instantaneous S_x(x, t) = (E × B)_x / μ₀")
    s_avg = profile.wave.time_averaged_intensity_w_per_m2 * (1.0 if profile.wave.propagation_axis == "+x" else -1.0)
    axes[2].axhline(s_avg, color="#b7950b", lw=1.4, ls="--", label=f"Time-Averaged ⟨S_x⟩ = {s_avg:.3g} W/m²")
    axes[2].set_ylabel("S_x (W/m²)")
    axes[2].set_xlabel("Propagation Coordinate x (m)")
    axes[2].legend(loc="upper right")

    return fig, axes


def create_electromagnetic_wave_animation(
    wave: "PlaneWaveVacuum",
    n_wavelengths: float = 2.0,
    n_points: int = 250,
    n_frames: int = 24,
    interval_ms: int = 60,
) -> tuple[Figure, mpl.animation.FuncAnimation]:
    """
    Create a reliable 2D Matplotlib FuncAnimation showing synchronized propagation of
    E_y(x, t) (V/m) and B_z(x, t) (nT) over one full oscillation period T = 1 / f.

    Can be rendered inline in Jupyter notebooks via `HTML(anim.to_jshtml())`.
    """
    from matplotlib.animation import FuncAnimation
    from .waves import evaluate_plane_wave_1d

    apply_scientific_style()
    x = np.linspace(0.0, n_wavelengths * wave.wavelength_m, n_points)
    prof0 = evaluate_plane_wave_1d(wave, x, t_s=0.0)

    fig, (ax_e, ax_b) = plt.subplots(2, 1, figsize=(7.8, 4.8), sharex=True, constrained_layout=True)

    (line_e,) = ax_e.plot(x, prof0.Ey, color=COLOR_POS_CHARGE, lw=2.1, label="Electric Field E_y(x, t)")
    ax_e.axhline(0.0, color="#555555", lw=0.8, ls="--")
    ax_e.set_ylim(-1.25 * max(wave.e0_v_per_m, 1.0), 1.25 * max(wave.e0_v_per_m, 1.0))
    ax_e.set_ylabel("E_y (V/m)")
    ax_e.set_title(
        f"Propagating Plane Wave in Vacuum (f = {wave.frequency_hz:.3g} Hz, λ = {wave.wavelength_m:.3g} m)"
    )
    ax_e.legend(loc="upper right")

    b_max_nt = max(wave.b0_tesla * 1e9, 1.0)
    (line_b,) = ax_b.plot(x, prof0.Bz * 1e9, color=COLOR_NEG_CHARGE, lw=2.1, label="Magnetic Field B_z(x, t)")
    ax_b.axhline(0.0, color="#555555", lw=0.8, ls="--")
    ax_b.set_ylim(-1.25 * b_max_nt, 1.25 * b_max_nt)
    ax_b.set_ylabel("B_z (nT)")
    ax_b.set_xlabel("Propagation Coordinate x (m)")
    ax_b.legend(loc="upper right")

    def _update(frame_idx: int):
        t_val = (frame_idx / float(n_frames)) * wave.period_s
        prof = evaluate_plane_wave_1d(wave, x, t_s=t_val)
        line_e.set_ydata(prof.Ey)
        line_b.set_ydata(prof.Bz * 1e9)
        return line_e, line_b

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=interval_ms, blit=True)
    return fig, anim


def plot_interactive_potential_3d_plotly(
    potential_result: ElectricPotentialResult,
    charges: Sequence[PointCharge] | None = None,
    title: str = "Interactive 3D Electrostatic Potential Landscape V(x, y) — Drag to Rotate in 360°",
):
    """
    Create an interactive, mouse-rotatable 3D WebGL surface and contour projection using Plotly.
    Users can left-click and drag to rotate in 3D, scroll to zoom, and hover to inspect exact
    (x, y, V) coordinates in kilovolts (kV).
    """
    import plotly.graph_objects as go

    V_kv = np.nan_to_num(potential_result.V * 1e-3, nan=0.0)
    finite_v = V_kv[np.isfinite(V_kv)]
    max_abs = float(np.percentile(np.abs(finite_v), 98.5)) if finite_v.size > 0 else 10.0
    if max_abs <= 0.0:
        max_abs = 1.0
    V_clipped = np.clip(V_kv, -max_abs, max_abs)

    fig = go.Figure(
        data=[
            go.Surface(
                x=potential_result.X,
                y=potential_result.Y,
                z=V_clipped,
                colorscale="RdBu_r",
                cmin=-max_abs,
                cmax=max_abs,
                colorbar=dict(title="V (kV)"),
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="limegreen",
                        project_z=True,
                    )
                ),
                hovertemplate="x: %{x:.2f} m<br>y: %{y:.2f} m<br>Potential V: %{z:.2f} kV<extra></extra>",
            )
        ]
    )

    if charges:
        qx = [c.x for c in charges]
        qy = [c.y for c in charges]
        qz = [max_abs if c.q_uc > 0 else (-max_abs if c.q_uc < 0 else 0.0) for c in charges]
        qtext = [f"{c.label or 'q'} ({c.q_uc:+.1f} μC)" for c in charges]
        fig.add_trace(
            go.Scatter3d(
                x=qx,
                y=qy,
                z=qz,
                mode="markers+text",
                text=qtext,
                textposition="top center",
                marker=dict(
                    size=7,
                    color=[COLOR_POS_CHARGE if c.q_uc >= 0 else COLOR_NEG_CHARGE for c in charges],
                    line=dict(color="white", width=2),
                ),
                name="Point Charges",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=860,
        height=620,
        margin=dict(l=15, r=15, t=55, b=15),
        scene=dict(
            xaxis_title="Horizontal x (m)",
            yaxis_title="Vertical y (m)",
            zaxis_title="Potential V (kV)",
            zaxis=dict(range=[-max_abs * 1.15, max_abs * 1.15]),
            camera=dict(eye=dict(x=1.45, y=-1.45, z=0.95)),
        ),
    )
    return fig


def plot_interactive_wave_3d_plotly(
    profile: PlaneWaveProfileResult,
    title: str | None = None,
):
    """
    Create an interactive, mouse-rotatable 3D Plotly visualization of a transverse plane
    electromagnetic wave showing E_y(x, t) and B_z(x, t) orthogonal vectors in 3D space.
    """
    import plotly.graph_objects as go

    x = profile.x
    e0 = profile.wave.e0_v_per_m if profile.wave.e0_v_per_m > 0 else 1.0
    b0 = profile.wave.b0_tesla if profile.wave.b0_tesla > 0 else 1.0
    ey_norm = profile.Ey / e0
    bz_norm = profile.Bz / b0
    zeros = np.zeros_like(x)

    if title is None:
        title = (
            f"Interactive 3D Plane Electromagnetic Wave (Drag to Rotate in 360°)<br>"
            f"<sup>f = {profile.wave.frequency_hz * 1e-6:.1f} MHz, λ = {profile.wave.wavelength_m:.2f} m, "
            f"E₀ = {profile.wave.e0_v_per_m:.1f} V/m, B₀ = {profile.wave.b0_tesla * 1e9:.1f} nT</sup>"
        )

    fig = go.Figure()

    # Propagation axis
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=zeros,
            z=zeros,
            mode="lines",
            line=dict(color="#222222", width=5),
            name=f"Propagation Axis ({profile.wave.propagation_axis})",
        )
    )

    # E-field curve (x-y plane)
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=ey_norm,
            z=zeros,
            mode="lines",
            line=dict(color=COLOR_POS_CHARGE, width=6),
            name=f"Electric Field E_y (E₀ = {profile.wave.e0_v_per_m:.1f} V/m)",
            customdata=profile.Ey,
            hovertemplate="x: %{x:.2f} m<br>E_y: %{customdata:.2f} V/m<extra></extra>",
        )
    )

    # B-field curve (x-z plane)
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=zeros,
            z=bz_norm,
            mode="lines",
            line=dict(color=COLOR_NEG_CHARGE, width=6),
            name=f"Magnetic Field B_z (B₀ = {profile.wave.b0_tesla * 1e9:.1f} nT)",
            customdata=profile.Bz * 1e9,
            hovertemplate="x: %{x:.2f} m<br>B_z: %{customdata:.2f} nT<extra></extra>",
        )
    )

    # Add orthogonal stem lines for E and B vectors
    step = max(1, len(x) // 40)
    ex_stems_x, ex_stems_y, ex_stems_z = [], [], []
    bx_stems_x, bx_stems_y, bx_stems_z = [], [], []
    for xi, eyi, bzi in zip(x[::step], ey_norm[::step], bz_norm[::step]):
        ex_stems_x.extend([xi, xi, None])
        ex_stems_y.extend([0.0, eyi, None])
        ex_stems_z.extend([0.0, 0.0, None])

        bx_stems_x.extend([xi, xi, None])
        bx_stems_y.extend([0.0, 0.0, None])
        bx_stems_z.extend([0.0, bzi, None])

    fig.add_trace(
        go.Scatter3d(
            x=ex_stems_x,
            y=ex_stems_y,
            z=ex_stems_z,
            mode="lines",
            line=dict(color=COLOR_POS_CHARGE, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=bx_stems_x,
            y=bx_stems_y,
            z=bx_stems_z,
            mode="lines",
            line=dict(color=COLOR_NEG_CHARGE, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=860,
        height=580,
        margin=dict(l=15, r=15, t=65, b=15),
        scene=dict(
            xaxis_title="Propagation x (m)",
            yaxis_title="E_y / E₀ (Electric)",
            zaxis_title="B_z / B₀ (Magnetic)",
            yaxis=dict(range=[-1.3, 1.3]),
            zaxis=dict(range=[-1.3, 1.3]),
            camera=dict(eye=dict(x=1.35, y=-1.55, z=0.85)),
        ),
    )
    return fig


