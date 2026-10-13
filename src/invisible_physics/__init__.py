"""
Invisible Physics — See the Unseen.

An interactive physics visualization and numerical computation package exploring:
- Electrostatics (Coulomb's law, vector electric fields, scalar potentials, numerical gradients)
- Magnetostatics (long straight wire, exact circular current loop, point magnetic dipole)
- Electromagnetic waves (transverse vacuum plane waves, dispersion, Poynting flux)
- Quantitative numerical validation against exact analytical solutions
"""

from __future__ import annotations

__version__ = "0.1.0"

from .constants import (
    C_VACUUM,
    COULOMB_K,
    COULOMB_TO_MICROCOULOMB,
    DEFAULT_CORE_RADIUS_M,
    ELEMENTARY_CHARGE,
    EPSILON_0,
    MICROCOULOMB_TO_COULOMB,
    MICROTESLA_TO_TESLA,
    MILLITESLA_TO_TESLA,
    MU_0,
    NANOCOULOMB_TO_COULOMB,
    TESLA_TO_MICROTESLA,
    TESLA_TO_MILLITESLA,
    Z_0_VACUUM,
)
from .electrostatics import (
    ElectricFieldResult,
    ElectricPotentialResult,
    PointCharge,
    create_coordinate_grid,
    electric_field_2d,
    electric_potential_2d,
    get_preset_charges,
    numerical_field_from_potential,
)
from .magnetism import (
    MagneticField2DResult,
    magnetic_field_circular_loop_on_axis,
    magnetic_field_circular_loop_xz,
    magnetic_field_dipole_2d,
    magnetic_field_straight_wire_1d,
    magnetic_field_straight_wire_2d,
)
from .validation import (
    GradientConsistencyReport,
    find_collinear_zero_field_point,
    validate_potential_gradient_consistency,
    verify_circular_loop_on_axis,
    verify_inverse_square_law,
)
from .visualization import (
    apply_scientific_style,
    create_electromagnetic_wave_animation,
    plot_charge_markers,
    plot_combined_field_and_potential,
    plot_electric_field_heatmap,
    plot_electric_field_streamlines,
    plot_electric_field_vectors,
    plot_electric_potential_landscape,
    plot_electromagnetic_wave_2d_panels,
    plot_electromagnetic_wave_3d,
    plot_interactive_potential_3d_plotly,
    plot_interactive_wave_3d_plotly,
    plot_magnetic_field_2d,
    plot_potential_surface_3d,
)
from .waves import (
    PlaneWaveProfileResult,
    PlaneWaveVacuum,
    evaluate_plane_wave_1d,
)

__all__ = [
    "__version__",
    # Constants
    "C_VACUUM",
    "COULOMB_K",
    "COULOMB_TO_MICROCOULOMB",
    "DEFAULT_CORE_RADIUS_M",
    "ELEMENTARY_CHARGE",
    "EPSILON_0",
    "MICROCOULOMB_TO_COULOMB",
    "MICROTESLA_TO_TESLA",
    "MILLITESLA_TO_TESLA",
    "MU_0",
    "NANOCOULOMB_TO_COULOMB",
    "TESLA_TO_MICROTESLA",
    "TESLA_TO_MILLITESLA",
    "Z_0_VACUUM",
    # Electrostatics
    "ElectricFieldResult",
    "ElectricPotentialResult",
    "PointCharge",
    "create_coordinate_grid",
    "electric_field_2d",
    "electric_potential_2d",
    "get_preset_charges",
    "numerical_field_from_potential",
    # Magnetism
    "MagneticField2DResult",
    "magnetic_field_circular_loop_on_axis",
    "magnetic_field_circular_loop_xz",
    "magnetic_field_dipole_2d",
    "magnetic_field_straight_wire_1d",
    "magnetic_field_straight_wire_2d",
    # Waves
    "PlaneWaveProfileResult",
    "PlaneWaveVacuum",
    "evaluate_plane_wave_1d",
    # Validation
    "GradientConsistencyReport",
    "find_collinear_zero_field_point",
    "validate_potential_gradient_consistency",
    "verify_circular_loop_on_axis",
    "verify_inverse_square_law",
    # Visualization
    "apply_scientific_style",
    "create_electromagnetic_wave_animation",
    "plot_charge_markers",
    "plot_combined_field_and_potential",
    "plot_electric_field_heatmap",
    "plot_electric_field_streamlines",
    "plot_electric_field_vectors",
    "plot_electric_potential_landscape",
    "plot_electromagnetic_wave_2d_panels",
    "plot_electromagnetic_wave_3d",
    "plot_interactive_potential_3d_plotly",
    "plot_interactive_wave_3d_plotly",
    "plot_magnetic_field_2d",
    "plot_potential_surface_3d",
]
