"""
Centralized SI physical constants and unit conversion factors for Invisible Physics.

All internal calculations in Invisible Physics use the International System of Units (SI):
- Length: meters (m)
- Time: seconds (s)
- Electric current: amperes (A)
- Electric charge: coulombs (C)
- Electric field: volts per meter (V/m) or newtons per coulomb (N/C)
- Electric potential: volts (V)
- Magnetic flux density (B-field): teslas (T)
- Frequency: hertz (Hz)

Reference:
    NIST CODATA 2022 Internationally Recommended Values of the Fundamental Physical Constants.
    https://physics.nist.gov/cuu/Constants/
"""

from __future__ import annotations

import math

# Speed of light in vacuum (exact by SI definition, m/s)
C_VACUUM: float = 299_792_458.0

# Vacuum electric permittivity epsilon_0 (F/m or C^2 / (N * m^2), CODATA 2022)
EPSILON_0: float = 8.8541878188e-12

# Vacuum magnetic permeability mu_0 (H/m or N / A^2, CODATA 2022)
MU_0: float = 1.25663706127e-6

# Coulomb's constant k_e = 1 / (4 * pi * epsilon_0) (N * m^2 / C^2)
COULOMB_K: float = 1.0 / (4.0 * math.pi * EPSILON_0)

# Characteristic impedance of free space Z_0 = mu_0 * c (ohms)
Z_0_VACUUM: float = MU_0 * C_VACUUM

# Elementary charge e (exact by SI definition, C)
ELEMENTARY_CHARGE: float = 1.602176634e-19

# Unit conversion factors (from user-facing convenience units to SI)
MICROCOULOMB_TO_COULOMB: float = 1e-6
COULOMB_TO_MICROCOULOMB: float = 1e6

NANOCOULOMB_TO_COULOMB: float = 1e-9
COULOMB_TO_NANOCOULOMB: float = 1e9

MILLITESLA_TO_TESLA: float = 1e-3
TESLA_TO_MILLITESLA: float = 1e3

MICROTESLA_TO_TESLA: float = 1e-6
TESLA_TO_MICROTESLA: float = 1e6

# Default singularity softening / core cutoff radius in meters (m)
# Ideal point charges and infinitely thin line/loop currents diverge as r -> 0.
# A documented core cutoff prevents floating-point division by zero and overflow.
DEFAULT_CORE_RADIUS_M: float = 0.05

