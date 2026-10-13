# Scientific Assumptions, Idealizations, and Numerical Limitations

Scientific integrity requires distinguishing exact physical laws from mathematical idealizations and numerical approximations. Below are the explicit assumptions and limitations of the models in **Invisible Physics**.

---

## 1. Classical Point-Charge & Thin-Filament Singularities

### The Idealization
Classical electrostatics models localized charges as zero-radius mathematical points ($q_i$ at $\mathbf{r}_i$), and magnetostatics models thin wires as zero-thickness line or loop filaments. Analytically, as evaluation distance $r = \|\mathbf{r} - \mathbf{r}_i\| \to 0$:
- Point-charge potential diverges as $1/r$ and electric field as $1/r^2$.
- Point-dipole magnetic field diverges as $1/r^3$.
- Line-current magnetic field diverges as $1/r$.

In reality, macroscopic conductors have finite radii, and at subatomic scales quantum electrodynamics (QED) and finite charge distributions replace classical point singularities.

### Our Numerical Handling
To prevent `ZeroDivisionError`, `NaN` propagation, or undocumented clipping, `invisible_physics` implements three explicit, user-selectable strategies controlled by `r_min` (default $0.05\ \text{m}$):
1. **`strategy="core_cutoff"` (default)**: Clamps radial denominators at $\max(r, r_{\min})$ while preserving exact Coulomb/Biot–Savart values at all points $r \ge r_{\min}$. At $r = 0$, vector components are set to $0$.
2. **`strategy="softening"`**: Replaces $r$ with $\sqrt{r^2 + r_{\min}^2}$ (Plummer softening), yielding a smooth $C^\infty$ potential everywhere at the cost of an $\mathcal{O}((r_{\min}/r)^2)$ approximation error outside the core.
3. **`strategy="mask"`**: Returns exact values for $r \ge r_{\min}$ and `np.nan` for $r < r_{\min}$.

All result containers (`ElectricFieldResult`, `ElectricPotentialResult`, `MagneticField2DResult`) include a boolean `core_mask` array identifying every grid point inside $r < r_{\min}$, and plots draw dashed circles of radius $r_{\min}$ around sources to make the core boundary visually transparent.

---

## 2. Point Magnetic Dipole vs. Real Bar Magnets

In [`magnetism.py`](../src/invisible_physics/magnetism.py), `magnetic_field_dipole_2d()` computes the field of an **idealized point magnetic dipole**:

$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi}\frac{3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m}}{r^3}$$

- **What it accurately represents**: The asymptotic far-field ($r \gg L$) of any localized steady current loop or magnetized body of characteristic size $L$.
- **What it does NOT represent**: The near-surface or interior magnetic field of a finite physical bar magnet of length $L$ and width $W$. Inside a real uniformly magnetized cylinder or bar magnet, $\mathbf{B}$ is finite and directed from South to North along the magnet's axis, whereas a point dipole is singular at the origin. Users studying near-field magnet geometry should compare the point dipole with `magnetic_field_circular_loop_xz()`, which resolves the finite radius $R$ of a current loop exactly.

---

## 3. Static Fields vs. Radiating Electromagnetic Waves

- Modules 1–3 (`electrostatics.py` and `magnetism.py`) solve **time-independent (static)** Maxwell equations ($\partial\mathbf{E}/\partial t = \mathbf{0}$, $\partial\mathbf{B}/\partial t = \mathbf{0}$). Static charges and steady direct currents do not radiate electromagnetic waves, and electric and magnetic fields are completely decoupled in the static regime.
- Module 4 (`waves.py`) solves **time-varying source-free** Maxwell equations in vacuum, where $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t$ and $\nabla \times \mathbf{B} = \mu_0\varepsilon_0\,\partial\mathbf{E}/\partial t$ couple $\mathbf{E}$ and $\mathbf{B}$ into self-propagating transverse waves traveling at $c$.
- Moving charges or time-varying currents in the intermediate (near-to-far) radiation zone require retarded potentials (Jefimenko's equations / Liénard–Wiechert potentials), which are outside the scope of the static and plane-wave modules.

---

## 4. Ideal Plane Waves & Vacuum vs. Material Media

- **Infinite Transverse Extent**: An ideal plane wave $\mathbf{E}(x, t) = E_0 \cos(kx - \omega t + \phi)\,\hat{\mathbf{y}}$ has uniform amplitude across the entire infinite $y$-$z$ plane. Real laser beams and antenna emissions have finite transverse profiles (e.g., Gaussian beams or spherical waves) and diffract over distance.
- **Vacuum vs. Dielectric Media**: `PlaneWaveVacuum` strictly enforces the vacuum dispersion relation $c = f\lambda$ and $|\mathbf{E}| = c|\mathbf{B}|$. In a dispersive material medium with complex refractive index $\tilde{n}(\omega)$, phase velocity ($v_p = c / \operatorname{Re}\tilde{n}$), group velocity ($v_g = \partial\omega/\partial k$), attenuation, and wave impedance ($Z = Z_0 / \tilde{n}$) all differ from vacuum values.

---

## 5. Finite-Difference Discretization Error

- Computing $\mathbf{E}_{\text{num}} = -\nabla V$ via `numerical_field_from_potential()` uses second-order central differences on a discrete grid of spacing $h$.
- Near point charges where higher derivatives $\partial^3 V / \partial x^3 \propto r^{-4}$ grow rapidly, truncation error $\mathcal{O}(h^2 r^{-4})$ becomes significant unless $h \ll r$. Furthermore, taking finite differences across the $r = r_{\min}$ core cutoff boundary introduces a localized derivative discontinuity at $r = r_{\min}$. Quantitative gradient comparisons should therefore always be restricted to $r \ge r_{\text{eval}} > r_{\min} + 2h$, as enforced by `validate_potential_gradient_consistency()`.

