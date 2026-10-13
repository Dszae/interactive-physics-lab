# Guide to Reproducible Physics Experiments

This guide summarizes the seven quantitative experiments implemented across [`notebooks/05_physics_experiments.ipynb`](../notebooks/05_physics_experiments.ipynb) and [`examples/example_experiments.py`](../examples/example_experiments.py). Each experiment follows the scientific method: **Question → Prediction → Physical Model → Computation → Quantitative Observation → Interpretation → Limitations**.

---

## Experiment 1: Charge Linearity ($|\mathbf{E}| \propto |q|$)

- **Question**: How does doubling a point charge's magnitude affect the electric-field magnitude $|\mathbf{E}|$ and scalar potential $V$ at a fixed observation distance?
- **Prediction**: By Coulomb's law, both $|\mathbf{E}(\mathbf{r})|$ and $V(\mathbf{r})$ are strictly linear in $q$. Doubling $q$ from $+1.0\ \mu\text{C}$ to $+2.0\ \mu\text{C}$ doubles $|\mathbf{E}|$ and $V$ at every point outside the singularity core.
- **Verification**: Evaluated at $r = 1.0\ \text{m}$, $|\mathbf{E}(2q)| / |\mathbf{E}(q)| = 2.000000000000$ to machine precision.

---

## Experiment 2: Inverse-Square Distance Law ($|\mathbf{E}| \propto 1/r^2$)

- **Question**: How does the electric-field magnitude of a point charge decay with radial distance $r$, and how does that contrast with its scalar potential $V(r)$?
- **Prediction**: $|\mathbf{E}(r)| \propto r^{-2}$ (log-log slope $= -2.0$), whereas $V(r) \propto r^{-1}$ (log-log slope $= -1.0$). Doubling $r$ reduces $|\mathbf{E}|$ by a factor of $4$ and $V$ by a factor of $2$.
- **Verification**: `verify_inverse_square_law()` computes the linear regression slope of $\ln|\mathbf{E}(r)|$ vs. $\ln r$ over $r \in [0.25, 4.0]\ \text{m}$, yielding $-2.000000000000$ with relative error $< 10^{-15}$.

---

## Experiment 3: Locating Zero-Field Null Points ($\mathbf{E}(\mathbf{r}_{\text{null}}) = \mathbf{0}$)

- **Question**: Where does the net electric field of two point charges vanish, and how does the null-point location depend on relative charge signs and magnitudes?
- **Prediction**:
  1. For two like charges ($q_1 q_2 > 0$ at $x_1 < x_2$), the repulsive fields oppose each other between the charges, producing a unique saddle null point inside $(x_1, x_2)$:
     $$x_{\text{null}} = \frac{x_1\sqrt{|q_2|} + x_2\sqrt{|q_1|}}{\sqrt{|q_1|} + \sqrt{|q_2|}}$$
  2. For two unequal opposite charges ($q_1 q_2 < 0, |q_1| \neq |q_2|$), the fields reinforce between the charges; the unique null point lies outside the segment $[x_1, x_2]$ on the side of the smaller-magnitude charge:
     $$x_{\text{null}} = \frac{x_2\sqrt{|q_1|} - x_1\sqrt{|q_2|}}{\sqrt{|q_1|} - \sqrt{|q_2|}}$$
  3. For a symmetric dipole ($q_2 = -q_1$), no finite null point exists.
- **Verification**: `find_collinear_zero_field_point()` uses Brent's root-finding method (`scipy.optimize.brentq`) on $E_x(x, 0)$ and matches the analytical root to $< 10^{-11}\ \text{m}$.

---

## Experiment 4: Equipotential Orthogonality & Numerical Gradient ($\mathbf{E} = -\nabla V$)

- **Question**: How accurately does the negative numerical gradient $-\nabla V$ computed on a 2D mesh reproduce the exact Coulomb vector field $\mathbf{E}(\mathbf{r})$?
- **Prediction**: Outside the singularity cores ($r \ge r_{\text{eval}}$), second-order central differences converge at rate $\mathcal{O}(h^2)$ where $h = \Delta x = \Delta y$ is the grid spacing, and the angle between $\mathbf{E}_{\text{Coulomb}}$ and $-\nabla V$ approaches $0^\circ$.
- **Verification**: `validate_potential_gradient_consistency()` demonstrates that refining the grid from $51 \times 51$ to $201 \times 201$ reduces the median relative error by over an order of magnitude (to $< 0.08\%$) with RMS angular deviation $< 0.05^\circ$.

---

## Experiment 5: Magnetic Field Scaling Around a Long Wire vs. Circular Loop

- **Question**: How does the magnetic flux density $|\mathbf{B}|$ decay with distance for an infinitely long straight wire compared to the axial field of a circular current loop?
- **Prediction**: A long wire decays as $r^{-1}$ ($B = \mu_0 I / (2\pi r)$) because the source is infinite in extent along $z$, whereas a finite circular loop of radius $R$ approaches $z^{-3}$ far along its axis ($z \gg R$), matching the asymptotic scaling of a magnetic dipole with moment $m = I \pi R^2$.
- **Verification**: Verified via `magnetic_field_straight_wire_1d()` and `verify_circular_loop_on_axis()`.

---

## Experiment 6: Vacuum Electromagnetic Wave Dispersion ($c = f\lambda$)

- **Question**: How does increasing wave frequency $f$ affect wavelength $\lambda$ and magnetic-field amplitude $B_0$ for a plane wave of fixed electric amplitude $E_0$ in vacuum?
- **Prediction**: Because phase speed in vacuum is fixed by fundamental constants ($c = 1/\sqrt{\mu_0\varepsilon_0}$), wavelength scales inversely with frequency ($\lambda = c / f$), while magnetic amplitude $B_0 = E_0 / c$ depends only on $E_0$ and is independent of frequency.

---

## Experiment 7: Grid Resolution & Singularity Regularization Sensitivity

- **Question**: How do finite grid spacing $h$ and the singularity cutoff strategy (`core_cutoff`, `softening`, `mask`) influence computed fields near and far from point charges?
- **Prediction**:
  - `core_cutoff` and `mask` are *exact* everywhere outside $r \ge r_{\min}$, whereas Plummer `softening` ($r_{\text{soft}} = \sqrt{r^2 + r_{\min}^2}$) introduces a smooth $\mathcal{O}((r_{\min}/r)^2)$ perturbation that decays rapidly for $r \gg r_{\min}$.
  - Inside $r < r_{\min}$, all three strategies regularize the unphysical classical point-charge divergence in documented, predictable ways.

