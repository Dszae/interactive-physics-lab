# Mathematical Models & Physical Foundations

This document details the analytical equations, coordinate conventions, SI unit systems, and numerical methods implemented in **Invisible Physics (`invisible_physics`)**.

---

## 1. Fundamental Physical Constants & SI Units

All internal computations in `invisible_physics` strictly adhere to the International System of Units (SI), using NIST CODATA 2022 values centralized in [`constants.py`](../src/invisible_physics/constants.py):

| Symbol | Quantity | Value | SI Unit |
| :--- | :--- | :--- | :--- |
| $c$ | Speed of light in vacuum | $299\,792\,458$ (exact) | $\text{m/s}$ |
| $\varepsilon_0$ | Vacuum electric permittivity | $8.8541878188 \times 10^{-12}$ | $\text{F/m}$ ($\text{C}^2 / (\text{N}\cdot\text{m}^2)$) |
| $\mu_0$ | Vacuum magnetic permeability | $1.25663706127 \times 10^{-6}$ | $\text{H/m}$ ($\text{N/A}^2$) |
| $k_e = \frac{1}{4\pi\varepsilon_0}$ | Coulomb constant | $8.9875517862 \times 10^{9}$ | $\text{N}\cdot\text{m}^2/\text{C}^2$ |
| $Z_0 = \mu_0 c$ | Characteristic impedance of free space | $376.730313412$ | $\Omega$ |

User-facing point charges are specified in microcoulombs ($\mu\text{C}$) and converted internally via $1\ \mu\text{C} = 10^{-6}\ \text{C}$.

---

## 2. Electrostatics (`electrostatics.py`)

### 2.1 Coulomb's Law & Linear Superposition

For $N$ stationary point charges $q_i$ located at positions $\mathbf{r}_i = (x_i, y_i)$ in the $z = 0$ plane, the electrostatic field $\mathbf{E}(\mathbf{r})$ at evaluation point $\mathbf{r} = (x, y)$ is given by linear superposition:

$$\mathbf{E}(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0} \sum_{i=1}^{N} q_i \frac{\mathbf{r} - \mathbf{r}_i}{\|\mathbf{r} - \mathbf{r}_i\|^3}$$

In Cartesian components with $\Delta x_i = x - x_i$, $\Delta y_i = y - y_i$, and $r_i = \sqrt{\Delta x_i^2 + \Delta y_i^2}$:

$$E_x(x, y) = k_e \sum_{i=1}^{N} \frac{q_i \Delta x_i}{r_i^3}, \qquad E_y(x, y) = k_e \sum_{i=1}^{N} \frac{q_i \Delta y_i}{r_i^3}$$

The field magnitude is $|\mathbf{E}| = \sqrt{E_x^2 + E_y^2}$ and the unit direction vector is $\hat{\mathbf{E}} = \mathbf{E} / |\mathbf{E}|$ wherever $|\mathbf{E}| > 0$.

### 2.2 Electrostatic Scalar Potential

Since electrostatic fields are conservative ($\nabla \times \mathbf{E} = \mathbf{0}$), $\mathbf{E}$ is the negative gradient of a scalar potential $V(\mathbf{r})$. Taking the standard reference $V(\infty) = 0$:

$$V(\mathbf{r}) = \frac{1}{4\pi\varepsilon_0} \sum_{i=1}^{N} \frac{q_i}{\|\mathbf{r} - \mathbf{r}_i\|}$$

Equipotential contours are curves of constant $V(x, y) = V_0$. Because the directional derivative of $V$ along an equipotential tangent $\hat{\mathbf{t}}$ vanishes ($\nabla V \cdot \hat{\mathbf{t}} = 0$), electric-field lines are everywhere orthogonal to equipotential contours and point from higher potential to lower potential:

$$\mathbf{E}(\mathbf{r}) = -\nabla V(\mathbf{r}) = -\left(\frac{\partial V}{\partial x}\,\hat{\mathbf{x}} + \frac{\partial V}{\partial y}\,\hat{\mathbf{y}}\right)$$

### 2.3 Numerical Gradient Discretization

In `numerical_field_from_potential()`, partial derivatives on a uniform Cartesian mesh with spacing $(\Delta x, \Delta y)$ are approximated using second-order centered finite differences at interior nodes:

$$\frac{\partial V}{\partial x}\bigg|_{j,k} = \frac{V_{j, k+1} - V_{j, k-1}}{2\Delta x} + \mathcal{O}(\Delta x^2), \qquad \frac{\partial V}{\partial y}\bigg|_{j,k} = \frac{V_{j+1, k} - V_{j-1, k}}{2\Delta y} + \mathcal{O}(\Delta y^2)$$

---

## 3. Magnetostatics (`magnetism.py`)

### 3.1 Infinitely Long Straight Wire

For a steady current $I$ flowing along the $z$-axis through $(x_0, y_0)$, Ampère's law and the Biot–Savart law yield an azimuthal magnetic flux density $\mathbf{B}$ in the $x$-$y$ plane at radial distance $r = \sqrt{(x - x_0)^2 + (y - y_0)^2}$:

$$|\mathbf{B}(r)| = \frac{\mu_0 |I|}{2\pi r}$$

$$\mathbf{B}(x, y) = \frac{\mu_0 I}{2\pi r^2}\bigl(-(y - y_0)\,\hat{\mathbf{x}} + (x - x_0)\,\hat{\mathbf{y}}\bigr)$$

By the right-hand rule, $I > 0$ (out of the page, $+\hat{\mathbf{z}}$, denoted $\odot$) produces counterclockwise magnetic field circulation; $I < 0$ (into the page, $-\hat{\mathbf{z}}$, denoted $\otimes$) produces clockwise circulation.

### 3.2 Circular Current Loop (Exact Off-Axis Elliptic Integral Solution)

Consider a circular loop of radius $R$ carrying current $I$ in the $x$-$y$ plane ($z = 0$), centered at the origin. In the meridional $x$-$z$ plane ($y = 0$) with cylindrical radial coordinate $\rho = |x|$, the exact Biot–Savart field components are expressed via complete elliptic integrals of the first and second kind, $K(m)$ and $E(m)$:

$$\alpha^2 = (R - \rho)^2 + z^2, \qquad \beta^2 = (R + \rho)^2 + z^2, \qquad m = \frac{4R\rho}{(R + \rho)^2 + z^2}$$

$$B_z(x, z) = \frac{\mu_0 I}{2\pi \beta}\left[K(m) + \frac{R^2 - \rho^2 - z^2}{\alpha^2} E(m)\right]$$

$$B_x(x, z) = \operatorname{sgn}(x)\frac{\mu_0 I z}{2\pi \rho \beta}\left[-K(m) + \frac{R^2 + \rho^2 + z^2}{\alpha^2} E(m)\right]$$

Along the symmetry axis ($\rho = 0 \implies m = 0, K(0) = E(0) = \pi/2$), this reduces identically to the elementary textbook result:

$$B_z(0, z) = \frac{\mu_0 I R^2}{2(R^2 + z^2)^{3/2}}, \qquad B_x(0, z) = 0$$

### 3.3 Point Magnetic Dipole (Far-Field Idealization)

At distances $r \gg R$ from a localized current distribution with magnetic dipole moment $\mathbf{m}$ (units $\text{A}\cdot\text{m}^2$), the lowest-order multipole expansion yields the point magnetic dipole field:

$$\mathbf{B}_{\text{dipole}}(\mathbf{r}) = \frac{\mu_0}{4\pi}\frac{3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m}}{r^3}$$

---

## 4. Electromagnetic Waves in Vacuum (`waves.py`)

In source-free vacuum ($\rho = 0, \mathbf{J} = \mathbf{0}$), Maxwell's equations decouple into the 3D wave equations:

$$\nabla^2 \mathbf{E} = \mu_0 \varepsilon_0 \frac{\partial^2 \mathbf{E}}{\partial t^2}, \qquad \nabla^2 \mathbf{B} = \mu_0 \varepsilon_0 \frac{\partial^2 \mathbf{B}}{\partial t^2}$$

with vacuum wave speed $c = 1 / \sqrt{\mu_0 \varepsilon_0}$. For a monochromatic linearly polarized plane wave propagating in the $\pm x$ direction:

$$\mathbf{E}(x, t) = E_0 \cos(\pm kx - \omega t + \phi)\,\hat{\mathbf{y}}, \qquad \mathbf{B}(x, t) = \pm B_0 \cos(\pm kx - \omega t + \phi)\,\hat{\mathbf{z}}$$

Maxwell's equations enforce three strict physical relationships:
1. **Dispersion relation**: $c = f\lambda$ (where $k = 2\pi/\lambda$ and $\omega = 2\pi f$).
2. **Transverse orthogonality**: $\mathbf{E} \perp \mathbf{B} \perp \mathbf{k}$, with $\mathbf{E} \times \mathbf{B}$ pointing in the direction of propagation.
3. **Field amplitude ratio**: $|\mathbf{E}(x, t)| = c\,|\mathbf{B}(x, t)|$ at every point in space and time ($B_0 = E_0 / c$).

The instantaneous directional energy flux density is given by the Poynting vector:

$$\mathbf{S}(x, t) = \frac{1}{\mu_0}(\mathbf{E} \times \mathbf{B}) = \pm\frac{E_0^2}{\mu_0 c}\cos^2(\pm kx - \omega t + \phi)\,\hat{\mathbf{x}}$$

with time-averaged irradiance $\langle |\mathbf{S}| \rangle = \frac{E_0^2}{2\mu_0 c} = \frac{E_0^2}{2 Z_0}$.

