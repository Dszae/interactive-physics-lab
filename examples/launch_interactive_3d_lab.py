"""
Generates and launches an interactive, 360-degree mouse-rotatable 3D & 2D WebGL Physics Laboratory
in your web browser using Plotly and Invisible Physics.

Usage:
    python examples/launch_interactive_3d_lab.py [--no-browser]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import webbrowser
import numpy as np

from invisible_physics import (
    PlaneWaveVacuum,
    create_coordinate_grid,
    electric_potential_2d,
    evaluate_plane_wave_1d,
    get_preset_charges,
    plot_interactive_potential_3d_plotly,
    plot_interactive_wave_3d_plotly,
)


def build_interactive_html(output_path: Path) -> Path:
    X, Y = create_coordinate_grid((-2.0, 2.0), (-2.0, 2.0), n_points=95)

    # 1. 3D Electric Dipole Potential Landscape
    dipole = get_preset_charges("dipole", scale_q_uc=1.5, separation_m=1.2)
    v_dip = electric_potential_2d(X, Y, dipole, r_min=0.06, strategy="softening")
    fig_dip = plot_interactive_potential_3d_plotly(
        v_dip,
        dipole,
        title="1. Electric Dipole (+1.5 μC, −1.5 μC) — 3D Potential Hill & Well (Left-Click & Drag to Rotate)",
    )

    # 2. 3D Symmetric 4-Charge Quadrupole Potential Landscape
    quad = get_preset_charges("symmetric_quadrupole", scale_q_uc=1.5, separation_m=1.4)
    v_quad = electric_potential_2d(X, Y, quad, r_min=0.06, strategy="softening")
    fig_quad = plot_interactive_potential_3d_plotly(
        v_quad,
        quad,
        title="2. Symmetric 4-Charge Quadrupole — 3D Potential Saddle & Peaks (Left-Click & Drag to Rotate)",
    )

    # 3. 3D Two Like Positive Charges (+q, +q)
    like_pair = get_preset_charges("two_like_positive", scale_q_uc=1.5, separation_m=1.2)
    v_like = electric_potential_2d(X, Y, like_pair, r_min=0.06, strategy="softening")
    fig_like = plot_interactive_potential_3d_plotly(
        v_like,
        like_pair,
        title="3. Two Like Positive Charges (+1.5 μC, +1.5 μC) — Twin Potential Hills & Saddle Null Point",
    )

    # 4. 3D Plane Electromagnetic Wave
    wave = PlaneWaveVacuum.from_frequency(frequency_hz=100.0e6, e0_v_per_m=120.0)
    x_wave = np.linspace(0.0, 2.5 * wave.wavelength_m, 300)
    prof = evaluate_plane_wave_1d(wave, x_wave, t_s=0.0)
    fig_wave = plot_interactive_wave_3d_plotly(prof)

    div_dip = fig_dip.to_html(full_html=False, include_plotlyjs="cdn")
    div_quad = fig_quad.to_html(full_html=False, include_plotlyjs=False)
    div_like = fig_like.to_html(full_html=False, include_plotlyjs=False)
    div_wave = fig_wave.to_html(full_html=False, include_plotlyjs=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Invisible Physics — Interactive 3D Rotatable Laboratory</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #f4f6f9;
      color: #1f2937;
      margin: 0;
      padding: 0;
    }}
    header {{
      background: #0f172a;
      color: #ffffff;
      padding: 24px 36px;
      border-bottom: 3px solid #2563eb;
    }}
    header h1 {{
      margin: 0 0 6px 0;
      font-size: 24px;
    }}
    header p {{
      margin: 0;
      color: #cbd5e1;
      font-size: 15px;
    }}
    .guide-banner {{
      background: #eff6ff;
      border-left: 5px solid #2563eb;
      margin: 20px auto;
      max-width: 960px;
      padding: 14px 20px;
      border-radius: 6px;
      font-size: 14.5px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto 48px auto;
      padding: 0 16px;
    }}
    .card {{
      background: #ffffff;
      border-radius: 10px;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
      padding: 20px;
      margin-bottom: 28px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .card p {{
      align-self: flex-start;
      margin: 12px 16px 0 16px;
      color: #374151;
      font-size: 14px;
      line-height: 1.55;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Invisible Physics — Interactive 360° 3D &amp; 2D Visualization Lab</h1>
    <p>True WebGL 3D Models: Left-click &amp; drag any plot to rotate in 3D, scroll to zoom, right-click to pan, and hover to inspect exact SI values.</p>
  </header>
  <div class="guide-banner">
    <strong>How to Interact with These 3D Models:</strong><br/>
    • <strong>Rotate in 3D:</strong> Hold down your <strong>Left Mouse Button</strong> on any 3D surface and drag in any direction.<br/>
    • <strong>Zoom In / Out:</strong> Roll your <strong>Mouse Scroll Wheel</strong> up or down over the plot.<br/>
    • <strong>Inspect Values:</strong> Hover your cursor over the 3D surface to see exact $(x, y, V)$ coordinates and green 2D equipotential contour projections!
  </div>
  <div class="container">
    <div class="card">
      {div_dip}
      <p><strong>Physical Interpretation:</strong> The positive charge ($+1.5\\ \\mu\\text{{C}}$, red peak) creates a steep electrostatic potential hill ($V > 0$), while the negative charge ($-1.5\\ \\mu\\text{{C}}$, blue well) creates a deep potential well ($V < 0$). Notice the green equipotential contour rings projected onto the bottom/top planes as you hover!</p>
    </div>
    <div class="card">
      {div_quad}
      <p><strong>Physical Interpretation:</strong> Four alternating charges ($+q, -q, +q, -q$) form two positive peaks and two negative valleys meeting at a central saddle point $(0, 0)$ where both $V = 0$ and $|\\mathbf{{E}}| = 0$.</p>
    </div>
    <div class="card">
      {div_like}
      <p><strong>Physical Interpretation:</strong> Two equal positive charges ($+1.5\\ \\mu\\text{{C}}, +1.5\\ \\mu\\text{{C}}$) create twin potential hills separated by a elevated saddle pass at $(0, 0)$ where $V > 0$ but the slope vanishes ($\\mathbf{{E}} = -\\nabla V = \\mathbf{{0}}$).</p>
    </div>
    <div class="card">
      {div_wave}
      <p><strong>Physical Interpretation:</strong> Rotate this 3D electromagnetic wave to view it along the $x$-axis (head-on) and confirm that the red electric field $\\mathbf{{E}}_y(x, t)$ and blue magnetic field $\\mathbf{{B}}_z(x, t)$ oscillate at exactly $90^\\circ$ to each other and to the propagation axis!</p>
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and launch the interactive 3D WebGL Physics Lab.")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser.")
    args = parser.parse_args()

    out_file = build_interactive_html(Path("interactive_3d_lab.html"))
    print(f"Built interactive 3D WebGL laboratory at: {out_file.resolve()}")
    if not args.no_browser:
        webbrowser.open(out_file.resolve().as_uri())


if __name__ == "__main__":
    main()

