# -*- coding: utf-8 -*-
"""Generate the README's conceptual illustration figures.

Three PNGs are written to ``docs/concepts/``:

    concept_pipeline.png            -- Four-stage flowchart of the
                                       SHAP-Compass pipeline.
    concept_unit_circle.png         -- How a single (feature, attribution)
                                       sample is mapped to (cos theta,
                                       sin theta) on the unit circle.
    concept_dci.png                 -- The geometric meaning of DCI: low
                                       DCI = dispersed regime centroids;
                                       high DCI = aligned centroids.

Re-run this script after changing the pipeline figure or whenever the
README needs refreshed concept art:

    python docs/build_concept_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

OUT = Path(__file__).resolve().parent / "concepts"
OUT.mkdir(parents=True, exist_ok=True)

REGIME_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#FFD700", "#984ea3", "#ff7f00"]


# ---------------------------------------------------------------------------
# 1. Pipeline flowchart
# ---------------------------------------------------------------------------

def draw_box(ax, xy, width, height, text, *,
             face: str = "#E3F2FD", edge: str = "#1565C0",
             fontsize: float = 10, fontweight: str = "bold") -> tuple[float, float]:
    """Draw a rounded box centred at ``xy`` and return the box centre."""
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=face, edgecolor=edge, linewidth=1.8,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color="#0D47A1")
    return (x, y)


def draw_arrow(ax, start, end, *, color: str = "#37474F") -> None:
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=18,
        color=color, linewidth=1.7,
    )
    ax.add_patch(arrow)


def make_pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.0), dpi=180)
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.5); ax.axis("off")

    # Stage 1 -- inputs
    in_x_X = draw_box(ax, (1.2, 6.4), 1.9, 0.8, "Features  X\n(N x J)",
                      face="#FFF8E1", edge="#F57C00")
    in_x_S = draw_box(ax, (1.2, 4.6), 1.9, 0.8, "Attributions  $\\Phi$\n(e.g. SHAP)",
                      face="#FFF8E1", edge="#F57C00")

    # Stage 2 -- Z-standardise + theta + COSSIN
    z_X = draw_box(ax, (4.0, 6.4), 1.6, 0.7, "Z-standardise\n$Z^F$",
                   face="#E3F2FD", edge="#1565C0")
    z_S = draw_box(ax, (4.0, 4.6), 1.6, 0.7, "Z-standardise\n$Z^S$",
                   face="#E3F2FD", edge="#1565C0")
    theta_box = draw_box(ax, (6.5, 5.5), 2.0, 0.9,
                          r"$\theta = \mathrm{arctan2}(Z^S,\, Z^F)$" "\n"
                          r"$(\cos\theta,\,\sin\theta)$",
                          face="#E8F5E9", edge="#2E7D32",
                          fontsize=10)
    sc_box = draw_box(ax, (9.4, 5.5), 2.2, 0.9,
                       "SHAP-Compass\nmatrix  (N x 2J)",
                       face="#E8F5E9", edge="#2E7D32")

    # Stage 3 -- SOM + Ward
    som_box = draw_box(ax, (3.2, 2.5), 2.4, 0.9,
                        "Self-Organising Map\n(M x M)",
                        face="#FCE4EC", edge="#AD1457")
    ward_box = draw_box(ax, (6.5, 2.5), 2.4, 0.9,
                         "Ward on neuron-level\nfingerprints",
                         face="#FCE4EC", edge="#AD1457")
    regimes_box = draw_box(ax, (9.7, 2.5), 2.0, 0.9,
                            "K attribution\nregimes",
                            face="#F3E5F5", edge="#6A1B9A")

    # Stage 4 -- outputs
    dci_box = draw_box(ax, (2.5, 0.7), 2.4, 0.8,
                        "DCI per feature\n(0..1, 4 bands)",
                        face="#FFF3E0", edge="#E65100", fontsize=10)
    heat_box = draw_box(ax, (6.0, 0.7), 2.8, 0.8,
                         "Bilayer feature heatmap\n" r"($Z^{F}$ upper, $Z^{S}$ lower)",
                         face="#FFF3E0", edge="#E65100", fontsize=10)
    qm_box = draw_box(ax, (9.7, 0.7), 2.0, 0.8,
                       "Quality M01-M21\n(per-metric output)",
                       face="#FFF3E0", edge="#E65100", fontsize=10)

    # Arrows
    draw_arrow(ax, (in_x_X[0] + 0.95, in_x_X[1]), (z_X[0] - 0.8, z_X[1]))
    draw_arrow(ax, (in_x_S[0] + 0.95, in_x_S[1]), (z_S[0] - 0.8, z_S[1]))
    draw_arrow(ax, (z_X[0] + 0.8, z_X[1]), (theta_box[0] - 1.0, theta_box[1] + 0.3))
    draw_arrow(ax, (z_S[0] + 0.8, z_S[1]), (theta_box[0] - 1.0, theta_box[1] - 0.3))
    draw_arrow(ax, (theta_box[0] + 1.0, theta_box[1]), (sc_box[0] - 1.1, sc_box[1]))
    draw_arrow(ax, (sc_box[0] - 0.6, sc_box[1] - 0.45), (som_box[0] + 0.6, som_box[1] + 0.45),
               color="#AD1457")
    draw_arrow(ax, (som_box[0] + 1.2, som_box[1]), (ward_box[0] - 1.2, ward_box[1]))
    draw_arrow(ax, (ward_box[0] + 1.2, ward_box[1]), (regimes_box[0] - 1.0, regimes_box[1]))
    draw_arrow(ax, (regimes_box[0] - 0.4, regimes_box[1] - 0.45),
               (dci_box[0] + 0.6, dci_box[1] + 0.4), color="#E65100")
    draw_arrow(ax, (regimes_box[0], regimes_box[1] - 0.45),
               (heat_box[0], heat_box[1] + 0.4), color="#E65100")
    draw_arrow(ax, (regimes_box[0] + 0.4, regimes_box[1] - 0.45),
               (qm_box[0] - 0.4, qm_box[1] + 0.4), color="#E65100")

    # Stage labels on the left
    for y, label, color in [
        (6.4, "Stage 1\nInputs", "#F57C00"),
        (5.5, "Stage 2\nTransform", "#1565C0"),
        (2.5, "Stage 3\nCluster", "#AD1457"),
        (0.7, "Stage 4\nReport", "#E65100"),
    ]:
        ax.text(0.05, y, label, fontsize=9, fontweight="bold",
                color=color, ha="left", va="center")

    ax.set_title("SHAP-Compass pipeline", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "concept_pipeline.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  concept_pipeline.png")


# ---------------------------------------------------------------------------
# 2. Unit-circle projection intuition
# ---------------------------------------------------------------------------

def make_unit_circle_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=180)

    # Three illustrative samples
    samples = {
        "A": {"zf": 1.2, "zs": 0.9, "color": "#1565C0"},
        "B": {"zf": -0.8, "zs": 1.5, "color": "#E65100"},
        "C": {"zf": -1.4, "zs": -1.1, "color": "#2E7D32"},
    }

    # Panel (a): Z^F vs Z^S scatter ----------------------------------------
    ax = axes[0]
    ax.axhline(0, color="#BDBDBD", lw=0.8)
    ax.axvline(0, color="#BDBDBD", lw=0.8)
    for name, s in samples.items():
        ax.scatter(s["zf"], s["zs"], s=140, color=s["color"], edgecolor="white",
                   linewidth=1.5, zorder=3)
        ax.annotate(name, (s["zf"], s["zs"]), xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=11, fontweight="bold", color=s["color"])
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel(r"$Z^F$  (standardised feature)", fontsize=11)
    ax.set_ylabel(r"$Z^S$  (standardised attribution)", fontsize=11)
    ax.set_title("(a)  Standardised plane", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # Panel (b): theta + r ------------------------------------------------------
    ax = axes[1]
    ax.axhline(0, color="#BDBDBD", lw=0.8)
    ax.axvline(0, color="#BDBDBD", lw=0.8)
    for name, s in samples.items():
        theta = np.arctan2(s["zs"], s["zf"])
        r = np.sqrt(s["zs"] ** 2 + s["zf"] ** 2)
        ax.plot([0, s["zf"]], [0, s["zs"]], color=s["color"], lw=2.2)
        ax.scatter(s["zf"], s["zs"], s=140, color=s["color"],
                   edgecolor="white", linewidth=1.5, zorder=3)
        # Theta arc
        arc_theta = np.linspace(0, theta, 30)
        arc_r = 0.32 + 0.08 * np.arange(len({k: v for k, v in samples.items()
                                              if v["color"] == s["color"]}))
        ax.plot(arc_r[0] * np.cos(arc_theta), arc_r[0] * np.sin(arc_theta),
                color=s["color"], lw=1.2)
        ax.annotate(
            rf"$\theta_{{{name}}}={np.degrees(theta):+.0f}^\circ$",
            (s["zf"], s["zs"]), xytext=(10, 10),
            textcoords="offset points",
            fontsize=9, color=s["color"],
        )
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel(r"$Z^F$", fontsize=11)
    ax.set_ylabel(r"$Z^S$", fontsize=11)
    ax.set_title(r"(b)  $\theta=\mathrm{arctan2}(Z^S,Z^F),\; r=\sqrt{(Z^F)^2+(Z^S)^2}$",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # Panel (c): unit-circle projection --------------------------------------
    ax = axes[2]
    circle_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), color="#9E9E9E", lw=1.2)
    ax.axhline(0, color="#E0E0E0", lw=0.6)
    ax.axvline(0, color="#E0E0E0", lw=0.6)
    for name, s in samples.items():
        theta = np.arctan2(s["zs"], s["zf"])
        ax.plot([0, np.cos(theta)], [0, np.sin(theta)],
                color=s["color"], lw=2.2)
        ax.scatter(np.cos(theta), np.sin(theta), s=160,
                   color=s["color"], edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(name, (np.cos(theta), np.sin(theta)),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=s["color"])
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel(r"$\cos\theta$", fontsize=11)
    ax.set_ylabel(r"$\sin\theta$", fontsize=11)
    ax.set_title(r"(c)  Unit-circle projection (drop $r$)",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    fig.suptitle(
        "From (feature, attribution) pair to SHAP-Compass vector",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "concept_unit_circle.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  concept_unit_circle.png")


# ---------------------------------------------------------------------------
# 3. DCI geometric meaning
# ---------------------------------------------------------------------------

def make_dci_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), dpi=180,
                              subplot_kw=dict(projection="polar"))

    # Panel (a): low DCI — dispersed centroids
    thetas_low = np.deg2rad([35, 120, 195, 260, 330])
    # Panel (b): high DCI — aligned centroids (small dispersion around 60°)
    thetas_high = np.deg2rad([55, 60, 65, 235, 245])

    def draw_panel(ax, thetas, title, dci_val, band_color):
        circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(circle, np.ones_like(circle), color="#9E9E9E", lw=1.2)
        # Regime centroids
        for k, t in enumerate(thetas):
            color = REGIME_COLORS[k % len(REGIME_COLORS)]
            ax.plot([t, t], [0, 1], color=color, lw=2.4)
            ax.plot(t, 1, "o", color=color, markersize=8)
        # Resultant after axial doubling
        C = float(np.mean(np.cos(2 * thetas)))
        S = float(np.mean(np.sin(2 * thetas)))
        R = float(np.sqrt(C ** 2 + S ** 2))
        result_theta = 0.5 * np.arctan2(S, C)
        ax.plot([result_theta, result_theta], [0, R], color="#212121",
                lw=3.0, linestyle="-", zorder=5)
        ax.plot(result_theta, R, "s", color="#212121", markersize=10, zorder=6)
        ax.set_ylim(0, 1.15)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=7)
        ax.set_title(f"{title}\nDCI = {dci_val:.2f}",
                     fontsize=12, fontweight="bold", pad=12, color=band_color)
        for sp in ax.spines.values():
            sp.set_edgecolor(band_color)
            sp.set_linewidth(3.0)
        return R

    R_low = draw_panel(axes[0], thetas_low,
                       "(a)  Low DCI — context-dependent",
                       dci_val=float(np.sqrt(
                           np.mean(np.cos(2 * thetas_low)) ** 2
                           + np.mean(np.sin(2 * thetas_low)) ** 2)),
                       band_color="#D32F2F")
    R_high = draw_panel(axes[1], thetas_high,
                        "(b)  High DCI — universal driver",
                        dci_val=float(np.sqrt(
                            np.mean(np.cos(2 * thetas_high)) ** 2
                            + np.mean(np.sin(2 * thetas_high)) ** 2)),
                        band_color="#2E7D32")

    fig.text(0.5, -0.02,
             "Coloured spokes: per-regime direction centroids on the unit circle."
             "  Black square: axial mean resultant (length = DCI).",
             ha="center", fontsize=10, style="italic")
    fig.suptitle("Directional Consistency Index (DCI)",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "concept_dci.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  concept_dci.png")


if __name__ == "__main__":
    print("[docs/build_concept_figures] writing concept figures ->", OUT)
    make_pipeline_figure()
    make_unit_circle_figure()
    make_dci_figure()
    print("done.")
