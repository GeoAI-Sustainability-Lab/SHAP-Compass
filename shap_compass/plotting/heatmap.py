"""Group-overview bar + Z^S heatmap and theta heatmap."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._palette import regime_color


def plot_group_overview(
    labels: np.ndarray,
    target: np.ndarray,
    feature_names,
    ZF: np.ndarray,  # accepted for backward call shape; unused
    ZS: np.ndarray,
    n_groups: int,
    *,
    regime_prefix: str = "R",
    figsize: tuple = (14, 5),
    dpi: int = 200,
    save_path=None,
):
    """Per-regime target-mean bar chart + mean Z^S heatmap."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi,
        gridspec_kw={"width_ratios": [0.35, 0.65]},
    )

    means = [float(target[labels == g].mean()) for g in range(1, n_groups + 1)]
    sizes = [int((labels == g).sum()) for g in range(1, n_groups + 1)]
    colors = [regime_color(g) for g in range(1, n_groups + 1)]

    bars = ax1.barh(range(n_groups), means, color=colors, edgecolor="white", height=0.6)
    ax1.set_yticks(range(n_groups))
    ax1.set_yticklabels(
        [f"{regime_prefix}{g}\n(n={sizes[g - 1]})" for g in range(1, n_groups + 1)],
        fontsize=9,
    )
    ax1.set_xlabel("Mean Target Value", fontsize=10)
    ax1.set_title("Regime Target Means", fontsize=11, fontweight="bold")
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, val in zip(bars, means):
        ax1.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=8,
        )

    n_features = len(feature_names)
    zs_matrix = np.zeros((n_groups, n_features))
    for g in range(1, n_groups + 1):
        zs_matrix[g - 1] = ZS[labels == g].mean(axis=0)

    vmax = float(np.abs(zs_matrix).max()) if np.abs(zs_matrix).max() > 0 else 1.0
    im = ax2.imshow(zs_matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(n_features))
    ax2.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(n_groups))
    ax2.set_yticklabels([f"{regime_prefix}{g}" for g in range(1, n_groups + 1)], fontsize=9)
    ax2.set_title("Mean Z(SHAP) by Regime x Feature", fontsize=11, fontweight="bold")

    cb = fig.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cb.set_label("Mean Z(SHAP)", fontsize=9)

    for i in range(n_groups):
        for j in range(n_features):
            val = zs_matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color=color)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plot_theta_heatmap(
    group_theta: np.ndarray,
    feature_names,
    n_groups: int,
    *,
    regime_prefix: str = "R",
    figsize: tuple = (10, 4),
    dpi: int = 200,
    save_path=None,
):
    """Regime x feature heatmap of mean direction angle theta (degrees)."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    theta_deg = np.degrees(group_theta)
    im = ax.imshow(theta_deg, cmap="twilight", aspect="auto", vmin=-180, vmax=180)

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([f"{regime_prefix}{g}" for g in range(1, n_groups + 1)], fontsize=10)
    ax.set_title("Regime Centroid Direction Angle (degrees)", fontsize=12, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("theta (deg)", fontsize=9)

    for i in range(n_groups):
        for j in range(len(feature_names)):
            ax.text(
                j, i, f"{theta_deg[i, j]:.0f}°",
                ha="center", va="center", fontsize=6,
                color="white" if abs(theta_deg[i, j]) > 90 else "black",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig
