"""SOM neuron grid, hit map, and target map."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib import cm

from ._palette import regime_color


def plot_som_grid(
    neuron_ids: list,
    neuron_labels: np.ndarray,
    neuron_sizes: np.ndarray,
    grid_size: tuple,
    n_groups: int,
    *,
    neuron_target_means: np.ndarray | None = None,
    target_label: str = "Target Mean",
    regime_prefix: str = "R",
    figsize: tuple = (20, 6),
    dpi: int = 200,
    save_path=None,
):
    """SOM neuron group + hit map (+ optional target distribution)."""
    rows, cols = grid_size
    n_panels = 3 if neuron_target_means is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, dpi=dpi)
    if n_panels == 2:
        ax1, ax2 = axes
        ax3 = None
    else:
        ax1, ax2, ax3 = axes

    group_grid = np.full((rows, cols), np.nan)
    size_grid = np.zeros((rows, cols))
    target_grid = np.full((rows, cols), np.nan)

    for idx, (r, c) in enumerate(neuron_ids):
        group_grid[r, c] = neuron_labels[idx]
        size_grid[r, c] = neuron_sizes[idx]
        if neuron_target_means is not None:
            target_grid[r, c] = neuron_target_means[idx]

    # Panel 1 -- regime assignment
    for r in range(rows):
        for c in range(cols):
            g = group_grid[r, c]
            if np.isnan(g):
                color = "#F5F5F5"; text = ""
            else:
                color = regime_color(int(g))
                text = f"{regime_prefix}{int(g)}"
            ax1.add_patch(plt.Rectangle(
                (c - 0.45, r - 0.45), 0.9, 0.9,
                facecolor=color, edgecolor="white",
                linewidth=1.5, alpha=0.85,
            ))
            if text:
                ax1.text(c, r, text, ha="center", va="center",
                         fontsize=6, fontweight="bold", color="white")

    ax1.set_xlim(-0.5, cols - 0.5); ax1.set_ylim(rows - 0.5, -0.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("SOM Column", fontsize=10)
    ax1.set_ylabel("SOM Row", fontsize=10)
    ax1.set_title("SOM Regime Assignment", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(cols)); ax1.set_yticks(range(rows))
    ax1.tick_params(labelsize=7)
    legend = [
        Patch(facecolor=regime_color(g), label=f"{regime_prefix}{g}")
        for g in range(1, n_groups + 1)
    ]
    ax1.legend(
        handles=legend, loc="upper center",
        bbox_to_anchor=(0.5, -0.08), fontsize=8,
        ncol=n_groups, frameon=False,
    )

    # Panel 2 -- hit map
    max_size = float(size_grid.max())
    for r in range(rows):
        for c in range(cols):
            sz = size_grid[r, c]
            if sz == 0:
                color = "#F5F5F5"; alpha = 0.3
            else:
                intensity = min(sz / max_size, 1.0) if max_size > 0 else 0
                alpha = 0.3 + 0.7 * intensity
                color = "#1565C0"
            ax2.add_patch(plt.Rectangle(
                (c - 0.45, r - 0.45), 0.9, 0.9,
                facecolor=color, edgecolor="white",
                linewidth=1.5, alpha=alpha,
            ))
            if sz > 0:
                intensity = sz / max_size if max_size > 0 else 0
                ax2.text(
                    c, r, f"{int(sz)}", ha="center", va="center",
                    fontsize=5.5,
                    color="white" if intensity > 0.5 else "black",
                )

    ax2.set_xlim(-0.5, cols - 0.5); ax2.set_ylim(rows - 0.5, -0.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("SOM Column", fontsize=10)
    ax2.set_ylabel("SOM Row", fontsize=10)
    active = int((size_grid > 0).sum())
    if max_size > 0:
        sub = f"Active: {active}/{rows * cols},   range {int(size_grid[size_grid > 0].min())}-{int(max_size)}"
    else:
        sub = f"Active: 0/{rows * cols}"
    ax2.set_title(f"SOM Hit Map (samples per neuron)\n{sub}",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(range(cols)); ax2.set_yticks(range(rows))
    ax2.tick_params(labelsize=7)

    # Panel 3 -- target distribution
    if ax3 is not None and neuron_target_means is not None:
        valid = target_grid[~np.isnan(target_grid)]
        if len(valid) > 0:
            vmin, vmax = float(valid.min()), float(valid.max())
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_target = cm.get_cmap("YlOrRd")
        for r in range(rows):
            for c in range(cols):
                val = target_grid[r, c]
                if np.isnan(val):
                    color = "#F5F5F5"; alpha = 0.3
                    text_color = "gray"
                else:
                    color = cmap_target(norm(val)); alpha = 1.0
                    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                    text_color = "white" if brightness < 0.5 else "black"
                ax3.add_patch(plt.Rectangle(
                    (c - 0.45, r - 0.45), 0.9, 0.9,
                    facecolor=color, edgecolor="white",
                    linewidth=1.5, alpha=alpha,
                ))
                if not np.isnan(val):
                    ax3.text(c, r, f"{val:.1f}", ha="center", va="center",
                             fontsize=5, fontweight="bold", color=text_color)
        ax3.set_xlim(-0.5, cols - 0.5); ax3.set_ylim(rows - 0.5, -0.5)
        ax3.set_aspect("equal")
        ax3.set_xlabel("SOM Column", fontsize=10)
        ax3.set_ylabel("SOM Row", fontsize=10)
        ax3.set_title(f"SOM {target_label}\nRange {vmin:.2f}-{vmax:.2f}",
                      fontsize=11, fontweight="bold")
        ax3.set_xticks(range(cols)); ax3.set_yticks(range(rows))
        ax3.tick_params(labelsize=7)
        sm = cm.ScalarMappable(cmap=cmap_target, norm=norm); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.set_label(target_label, fontsize=9)
        cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig
