"""Spatial distribution plots for SHAP-Compass regimes."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._palette import regime_color


def _maybe_twd97_to_wgs84(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert TWD97 (EPSG:3826) to WGS84 if pyproj is available."""
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return np.asarray(lon), np.asarray(lat)
    except Exception:
        lon = x / 111320 + 121.0 - (250000 / 111320)
        lat = y / 110540
        return lon, lat


def plot_spatial(
    labels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_groups: int,
    target: np.ndarray | None = None,
    *,
    convert_to_wgs84: bool = True,
    regime_prefix: str = "R",
    target_label: str = "Target",
    figsize: tuple = (16, 7),
    dpi: int = 200,
    save_path=None,
):
    """Spatial map of regime labels with an optional target panel.

    When ``convert_to_wgs84=True`` and the supplied coordinates look like
    TWD97 (X > 1000) the points are projected to lon/lat.
    """
    x = np.asarray(x); y = np.asarray(y)
    if convert_to_wgs84 and x.max() > 1000:
        lon, lat = _maybe_twd97_to_wgs84(x, y)
        xlabel, ylabel = "Longitude (°E)", "Latitude (°N)"
    else:
        lon, lat = x, y
        xlabel, ylabel = "X", "Y"

    ncols = 2 if target is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=dpi)
    if ncols == 1:
        axes = [axes]

    pad_x = (lon.max() - lon.min()) * 0.05
    pad_y = (lat.max() - lat.min()) * 0.05
    xlim = (lon.min() - pad_x, lon.max() + pad_x)
    ylim = (lat.min() - pad_y, lat.max() + pad_y)

    ax = axes[0]
    for g in range(1, n_groups + 1):
        mask = labels == g
        ax.scatter(
            lon[mask], lat[mask],
            c=regime_color(g), s=12, alpha=0.7, edgecolors="none",
            label=f"{regime_prefix}{g} (n={int(mask.sum())})",
        )
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("SHAP-Compass Regime Spatial Distribution", fontsize=12, fontweight="bold")
    ax.legend(
        fontsize=8, markerscale=2, loc="upper left",
        bbox_to_anchor=(0.0, -0.08), ncol=n_groups, frameon=False,
    )
    ax.set_aspect("equal")

    if target is not None:
        ax2 = axes[1]
        sc = ax2.scatter(
            lon, lat, c=target, cmap="YlOrRd",
            s=12, alpha=0.7, vmin=0, vmax=10, edgecolors="none",
        )
        # Attach the colorbar with a fixed-width inset so the right
        # panel's plot area stays the same width as the left panel.
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="3.5%", pad=0.08)
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(target_label, fontsize=9)
        ax2.set_xlim(xlim); ax2.set_ylim(ylim)
        ax2.set_xlabel(xlabel, fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_title(f"{target_label} distribution", fontsize=12, fontweight="bold")
        ax2.set_aspect("equal")

    plt.subplots_adjust(wspace=0.15)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


def plot_group_facets(
    labels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_groups: int,
    target: np.ndarray | None = None,
    *,
    convert_to_wgs84: bool = True,
    regime_prefix: str = "R",
    figsize: tuple | None = None,
    dpi: int = 200,
    save_path=None,
):
    """Faceted spatial map: one subplot per regime."""
    x = np.asarray(x); y = np.asarray(y)
    if convert_to_wgs84 and x.max() > 1000:
        lon, lat = _maybe_twd97_to_wgs84(x, y)
        xlabel, ylabel = "Longitude (°E)", "Latitude (°N)"
    else:
        lon, lat = x, y
        xlabel, ylabel = "X", "Y"

    cols = min(3, n_groups)
    rows = (n_groups + cols - 1) // cols
    if figsize is None:
        figsize = (5 * cols, 5.5 * rows + 0.8)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    axes = np.atleast_1d(axes).ravel()

    pad_x = (lon.max() - lon.min()) * 0.06
    pad_y = (lat.max() - lat.min()) * 0.06
    xlim = (lon.min() - pad_x, lon.max() + pad_x)
    ylim = (lat.min() - pad_y, lat.max() + pad_y)

    for g in range(n_groups):
        ax = axes[g]
        mask = labels == (g + 1)
        ax.scatter(lon[~mask], lat[~mask], c="#E0E0E0", s=4, alpha=0.3, edgecolors="none")
        color = regime_color(g + 1)
        ax.scatter(lon[mask], lat[mask], c=color, s=14, alpha=0.85, edgecolors="none")
        title_bits = [f"{regime_prefix}{g + 1}", f"n={int(mask.sum())}"]
        if target is not None and mask.any():
            title_bits.append(f"mean={float(target[mask].mean()):.2f}")
        ax.set_title("  ".join(title_bits), fontsize=10, fontweight="bold", color=color)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        if g % cols == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if g >= (rows - 1) * cols:
            ax.set_xlabel(xlabel, fontsize=9)

    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    legend = [
        Patch(facecolor=regime_color(g + 1), label=f"{regime_prefix}{g + 1}")
        for g in range(n_groups)
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=n_groups,
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle("Spatial Distribution by Regime", fontsize=14, fontweight="bold")
    plt.subplots_adjust(hspace=0.25, wspace=0.12)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig
