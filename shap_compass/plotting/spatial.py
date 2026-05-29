"""Spatial distribution plots for SHAP-Compass regimes.

When the optional `cartopy` package is installed (`pip install -e ".[spatial]"`),
:func:`plot_spatial` renders the points on top of a real basemap with
state and coastline outlines. Otherwise it falls back to a plain
scatter plot on white background.
"""

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


def _try_cartopy():
    """Return (ccrs, cfeature) if cartopy is importable, else (None, None)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None


def _decorate_basemap(ax, cfeature, xlim, ylim) -> None:
    """Add land / ocean / states / coastline to a cartopy GeoAxes."""
    ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f1ea", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dfe9f0", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#dfe9f0", zorder=1)
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        edgecolor="#9a9a9a", linewidth=0.4, zorder=2,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor="#555555", linewidth=0.6, zorder=2,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        edgecolor="#555555", linewidth=0.5, zorder=2,
    )


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
    add_basemap: bool = True,
    save_path=None,
):
    """Spatial map of regime labels with an optional target panel.

    Parameters
    ----------
    add_basemap : bool, default True
        Draw land / ocean / state / coastline outlines under the points
        using ``cartopy`` if it is installed. When cartopy is unavailable
        or ``add_basemap=False`` the plot falls back to a plain scatter
        on white background.
    """
    x = np.asarray(x); y = np.asarray(y)
    if convert_to_wgs84 and x.max() > 1000:
        lon, lat = _maybe_twd97_to_wgs84(x, y)
        xlabel, ylabel = "Longitude (°E)", "Latitude (°N)"
    else:
        lon, lat = x, y
        xlabel, ylabel = "Longitude", "Latitude"

    ccrs, cfeature = _try_cartopy() if add_basemap else (None, None)
    use_basemap = ccrs is not None

    ncols = 2 if target is not None else 1
    subplot_kw = {"projection": ccrs.PlateCarree()} if use_basemap else None
    fig, axes = plt.subplots(
        1, ncols, figsize=figsize, dpi=dpi, subplot_kw=subplot_kw,
    )
    if ncols == 1:
        axes = [axes]

    pad_x = (lon.max() - lon.min()) * 0.05
    pad_y = (lat.max() - lat.min()) * 0.05
    xlim = (lon.min() - pad_x, lon.max() + pad_x)
    ylim = (lat.min() - pad_y, lat.max() + pad_y)

    # ----- left panel: regime scatter ---------------------------------
    ax = axes[0]
    if use_basemap:
        _decorate_basemap(ax, cfeature, xlim, ylim)

    for g in range(1, n_groups + 1):
        mask = labels == g
        kwargs = dict(
            c=regime_color(g), s=10, alpha=0.75, edgecolors="none",
            label=f"{regime_prefix}{g} (n={int(mask.sum())})",
        )
        if use_basemap:
            kwargs["transform"] = ccrs.PlateCarree()
            kwargs["zorder"] = 5
        ax.scatter(lon[mask], lat[mask], **kwargs)

    if not use_basemap:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("SHAP-Compass Regime Spatial Distribution",
                 fontsize=12, fontweight="bold")

    # ----- right panel: target field ----------------------------------
    if target is not None:
        ax2 = axes[1]
        if use_basemap:
            _decorate_basemap(ax2, cfeature, xlim, ylim)
            scatter_kwargs = dict(transform=ccrs.PlateCarree(), zorder=5)
        else:
            scatter_kwargs = {}

        vmax = float(np.nanpercentile(target, 98))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(max(target.max(), 1.0))
        sc = ax2.scatter(
            lon, lat, c=target, cmap="YlOrRd",
            s=10, alpha=0.75, vmin=0, vmax=vmax, edgecolors="none",
            **scatter_kwargs,
        )

        # Colorbar attached to the right of ax2. For cartopy GeoAxes we
        # use the standard fig.colorbar(ax=...) which handles geo axes
        # cleanly; for plain matplotlib axes we use an axes-divider so
        # the right panel's plot area stays the same width as the left.
        if use_basemap:
            cb = fig.colorbar(sc, ax=ax2, fraction=0.035, pad=0.02)
        else:
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="3.5%", pad=0.08)
            cb = fig.colorbar(sc, cax=cax)
        cb.set_label(target_label, fontsize=9)

        if not use_basemap:
            ax2.set_xlim(xlim); ax2.set_ylim(ylim)
            ax2.set_aspect("equal")
        ax2.set_xlabel(xlabel, fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_title(f"{target_label} distribution",
                      fontsize=12, fontweight="bold")

    # ----- regime legend below the figure (one row, centred) ---------
    fig.subplots_adjust(bottom=0.18, wspace=0.12)
    legend_handles = [
        Patch(facecolor=regime_color(g), label=f"{regime_prefix}{g}")
        for g in range(1, n_groups + 1)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=min(n_groups, 8), fontsize=10, frameon=False,
        handletextpad=0.4, columnspacing=1.2,
    )

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
