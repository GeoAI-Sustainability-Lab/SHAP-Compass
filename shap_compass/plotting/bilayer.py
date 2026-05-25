"""Bilayer feature heatmap and per-feature unit-circle plot.

These two functions reproduce the headline regime-level figures of the
SHAP-Compass paper (Fig.7 / Fig.11 and Fig.9 / Fig.13). They replace the
rose / polar diagrams used in earlier internal versions because the
bilayer split-cell layout makes the Z^F (feature value) vs Z^S
(attribution) coupling per regime visible at a glance and scales to
dozens of features.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from ..dci import DCI_BAND_COLORS, dci_band


# ---------------------------------------------------------------------------
# Bilayer feature heatmap  (Fig.7 / Fig.11)
# ---------------------------------------------------------------------------

def plot_bilayer_heatmap(
    ZF_group: np.ndarray,
    ZS_group: np.ndarray,
    feature_names: Sequence[str],
    n_groups: int,
    *,
    feature_dimensions: Optional[Mapping[str, Sequence[str]]] = None,
    feature_codes: Optional[Mapping[str, str]] = None,
    regime_prefix: str = "R",
    annotate_threshold: float = 0.5,
    annotate_fontsize: float = 8.0,
    cmap: str = "RdBu_r",
    vmax: Optional[float] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 200,
    save_path=None,
):
    """Regimes x features split-cell heatmap.

    Each cell is split horizontally into two half-cells:

        upper half = Z^F   (regime mean of standardised feature value)
        lower half = Z^S   (regime mean of standardised SHAP attribution)

    Both halves share a symmetric ``RdBu_r`` colourmap so that red marks
    positive standardised values and blue marks negative ones. A cell is
    annotated with the (Z^F, Z^S) values only when ``|Z^F| >= annotate_threshold``
    -- this keeps the figure legible at large J.

    Parameters
    ----------
    ZF_group, ZS_group : np.ndarray, shape (n_groups, n_features)
        Regime means of Z^F and Z^S.
    feature_names : list of str
        Feature labels (length n_features).
    n_groups : int
        Number of regimes.
    feature_dimensions : mapping {dimension name -> ordered features}, optional
        When provided, columns are grouped by functional dimension in the
        order of the mapping. Features not listed in any dimension are
        appended at the end. Vertical separator lines mark dimension
        boundaries and dimension names are written above the heatmap.
        Matches the layout of Fig.7 / Fig.11 in the paper.
    feature_codes : mapping {feature name -> short code}, optional
        Short code (e.g. ``"TF1"``) prefixed to the column tick label.
    regime_prefix : str
        Prefix for the y-tick labels (default ``"R"`` produces R1, R2, ...).
        Use ``"TG"`` for Taiwan regimes and ``"UG"`` for CONUS regimes.
    annotate_threshold : float
        Annotate the cell only when ``|Z^F| >= annotate_threshold``.
    annotate_fontsize : float
        Font size of cell annotations.
    cmap : str
        A matplotlib colormap name. Default symmetric diverging map.
    vmax : float, optional
        Symmetric colour limit. Default ``max(|ZF|, |ZS|)``.
    figsize : (width, height), optional
        Default scales with feature and regime counts.
    """
    ZF_group = np.asarray(ZF_group, dtype=float)
    ZS_group = np.asarray(ZS_group, dtype=float)
    feature_names = list(feature_names)
    n_features_total = len(feature_names)

    if ZF_group.shape != (n_groups, n_features_total):
        raise ValueError(
            f"ZF_group shape {ZF_group.shape} != (n_groups={n_groups}, "
            f"n_features={n_features_total})"
        )
    if ZS_group.shape != ZF_group.shape:
        raise ValueError("ZS_group must have the same shape as ZF_group")

    # Column ordering -------------------------------------------------------
    if feature_dimensions is not None:
        sorted_feats: list[str] = []
        dim_sizes: list[tuple[str, int]] = []
        for dim_name, feats in feature_dimensions.items():
            sub = [f for f in feats if f in feature_names]
            if sub:
                sorted_feats.extend(sub)
                dim_sizes.append((dim_name, len(sub)))
        leftover = [f for f in feature_names if f not in sorted_feats]
        if leftover:
            sorted_feats.extend(leftover)
            dim_sizes.append(("(other)", len(leftover)))
    else:
        sorted_feats = list(feature_names)
        dim_sizes = []

    col_index = [feature_names.index(f) for f in sorted_feats]
    zf_s = ZF_group[:, col_index]
    zs_s = ZS_group[:, col_index]
    N = len(sorted_feats)

    if vmax is None:
        vmax = float(max(np.abs(zf_s).max(), np.abs(zs_s).max(), 1e-9))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    if figsize is None:
        figsize = (max(12.0, 0.35 * N + 4), max(5.0, 0.9 * n_groups + 3))

    gap_col = 0.08
    gap_row = 0.14
    gap_inner = 0.03

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for k in range(n_groups):
        for i in range(N):
            x0 = i - 0.5 + gap_col / 2
            w = 1 - gap_col
            y0 = k - 0.5 + gap_row / 2
            h_total = 1 - gap_row
            h_each = h_total / 2 - gap_inner / 2

            # After y-axis inversion (set_ylim(K - 0.5, -2.5)) low y values
            # appear at the top. So the rectangle at y0 is the upper half.
            ax.add_patch(Rectangle(
                (x0, y0), w, h_each,
                facecolor=cmap_obj(norm(zf_s[k, i])), edgecolor="none",
            ))
            ax.add_patch(Rectangle(
                (x0, y0 + h_each + gap_inner), w, h_each,
                facecolor=cmap_obj(norm(zs_s[k, i])), edgecolor="none",
            ))

            if abs(zf_s[k, i]) >= annotate_threshold:
                ax.text(
                    i, y0 + h_each / 2,
                    f"{zf_s[k, i]:+.1f}",
                    ha="center", va="center", fontsize=annotate_fontsize,
                )
                ax.text(
                    i, y0 + h_each + gap_inner + h_each / 2,
                    f"{zs_s[k, i]:+.1f}",
                    ha="center", va="center", fontsize=annotate_fontsize,
                )

    # Y-axis: regimes
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(
        [f"{regime_prefix}{g + 1}" for g in range(n_groups)],
        fontsize=14, fontweight="bold",
    )
    ax.tick_params(axis="y", length=0)

    # X-axis: features (codes optional)
    ax.set_xticks(range(N))
    if feature_codes is not None:
        labels_x = [
            f"{feature_codes.get(f, '')}  {f}".strip() for f in sorted_feats
        ]
    else:
        labels_x = list(sorted_feats)
    ax.set_xticklabels(labels_x, rotation=90, fontsize=9, ha="center")
    ax.tick_params(axis="x", length=0, pad=2)

    # Dimension labels and separators
    cum = 0
    for dim, sz in dim_sizes:
        left = cum - 0.5
        if cum > 0:
            ax.vlines(left, ymin=-0.5, ymax=n_groups - 0.5, color="black", lw=1.0)
        ax.text(
            cum + sz / 2 - 0.5, -1.0, dim,
            ha="center", va="top",
            fontsize=11, fontweight="bold",
        )
        cum += sz

    ax.hlines(-0.5, xmin=-0.5, xmax=N - 0.5, color="black", lw=0.8)

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(n_groups - 0.5, -2.5 if dim_sizes else -1.0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax, fraction=0.015, pad=0.005, shrink=0.7, location="right",
    )
    cbar.set_label("Z-score (red = positive, blue = negative)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.text(
        0.5, 0.01,
        "Upper half-cells: Z^F (standardised feature value)   |   "
        "Lower half-cells: Z^S (standardised SHAP attribution)   |   "
        f"Cells with |Z^F| >= {annotate_threshold:g} are annotated",
        ha="center", fontsize=10, style="italic",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Per-feature unit-circle plot  (Fig.9 / Fig.13)
# ---------------------------------------------------------------------------

def plot_per_feature_unit_circle(
    group_theta: np.ndarray,
    dci_df: pd.DataFrame,
    feature_names: Sequence[str],
    n_groups: int,
    *,
    top_features: Optional[int] = None,
    sort_by: str = "dci",
    regime_prefix: str = "R",
    figsize: Optional[tuple] = None,
    dpi: int = 200,
    save_path=None,
):
    """One unit-circle subplot per feature, sorted by descending DCI.

    Each panel shows where the K regime centroids sit on the unit circle
    for a given feature. The subplot border is colour-coded by DCI band:

        DCI >= 0.75 : green (high consistency)
        0.50 - 0.75 : yellow (medium)
        0.25 - 0.50 : orange (low)
        < 0.25      : red (context-dependent)

    Parameters
    ----------
    group_theta : np.ndarray, shape (n_groups, n_features)
        Regime centroid direction angles in radians.
    dci_df : pd.DataFrame
        Output of :func:`shap_compass.compute_dci`, with columns
        ``feature``, ``DCI``, ``band``. Used to order and colour-band
        panels.
    feature_names : list of str
        Feature labels (length n_features).
    n_groups : int
        Number of regimes (rows of ``group_theta``).
    top_features : int, optional
        Show only the top-N features by mean abs DCI rank. When ``None``
        (default) all features are shown -- recommended for J <= 30.
        For Fig.13 in the paper the value used was 20.
    sort_by : ``"dci"`` or ``"feature"``
        Column ordering. ``"dci"`` (default) reproduces Fig.9 / Fig.13.
    """
    from ._palette import regime_color

    feature_names = list(feature_names)
    group_theta = np.asarray(group_theta, dtype=float)

    if sort_by == "dci":
        ordered = dci_df.sort_values("DCI", ascending=False)["feature"].tolist()
    else:
        ordered = list(feature_names)

    if top_features is not None:
        ordered = ordered[:top_features]

    n_panels = len(ordered)
    if n_panels == 0:
        raise ValueError("No features to plot.")
    cols = min(5, n_panels)
    rows = (n_panels + cols - 1) // cols

    if figsize is None:
        figsize = (3.0 * cols, 3.2 * rows + 1.0)

    fig, axes = plt.subplots(
        rows, cols, figsize=figsize, dpi=dpi,
        subplot_kw=dict(projection="polar"),
    )
    axes = np.atleast_1d(axes).ravel()

    dci_lookup = dict(zip(dci_df["feature"], dci_df["DCI"]))
    band_lookup = dict(zip(dci_df["feature"], dci_df["band"]))

    theta_circle = np.linspace(0, 2 * np.pi, 200)
    one = np.ones_like(theta_circle)

    for idx, feat in enumerate(ordered):
        ax = axes[idx]
        if feat not in feature_names:
            ax.set_visible(False)
            continue
        j = feature_names.index(feat)
        dci_v = float(dci_lookup.get(feat, 0.0))
        band = band_lookup.get(feat, dci_band(dci_v))
        border = DCI_BAND_COLORS[band]

        ax.plot(theta_circle, one, color="#bbbbbb", lw=0.8, zorder=1)

        for g in range(n_groups):
            t = float(group_theta[g, j])
            color = regime_color(g + 1)
            ax.plot([t, t], [0, 1], color=color, lw=2.0, alpha=0.85, zorder=3)
            ax.plot(t, 1, "o", color=color, markersize=6, zorder=4)

        ax.set_ylim(0, 1.15)
        ax.set_rticks([])
        ax.tick_params(labelsize=6)
        ax.set_title(
            f"{feat}\nDCI={dci_v:.2f}",
            fontsize=9, fontweight="bold", pad=8,
        )

        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(3.0)

    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color=regime_color(g + 1), lw=3,
               label=f"{regime_prefix}{g + 1}")
        for g in range(n_groups)
    ]
    border_items = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=DCI_BAND_COLORS["high"],
                      linewidth=3, label="DCI >= 0.75 (high)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white",
                      edgecolor=DCI_BAND_COLORS["medium"],
                      linewidth=3, label="0.50-0.75 (medium)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=DCI_BAND_COLORS["low"],
                      linewidth=3, label="0.25-0.50 (low)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white",
                      edgecolor=DCI_BAND_COLORS["context-dependent"],
                      linewidth=3, label="< 0.25 (context-dependent)"),
    ]
    fig.legend(
        handles=legend_items + border_items,
        loc="lower center",
        ncol=min(8, n_groups + 4),
        fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Per-Feature Regime Direction (sorted by descending DCI)",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig
