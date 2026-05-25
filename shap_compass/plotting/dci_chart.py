"""DCI ranking bar chart."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from ..dci import DCI_BAND_COLORS, DCI_BANDS


def plot_dci_ranking(
    dci_df: pd.DataFrame,
    ax=None,
    figsize: tuple = (8, 6),
    dpi: int = 200,
    save_path=None,
):
    """Horizontal bar chart of DCI by feature with the four interpretation bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    df = dci_df.sort_values("DCI", ascending=True)
    colors = [DCI_BAND_COLORS[b] for b in df["band"]]

    ax.barh(range(len(df)), df["DCI"], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Directional Consistency Index (DCI)", fontsize=11)
    ax.set_title(
        "DCI Ranking — cross-regime feature direction consistency",
        fontsize=12, fontweight="bold",
        pad=14,
    )
    for x in DCI_BANDS:
        ax.axvline(x=x, color="#888", ls="--", lw=0.8, alpha=0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = [
        Patch(facecolor=DCI_BAND_COLORS["high"], label="high (>=0.75)"),
        Patch(facecolor=DCI_BAND_COLORS["medium"], label="medium (0.50-0.75)"),
        Patch(facecolor=DCI_BAND_COLORS["low"], label="low (0.25-0.50)"),
        Patch(facecolor=DCI_BAND_COLORS["context-dependent"], label="context-dependent (<0.25)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, ax
