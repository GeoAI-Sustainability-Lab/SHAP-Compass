"""Directional Consistency Index (DCI).

DCI quantifies how consistently a feature's attribution direction points
across all regimes. For each feature j let theta_{g,j} be the regime
g's centroid direction. After axial doubling (2-theta transform, so that
theta and theta + pi are identified as the same mechanism axis), DCI is the
resultant length of the regime centroids on the doubled circle:

    DCI_j = sqrt( mean_g cos(2 theta_{g,j})^2 + mean_g sin(2 theta_{g,j})^2 )

with DCI_j in [0, 1]. Values near 1 indicate a universal driver (all
regimes agree on the attribution direction), while values near 0 indicate
a context-dependent driver (regime-specific mechanisms).

The four interpretation bands are:

    >= 0.75 : high consistency       (green)
    0.50 - 0.75 : medium               (yellow)
    0.25 - 0.50 : low                  (orange)
    < 0.25 : strongly context-dependent (red)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import circular_R_axial


DCI_BANDS = (0.25, 0.50, 0.75)
"""Lower bounds of (low, medium, high) bands. DCI < 0.25 is the fourth band."""

DCI_BAND_COLORS = {
    "context-dependent": "#D32F2F",
    "low": "#F57C00",
    "medium": "#FBC02D",
    "high": "#2E7D32",
}


def dci_band(value: float) -> str:
    """Return the named DCI band for a single DCI value."""
    if value >= DCI_BANDS[2]:
        return "high"
    if value >= DCI_BANDS[1]:
        return "medium"
    if value >= DCI_BANDS[0]:
        return "low"
    return "context-dependent"


def compute_dci(
    group_theta: np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute axial DCI per feature across regime centroids.

    Parameters
    ----------
    group_theta : np.ndarray, shape (n_regimes, n_features)
        Per-regime mean direction angles (radians).
    feature_names : list of str, optional
        Names for the output table. Defaults to ``F1, F2, ...``.

    Returns
    -------
    pd.DataFrame
        Columns ``feature``, ``DCI``, ``R_axial``, ``rank``, ``band``.
        Sorted by DCI descending.
    """
    n_groups, n_features = group_theta.shape

    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(n_features)]

    records = []
    for j in range(n_features):
        thetas = group_theta[:, j]
        R_ax = circular_R_axial(thetas)
        records.append({
            "feature": feature_names[j],
            "DCI": float(R_ax),
            "R_axial": float(R_ax),
        })

    df = pd.DataFrame(records).sort_values("DCI", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["band"] = df["DCI"].apply(dci_band)
    return df


def compute_dci_within_group(
    neuron_theta: np.ndarray,
    neuron_groups: np.ndarray,
    n_groups: int,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Per-regime, per-feature DCI computed on neuron centroids.

    Measures directional concentration of the SOM neurons assigned to each
    regime — high values indicate the regime's neurons agree on direction
    for that feature.
    """
    n_features = neuron_theta.shape[1]
    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(n_features)]

    records = []
    for g in range(1, n_groups + 1):
        mask = neuron_groups == g
        if mask.sum() < 2:
            continue
        for j in range(n_features):
            R_ax = circular_R_axial(neuron_theta[mask, j])
            records.append({
                "group": g,
                "feature": feature_names[j],
                "DCI_within": float(R_ax),
                "R_axial_within": float(R_ax),
            })

    return pd.DataFrame(records)
