"""Ward hierarchical clustering and label utilities for SHAP-Compass."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def ward_cluster(cossin: np.ndarray, n_clusters: int) -> np.ndarray:
    """Ward linkage on a SHAP-Compass (cos theta, sin theta) matrix.

    Euclidean distance on the doubled (cos theta, sin theta) representation
    is monotone in 2 (1 - cos delta_theta) summed over features, so Ward
    minimizes circular dissimilarity and handles the 0 / 2 pi wrap by
    construction.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(cossin)


def remap_by_target(labels: np.ndarray, target: np.ndarray, descending: bool = True) -> np.ndarray:
    """Relabel groups by mean target value.

    With ``descending=True`` (the package default) the regime with the
    highest mean target is labelled 1, the next-highest 2, and so on.
    """
    unique = np.unique(labels)
    means = {c: float(target[labels == c].mean()) for c in unique}
    sorted_clusters = sorted(means, key=means.get, reverse=descending)
    remap = {old: new + 1 for new, old in enumerate(sorted_clusters)}
    return np.array([remap[l] for l in labels])


def eta_squared(labels: np.ndarray, target: np.ndarray) -> float:
    """eta^2 = SS_between / SS_total between regimes and a target."""
    grand_mean = float(target.mean())
    ss_total = float(np.sum((target - grand_mean) ** 2))
    if ss_total == 0:
        return 0.0

    ss_between = 0.0
    for c in np.unique(labels):
        mask = labels == c
        ss_between += float(mask.sum()) * (float(target[mask].mean()) - grand_mean) ** 2
    return float(ss_between / ss_total)


def eta_squared_scan(cossin: np.ndarray, target: np.ndarray, k_range=range(2, 13)):
    """Scan k values and report eta^2 of Ward clusters against a target."""
    out: dict[int, float] = {}
    for k in k_range:
        labels = ward_cluster(cossin, n_clusters=k)
        out[k] = eta_squared(labels, target)
    return out
