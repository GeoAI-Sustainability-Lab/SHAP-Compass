"""Quality metrics M01-M21 for SHAP-Compass.

These 21 indicators provide a quality dashboard. They serve as a
self-check on a SHAP-Compass configuration (SOM size, k, linkage, ...).
Two hard pre-filters apply before any cross-configuration ranking:

    1. M05 == 1.0           — every SOM neuron has at least one sample.
    2. M19 min fraction >= 0.03 — the smallest regime holds >= 3 % of
                                  samples (Dalmaijer et al. 2022).

Within the valid pool, three "core" metrics are the primary selectors:

    M13 — bootstrap ARI stability
    M18 — target separability eta^2 in the low-target band
    M20 — Moran's I spatial continuity (only when coordinates are provided)

All other metrics are reported as independent diagnostics; callers should
inspect each metric on its own rather than collapsing them into a single
aggregate score.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage as sp_linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)


# ---------------------------------------------------------------------------
# Metric names and ranking directions
# ---------------------------------------------------------------------------

METRIC_COLS = [f"M{i:02d}" for i in range(1, 22)]

METRIC_NAMES = {
    "M01": "SHAP signal chain (sum SHAP ~ target R^2)",
    "M02": "Non-linearity richness (1 - mean R^2 of Z^F~Z^S)",
    "M03": "SOM topological preservation (1 - topographic error)",
    "M04": "SOM quantization accuracy (1 / (1 + QE))",
    "M05": "SOM neuron utilisation (fraction active)",
    "M06": "Signal uniqueness (1 - ARI vs attribution-only)",
    "M07": "SHAP-Compass richness (var(COSSIN) / (var + var Z^S))",
    "M08": "Silhouette coefficient",
    "M09": "Calinski-Harabasz index",
    "M10": "Davies-Bouldin index (lower is better)",
    "M11": "Cophenetic correlation",
    "M12": "Target separability eta^2",
    "M13": "Bootstrap ARI stability",
    "M14": "Out-of-sample retention eta^2(test) / eta^2(train)",
    "M15": "Within-regime SHAP sign agreement",
    "M16": "Anti-cyclicity (1 - ARI vs target-percentile groups)",
    "M17": "K-optimality eta^2(k) / max eta^2",
    "M18": "Low-target band eta^2",
    "M19": "Regime size evenness (Pielou's J, min fraction >= 3%)",
    "M20": "Target adjacent min-gap / IQR",
    "M21": "Inter-regime mechanism diversity",
}

CORE_METRICS = ("M13", "M18", "M20")
"""Three primary selectors applied within the valid pool."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eta2(y: np.ndarray, g: np.ndarray) -> float:
    """eta^2 = SS_between / SS_total of ``y`` partitioned by ``g``."""
    uniq = np.unique(g)
    groups = [y[g == c] for c in uniq if (g == c).sum() > 0]
    if len(groups) < 2:
        return 0.0
    grand_mean = float(y.mean())
    ss_b = sum(len(grp) * (float(grp.mean()) - grand_mean) ** 2 for grp in groups)
    ss_t = float(np.sum((y - grand_mean) ** 2))
    return float(ss_b / ss_t) if ss_t > 0 else 0.0


# ---------------------------------------------------------------------------
# M01, M02 — SHAP signal-chain checks
# ---------------------------------------------------------------------------

def compute_M01(attributions_raw: np.ndarray, target: np.ndarray) -> float:
    """R^2 between row-summed SHAP attributions and the target."""
    shap_sum = attributions_raw.sum(axis=1)
    valid = ~np.isnan(shap_sum) & ~np.isnan(target)
    if valid.sum() < 3:
        return 0.0
    r, _ = stats.pearsonr(shap_sum[valid], target[valid])
    return float(r ** 2)


def compute_M02(ZF: np.ndarray, ZS: np.ndarray) -> float:
    """1 - mean R^2 of feature-wise linear Z^F ~ Z^S fits."""
    n_features = ZF.shape[1]
    r2 = []
    for j in range(n_features):
        _, _, rv, _, _ = linregress(ZF[:, j], ZS[:, j])
        r2.append(rv ** 2)
    return float(1 - np.mean(r2))


# ---------------------------------------------------------------------------
# M03, M04, M05 — SOM diagnostics
# ---------------------------------------------------------------------------

def compute_M03(som, data: np.ndarray) -> float:
    return float(1 - float(som.topographic_error(data)))


def compute_M04(som, data: np.ndarray) -> float:
    return float(1 / (1 + float(som.quantization_error(data))))


def compute_M05(neuron_sizes: np.ndarray, som_grid: tuple) -> float:
    n_total = som_grid[0] * som_grid[1]
    return float(len(neuron_sizes) / n_total)


# ---------------------------------------------------------------------------
# M06, M07 — Signal-vs-attribution-only comparisons
# ---------------------------------------------------------------------------

def compute_M06(cossin: np.ndarray, neuron_labels: np.ndarray, n_clusters: int) -> float:
    """1 - ARI between Ward(COSSIN) and Ward(sin only)."""
    if len(cossin) <= n_clusters:
        return 0.5
    n_features = cossin.shape[1] // 2
    sin_only = cossin[:, n_features:]
    try:
        alt = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(sin_only)
    except Exception:
        return 0.5
    return float(np.clip(1 - adjusted_rand_score(neuron_labels, alt), 0, 1))


def compute_M07(cossin: np.ndarray, ZS_neurons: np.ndarray) -> float:
    var_D = float(np.var(cossin))
    var_S = float(np.var(ZS_neurons))
    denom = var_D + var_S
    return float(var_D / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# M08 - M11 — internal cluster validity
# ---------------------------------------------------------------------------

def compute_M08(cossin: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(silhouette_score(cossin, labels))
    except Exception:
        return 0.0


def compute_M09(cossin: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(calinski_harabasz_score(cossin, labels))
    except Exception:
        return 0.0


def compute_M10(cossin: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(davies_bouldin_score(cossin, labels))
    except Exception:
        return 999.0


def compute_M11(cossin: np.ndarray, linkage_method: str = "ward") -> float:
    try:
        Z = sp_linkage(cossin, method=linkage_method)
        c, _ = cophenet(Z, pdist(cossin))
        return float(c)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# M12 - M15 — external validation
# ---------------------------------------------------------------------------

def compute_M12(labels: np.ndarray, target: np.ndarray) -> float:
    valid = labels > 0
    if valid.sum() < 3:
        return 0.0
    return _eta2(target[valid], labels[valid])


def compute_M13(
    cossin: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    linkage_method: str = "ward",
    n_bootstrap: int = 50,
    random_state: int = 42,
) -> float:
    n_units = len(cossin)
    rng = np.random.default_rng(random_state)
    aris: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_units, n_units, replace=True)
        boot_data = cossin[idx]
        try:
            boot_labels = AgglomerativeClustering(
                n_clusters=n_clusters, linkage=linkage_method
            ).fit_predict(boot_data)
        except Exception:
            continue
        mapped = np.array([
            boot_labels[np.argmin(np.linalg.norm(cossin[i] - boot_data, axis=1))]
            for i in range(n_units)
        ])
        aris.append(adjusted_rand_score(labels, mapped))

    return float(np.mean(aris)) if aris else 0.0


def compute_M14(
    features: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    is_train: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> float:
    from sklearn.ensemble import RandomForestClassifier

    valid = labels > 0
    X = features[valid]; y = target[valid]
    g = labels[valid]; tr = is_train[valid]; te = ~tr

    if tr.sum() < 10 or te.sum() < 10 or len(np.unique(g[tr])) < 2:
        return 0.0

    clf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
    clf.fit(X[tr], g[tr])
    g_pred = clf.predict(X[te])

    eta_train = _eta2(y[tr], g[tr])
    eta_test = _eta2(y[te], g_pred)
    return float(np.clip(eta_test / eta_train, 0, 1)) if eta_train > 0 else 0.0


def compute_M15(attributions_raw: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
    valid = labels > 0
    s = attributions_raw[valid]; g_arr = labels[valid]
    cs: list[float] = []
    for c in range(1, n_clusters + 1):
        grp = s[g_arr == c]
        if len(grp) < 2:
            continue
        direction = np.sign(grp.mean(axis=0))
        direction[direction == 0] = 1
        cs.append(float((np.sign(grp) == direction).mean(axis=1).mean()))
    if not cs:
        return 0.0
    return float(np.clip((np.mean(cs) - 0.5) / 0.5, 0, 1))


# ---------------------------------------------------------------------------
# M16, M17, M18 — mechanism / k-selection
# ---------------------------------------------------------------------------

def compute_M16(labels: np.ndarray, target: np.ndarray, n_clusters: int) -> float:
    valid = labels > 0
    if valid.sum() < n_clusters * 2:
        return 0.0
    target_v = target[valid]; g_v = labels[valid]
    thresholds = np.percentile(target_v, np.linspace(0, 100, n_clusters + 1)[1:-1])
    target_rank = np.digitize(target_v, thresholds)
    return float(np.clip(1 - adjusted_rand_score(g_v, target_rank), 0, 1))


def compute_M17(
    cossin: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    linkage_method: str = "ward",
    k_range=range(2, 11),
) -> float:
    valid = labels > 0
    if valid.sum() < 3:
        return 0.0
    eta2_k = _eta2(target[valid], labels[valid])
    eta2_max = eta2_k
    for kk in k_range:
        if kk > len(cossin) or kk == n_clusters:
            continue
        try:
            kl = AgglomerativeClustering(n_clusters=kk, linkage=linkage_method).fit_predict(cossin)
            kk_eta = _eta2(target[valid], (kl + 1)[valid] if len(kl) == len(valid) else kl)
            eta2_max = max(eta2_max, kk_eta)
        except Exception:
            continue
    return float(np.clip(eta2_k / eta2_max, 0, 1)) if eta2_max > 0 else 0.0


def compute_M18(
    labels: np.ndarray,
    target: np.ndarray,
    features: np.ndarray,
    n_clusters: int,
    percentile: float = 33.33,
) -> float:
    valid = labels > 0
    target_v = target[valid]; g_v = labels[valid]; X_v = features[valid]
    threshold = np.percentile(target_v, percentile)
    mask_low = target_v <= threshold
    if mask_low.sum() < n_clusters * 2:
        return 0.0

    g_low = g_v[mask_low]; X_low = X_v[mask_low]
    n_features = features.shape[1]
    eta2_list: list[float] = []
    for j in range(n_features):
        groups = [
            X_low[g_low == c, j]
            for c in range(1, n_clusters + 1)
            if (g_low == c).sum() >= 2
        ]
        if len(groups) < 2:
            continue
        try:
            H, _ = stats.kruskal(*groups)
            n_t = sum(len(g) for g in groups); k_a = len(groups)
            eta2 = max(0.0, (H - k_a + 1) / (n_t - k_a)) if n_t > k_a else 0.0
            eta2_list.append(eta2)
        except Exception:
            pass
    return float(np.mean(eta2_list)) if eta2_list else 0.0


# ---------------------------------------------------------------------------
# M19, M20, M21 — interpretability
# ---------------------------------------------------------------------------

def compute_M19(labels: np.ndarray, n_clusters: int, min_frac_threshold: float = 0.03) -> float:
    """Pielou evenness of regime sizes with the 3 % minimum-fraction floor."""
    n = int((labels > 0).sum())
    if n == 0 or n_clusters <= 1:
        return 0.0
    counts = np.array([(labels == c).sum() for c in range(1, n_clusters + 1)], dtype=float)
    min_fraction = float(counts.min() / n)
    if min_fraction < min_frac_threshold:
        return 0.0
    probs = counts / n
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(np.clip(entropy / np.log(float(n_clusters)), 0.0, 1.0))


def compute_M19_min_frac(labels: np.ndarray, n_clusters: int) -> float:
    n = int((labels > 0).sum())
    if n == 0:
        return 0.0
    counts = [(labels == c).sum() for c in range(1, n_clusters + 1)]
    return float(min(counts) / n)


def compute_M20(labels: np.ndarray, target: np.ndarray, n_clusters: int) -> float:
    """Smallest adjacent gap of regime target-means, normalised by IQR(target)."""
    valid = labels > 0
    target_v = target[valid]; g_v = labels[valid]
    means = sorted([
        float(np.nanmean(target_v[g_v == c]))
        for c in range(1, n_clusters + 1)
        if (g_v == c).sum() > 0
    ])
    if len(means) < 2:
        return 0.0
    gaps = np.diff(means)
    iqr = float(np.percentile(target_v, 75) - np.percentile(target_v, 25))
    if iqr <= 0:
        return 0.0
    return float(np.clip(gaps.min() / iqr, 0.0, 2.0))


def compute_M21(cossin: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
    """Mean (1 - |cosine similarity|) between regime centroids in COSSIN space."""
    means: list[np.ndarray] = []
    for c in range(1, n_clusters + 1):
        mask = labels == c
        if mask.sum() == 0:
            continue
        means.append(cossin[mask].mean(axis=0))
    if len(means) < 2:
        return 0.0
    M = np.array(means)
    norms = np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    sim = (M / norms) @ (M / norms).T

    total, count = 0.0, 0
    n_g = len(means)
    for i in range(n_g):
        for j in range(i + 1, n_g):
            total += 1.0 - abs(float(sim[i, j]))
            count += 1
    return float(total / count) if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Convenience: compute every available metric on a SHAPCompassResults
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results,
    target: np.ndarray,
    features_raw: np.ndarray | None = None,
    attributions_raw: np.ndarray | None = None,
    som=None,
    data_for_som: np.ndarray | None = None,
    is_train: np.ndarray | None = None,
) -> dict:
    """Compute every metric whose required inputs are available."""
    metrics: dict[str, float] = {}
    n_clusters = results.n_groups
    labels = results.labels

    if attributions_raw is not None and target is not None:
        metrics["M01"] = compute_M01(attributions_raw, target)
    if results.ZF is not None and results.ZS is not None:
        metrics["M02"] = compute_M02(results.ZF, results.ZS)

    if som is not None and data_for_som is not None:
        metrics["M03"] = compute_M03(som, data_for_som)
        metrics["M04"] = compute_M04(som, data_for_som)
    if results.neuron_sizes is not None and results.som_grid is not None:
        metrics["M05"] = compute_M05(results.neuron_sizes, results.som_grid)

    if results.neuron_cossin is not None and results.neuron_labels is not None:
        metrics["M06"] = compute_M06(results.neuron_cossin, results.neuron_labels, n_clusters)
        n_feat = results.neuron_cossin.shape[1] // 2
        metrics["M07"] = compute_M07(results.neuron_cossin, results.neuron_cossin[:, n_feat:])

        nc = results.neuron_cossin
        nl = results.neuron_labels
        metrics["M08"] = compute_M08(nc, nl)
        metrics["M09"] = compute_M09(nc, nl)
        metrics["M10"] = compute_M10(nc, nl)
        metrics["M11"] = compute_M11(nc)

    if labels is not None and target is not None:
        metrics["M12"] = compute_M12(labels, target)
    if results.neuron_cossin is not None and results.neuron_labels is not None:
        metrics["M13"] = compute_M13(results.neuron_cossin, results.neuron_labels, n_clusters)
    if (features_raw is not None and target is not None
            and labels is not None and is_train is not None):
        metrics["M14"] = compute_M14(features_raw, target, labels, is_train, n_clusters)
    if attributions_raw is not None and labels is not None:
        metrics["M15"] = compute_M15(attributions_raw, labels, n_clusters)

    if labels is not None and target is not None:
        metrics["M16"] = compute_M16(labels, target, n_clusters)
    if labels is not None and target is not None:
        metrics["M18"] = compute_M18(
            labels, target,
            features_raw if features_raw is not None else results.ZF,
            n_clusters,
        )

    if labels is not None:
        metrics["M19"] = compute_M19(labels, n_clusters)
        metrics["M19_min_frac"] = compute_M19_min_frac(labels, n_clusters)
    if labels is not None and target is not None:
        metrics["M20"] = compute_M20(labels, target, n_clusters)
    if results.neuron_cossin is not None and results.neuron_labels is not None:
        metrics["M21"] = compute_M21(results.neuron_cossin, results.neuron_labels, n_clusters)

    return metrics
