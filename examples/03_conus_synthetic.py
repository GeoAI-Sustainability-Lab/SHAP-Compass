# -*- coding: utf-8 -*-
"""SHAP-Compass — CONUS-style synthetic case study.

Mirrors the CONUS workflow of Section 3.2 of the paper using a synthetic
dataset large enough to exercise the larger SOM configuration
(20 x 20 grid, k = 7). All values are simulated; no real Ransom et al.
(2022) data is bundled here.

Why synthetic only?
    The CONUS evaluation in the paper depends on public USGS / EPA layers
    and the curated Ransom et al. (2022) compilation; downloading and
    pre-processing those files is outside the scope of a Python package.
    This script demonstrates that the SHAP-Compass pipeline scales to
    high-dimensional cases (>= 30 features, SOM 20 x 20) and produces the
    bilayer heatmap with the multi-dimension grouping shown as Fig.11 in
    the paper.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_compass import (
    SHAPCompass,
    compute_all_metrics,
)
from shap_compass.plotting import (
    plot_dci_ranking,
    plot_bilayer_heatmap,
    plot_per_feature_unit_circle,
    plot_som_grid,
    plot_ward_dendrogram,
    plot_spatial,
)


# ---------------------------------------------------------------------------
# Feature schema — 35 synthetic features in 9 functional dimensions
# ---------------------------------------------------------------------------
FEATURE_DIMENSIONS = {
    "N source": ["Crop1974", "Crop1982", "Crop1992", "Farm_Count", "NO3_dep_85", "NO3_dep_92"],
    "Natural cover": ["NatLC_74", "NatLC_92"],
    "Soil drainage": ["DrnCls_3", "DrnCls_6", "DrnCls_9", "HydGrp_B", "Silt"],
    "Denitrification": ["WC", "AWC", "Bulk_Dens", "Org_Matter"],
    "Water cycle": ["Precip", "Recharge", "Runoff", "Stream_Dist", "BFI"],
    "Temperature": ["Temp", "PET"],
    "Soil chemistry": ["LR_As", "Mn", "Mg"],
    "Well / depth": ["Well_Depth", "Screen_Top"],
    "Social": ["Pop_Dens", "Well_Dens"],
}

FEATURE_NAMES = [f for feats in FEATURE_DIMENSIONS.values() for f in feats]
FEATURE_CODES = {name: f"UF{i + 1}" for i, name in enumerate(FEATURE_NAMES)}
J = len(FEATURE_NAMES)

print(f"[CONUS synthetic] {J} features across {len(FEATURE_DIMENSIONS)} dimensions")


# ---------------------------------------------------------------------------
# Synthetic generator — 7 attribution regimes
# ---------------------------------------------------------------------------
def make_conus_synthetic(n_total: int = 6000, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    regimes = [
        {"target_loc": 9.5, "n": 800, "feature_shift": {"Crop1992": 2.0, "Farm_Count": 1.2, "NO3_dep_92": 1.5},
         "shap_coef": {"Crop1992": 0.8, "Farm_Count": 0.5, "NO3_dep_92": 0.7, "NatLC_92": -0.4}},
        {"target_loc": 5.0, "n": 900, "feature_shift": {"Recharge": 1.6, "Precip": 1.2, "Crop1982": 0.8},
         "shap_coef": {"Recharge": 0.6, "Precip": 0.5, "Crop1982": 0.6, "AWC": -0.4}},
        {"target_loc": 2.6, "n": 850, "feature_shift": {"Well_Depth": -1.0, "DrnCls_3": 1.4, "Pop_Dens": 0.9},
         "shap_coef": {"Well_Depth": -0.6, "DrnCls_3": 0.5, "Pop_Dens": 0.4}},
        {"target_loc": 1.5, "n": 950, "feature_shift": {"DrnCls_9": -1.5, "WC": 1.0, "Org_Matter": 1.0},
         "shap_coef": {"DrnCls_9": -0.7, "WC": -0.5, "Org_Matter": -0.4}},
        {"target_loc": 0.8, "n": 850, "feature_shift": {"Temp": 1.4, "PET": 1.2, "Runoff": -0.8},
         "shap_coef": {"Temp": 0.4, "PET": 0.3, "Runoff": -0.4, "Crop1974": -0.5}},
        {"target_loc": 0.4, "n": 850, "feature_shift": {"NatLC_92": 2.0, "Org_Matter": 1.2, "Well_Depth": 1.0},
         "shap_coef": {"NatLC_92": -0.7, "Org_Matter": -0.5, "Well_Depth": -0.4}},
        {"target_loc": 0.1, "n": 800, "feature_shift": {"LR_As": 1.5, "Mn": 1.0, "BFI": 1.4},
         "shap_coef": {"LR_As": -0.5, "Mn": -0.4, "BFI": -0.4, "Crop1974": -0.4}},
    ]

    delta = n_total - sum(rg["n"] for rg in regimes)
    regimes[3]["n"] += delta

    feature_blocks, shap_blocks, target_blocks, truth_blocks = [], [], [], []
    coord_x_blocks, coord_y_blocks = [], []
    for idx, rg in enumerate(regimes, start=1):
        n = rg["n"]
        Xg = rng.standard_normal((n, J))
        for fname, shift in rg["feature_shift"].items():
            Xg[:, FEATURE_NAMES.index(fname)] += shift
        Sg = rng.standard_normal((n, J)) * 0.3
        for fname, coef in rg["shap_coef"].items():
            Sg[:, FEATURE_NAMES.index(fname)] += Xg[:, FEATURE_NAMES.index(fname)] * coef

        target_signal = Sg.sum(axis=1)
        yg = (
            rg["target_loc"]
            + 0.3 * (target_signal - target_signal.mean())
            + rng.normal(0, 0.8, n)
        )
        yg = np.clip(yg, 0.01, 25.0)

        # Pretend a US-shaped longitude / latitude swath
        cx = -98.0 + (idx - 4) * 1.5 + rng.standard_normal(n) * 2.0
        cy = 39.0 + (idx - 4) * 0.6 + rng.standard_normal(n) * 1.5

        feature_blocks.append(Xg); shap_blocks.append(Sg)
        target_blocks.append(yg); truth_blocks.append(np.full(n, idx, dtype=int))
        coord_x_blocks.append(cx); coord_y_blocks.append(cy)

    features = np.vstack(feature_blocks)
    shap_values = np.vstack(shap_blocks)
    target = np.concatenate(target_blocks)
    truth = np.concatenate(truth_blocks)
    cx = np.concatenate(coord_x_blocks); cy = np.concatenate(coord_y_blocks)
    order = rng.permutation(len(target))
    return features[order], shap_values[order], target[order], truth[order], cx[order], cy[order]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
features, shap_values, target, truth, coord_x, coord_y = make_conus_synthetic(
    n_total=6000, random_state=42,
)

print(f"[CONUS synthetic] n = {len(target)}, target mean = {target.mean():.2f}")

compass = SHAPCompass(
    features=features,
    attributions=shap_values,
    feature_names=FEATURE_NAMES,
    target=target,
)
# A 20 x 20 SOM matches the configuration of Fig.10 / Fig.11 of the paper.
results = compass.fit(som_grid=(20, 20), n_regimes=7, random_state=42)
results.summary()

OUT = Path(__file__).parent / "conus_synthetic" / "output"
IMG = OUT / "figures"
IMG.mkdir(parents=True, exist_ok=True)

n_groups = results.n_groups
ZF_g = np.zeros((n_groups, J))
ZS_g = np.zeros((n_groups, J))
for g in range(1, n_groups + 1):
    mask = results.labels == g
    if mask.any():
        ZF_g[g - 1] = results.ZF[mask].mean(axis=0)
        ZS_g[g - 1] = results.ZS[mask].mean(axis=0)

neuron_t = np.zeros(len(results.neuron_ids))
for ni in range(len(results.neuron_ids)):
    m = results._sample_to_neuron == ni
    if m.any():
        neuron_t[ni] = target[m].mean()

plot_som_grid(
    results.neuron_ids, results.neuron_labels, results.neuron_sizes,
    results.som_grid, n_groups,
    neuron_target_means=neuron_t,
    target_label="Target (synthetic)",
    regime_prefix="UG",
    figsize=(26, 8),
    save_path=IMG / "som_grid.png",
)
plt.close("all")

plot_ward_dendrogram(
    results.neuron_cossin, results.neuron_labels, n_groups,
    regime_prefix="UG",
    save_path=IMG / "ward_dendrogram.png",
)
plt.close("all")

plot_dci_ranking(results.dci, save_path=IMG / "dci_ranking.png", figsize=(9, 8))
plt.close("all")

plot_bilayer_heatmap(
    ZF_g, ZS_g, FEATURE_NAMES, n_groups,
    feature_dimensions=FEATURE_DIMENSIONS,
    feature_codes=FEATURE_CODES,
    regime_prefix="UG",
    save_path=IMG / "bilayer_feature_heatmap.png",
)
plt.close("all")

plot_per_feature_unit_circle(
    results.group_theta, results.dci, FEATURE_NAMES, n_groups,
    top_features=20,
    regime_prefix="UG",
    save_path=IMG / "per_feature_unit_circle_top20.png",
)
plt.close("all")

plot_spatial(
    results.labels, coord_x, coord_y, n_groups, target=target,
    convert_to_wgs84=False,
    regime_prefix="UG",
    target_label="Synthetic NO3-N (mg/L)",
    save_path=IMG / "spatial_distribution.png",
)
plt.close("all")

print(f"[CONUS synthetic] figures saved to {IMG}")

metrics = compute_all_metrics(
    results, target=target,
    features_raw=features, attributions_raw=shap_values,
)
print(f"[CONUS synthetic] {len(metrics)} quality metrics; eta^2 = {results.eta_sq:.3f}")

results.dci.to_csv(OUT / "dci_ranking.csv", index=False)
pd.DataFrame([{"metric": k, "value": v} for k, v in sorted(metrics.items())]).to_csv(
    OUT / "quality_metrics.csv", index=False,
)
pd.DataFrame({
    "synthetic_truth_regime": truth,
    "recovered_regime": results.labels,
    "target": target,
}).to_csv(OUT / "regime_assignments.csv", index=False)
print(f"[CONUS synthetic] outputs saved to {OUT}")
