# -*- coding: utf-8 -*-
"""SHAP-Compass — Taiwan-style synthetic case study.

Reproduces the *workflow* of the paper's Taiwan case study (Section 3.1)
on a fully synthetic dataset: 2,375 samples, 17 features grouped into 7
functional dimensions, SOM 9 x 9, k = 6 attribution regimes.

The real EPA groundwater monitoring data used in the paper cannot be
redistributed because access is restricted to permit holders; this script
therefore *simulates* a comparable directional structure so that the
SHAP-Compass package can be evaluated end-to-end without any data
download. Feature names, dimension groupings, regime sizes, and target
distribution are illustrative only.
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
    intensity_stratify_from_results,
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
# 1. Feature schema -- 17 features in 7 functional dimensions
# ---------------------------------------------------------------------------
FEATURE_DIMENSIONS = {
    "N source": ["Agri_Ratio", "Livestock_Ratio", "Agri_Gradient", "Livestock_Gradient", "Household_Density"],
    "Natural cover": ["Forest_Ratio", "Vegetation_Index"],
    "Soil drainage": ["Clay_Sand_Ratio"],
    "Denitrification": ["AWC", "Denitrification_Pot", "Infiltration"],
    "Water flux": ["Rainfall", "Leaching_Risk"],
    "Heat": ["Temperature", "Solar_Radiation"],
    "Topography": ["Elevation", "Slope"],
}

FEATURE_NAMES = [f for feats in FEATURE_DIMENSIONS.values() for f in feats]
FEATURE_CODES = {name: f"F{i + 1}" for i, name in enumerate(FEATURE_NAMES)}
J = len(FEATURE_NAMES)  # 17

assert J == 17, f"Expected 17 features, got {J}"

print(f"[Taiwan synthetic] {J} features across {len(FEATURE_DIMENSIONS)} dimensions")


# ---------------------------------------------------------------------------
# 2. Synthetic data generator -- 6 attribution regimes
# ---------------------------------------------------------------------------
def make_taiwan_synthetic(n_total: int = 2375, random_state: int = 42):
    """Generate 6 attribution regimes with distinct directional signatures."""
    rng = np.random.default_rng(random_state)
    regimes = [
        {  # R1 — high target, agri / livestock driven
            "n": 340, "target_loc": 8.0, "target_scale": 1.8,
            "feature_shift": {"Agri_Ratio": 2.0, "Livestock_Ratio": 1.5,
                              "Household_Density": 0.8},
            "shap_coef": {"Agri_Ratio": 0.9, "Livestock_Ratio": 0.7,
                          "Forest_Ratio": -0.4},
        },
        {  # R2 — high target, agri but high leaching risk
            "n": 410, "target_loc": 5.5, "target_scale": 1.4,
            "feature_shift": {"Agri_Ratio": 1.2, "Leaching_Risk": 1.4,
                              "Rainfall": 0.9},
            "shap_coef": {"Agri_Ratio": 0.7, "Leaching_Risk": 0.8,
                          "Rainfall": 0.5, "AWC": -0.4},
        },
        {  # R3 — mid target, agri + temperature
            "n": 450, "target_loc": 2.5, "target_scale": 0.9,
            "feature_shift": {"Agri_Ratio": 0.8, "Temperature": 1.0,
                              "Solar_Radiation": 0.6},
            "shap_coef": {"Agri_Ratio": 0.5, "Temperature": 0.4,
                          "Denitrification_Pot": -0.5},
        },
        {  # R4 — mid target, soil drainage-driven
            "n": 420, "target_loc": 1.8, "target_scale": 0.8,
            "feature_shift": {"Clay_Sand_Ratio": -1.3, "AWC": -0.9,
                              "Infiltration": 1.1},
            "shap_coef": {"Clay_Sand_Ratio": -0.6, "AWC": -0.5,
                          "Infiltration": 0.6},
        },
        {  # R5 — low target, forest dominated
            "n": 360, "target_loc": 0.6, "target_scale": 0.4,
            "feature_shift": {"Forest_Ratio": 2.0, "Vegetation_Index": 1.6,
                              "Elevation": 1.2, "Slope": 1.0},
            "shap_coef": {"Forest_Ratio": -0.7, "Vegetation_Index": -0.6,
                          "Agri_Ratio": -0.4},
        },
        {  # R6 — very low target, mountainous
            "n": 395, "target_loc": 0.2, "target_scale": 0.2,
            "feature_shift": {"Elevation": 2.4, "Slope": 1.9,
                              "Forest_Ratio": 1.7},
            "shap_coef": {"Elevation": -0.7, "Slope": -0.5,
                          "Rainfall": -0.4, "Temperature": -0.5},
        },
    ]

    delta = n_total - sum(rg["n"] for rg in regimes)
    regimes[2]["n"] += delta

    feature_blocks: list[np.ndarray] = []
    shap_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    coord_x_blocks: list[np.ndarray] = []
    coord_y_blocks: list[np.ndarray] = []

    for idx, rg in enumerate(regimes, start=1):
        n = rg["n"]
        Xg = rng.standard_normal((n, J))
        for fname, shift in rg["feature_shift"].items():
            Xg[:, FEATURE_NAMES.index(fname)] += shift

        Sg = rng.standard_normal((n, J)) * 0.25
        for fname, coef in rg["shap_coef"].items():
            Sg[:, FEATURE_NAMES.index(fname)] += Xg[:, FEATURE_NAMES.index(fname)] * coef

        # Target driven by attributions plus regime baseline
        target_signal = Sg.sum(axis=1)
        yg = (
            rg["target_loc"]
            + 0.35 * (target_signal - target_signal.mean())
            + rng.normal(0, rg["target_scale"], n)
        )
        yg = np.clip(yg, 0.01, 25.0)

        # Synthetic spatial coordinates that vary by regime
        cx = 121.0 + 0.8 * rng.standard_normal(n) + (idx - 3) * 0.3
        cy = 23.5 + 0.7 * rng.standard_normal(n) + (idx - 3) * 0.25

        feature_blocks.append(Xg)
        shap_blocks.append(Sg)
        target_blocks.append(yg)
        truth_blocks.append(np.full(n, idx, dtype=int))
        coord_x_blocks.append(cx)
        coord_y_blocks.append(cy)

    features = np.vstack(feature_blocks)
    shap_values = np.vstack(shap_blocks)
    target = np.concatenate(target_blocks)
    truth = np.concatenate(truth_blocks)
    coord_x = np.concatenate(coord_x_blocks)
    coord_y = np.concatenate(coord_y_blocks)

    order = rng.permutation(len(target))
    return (
        features[order], shap_values[order], target[order],
        truth[order], coord_x[order], coord_y[order],
    )


# ---------------------------------------------------------------------------
# 3. Fit SHAP-Compass on the synthetic dataset
# ---------------------------------------------------------------------------
features, shap_values, target, truth, coord_x, coord_y = make_taiwan_synthetic(
    n_total=2375, random_state=42,
)

print(f"[Taiwan synthetic] n_samples = {len(target)}, target mean = {target.mean():.3f}")

compass = SHAPCompass(
    features=features,
    attributions=shap_values,
    feature_names=FEATURE_NAMES,
    target=target,
)
results = compass.fit(som_grid=(9, 9), n_regimes=6, random_state=42)
results.summary()


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
OUT = Path(__file__).parent / "taiwan_synthetic" / "output"
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
    regime_prefix="TG",
    save_path=IMG / "som_grid.png",
)
plt.close("all")

plot_ward_dendrogram(
    results.neuron_cossin, results.neuron_labels, n_groups,
    regime_prefix="TG",
    save_path=IMG / "ward_dendrogram.png",
)
plt.close("all")

plot_dci_ranking(results.dci, save_path=IMG / "dci_ranking.png")
plt.close("all")

plot_bilayer_heatmap(
    ZF_g, ZS_g, FEATURE_NAMES, n_groups,
    feature_dimensions=FEATURE_DIMENSIONS,
    feature_codes=FEATURE_CODES,
    regime_prefix="TG",
    save_path=IMG / "bilayer_feature_heatmap.png",
)
plt.close("all")

plot_per_feature_unit_circle(
    results.group_theta, results.dci, FEATURE_NAMES, n_groups,
    regime_prefix="TG",
    save_path=IMG / "per_feature_unit_circle.png",
)
plt.close("all")

plot_spatial(
    results.labels, coord_x, coord_y, n_groups, target=target,
    convert_to_wgs84=False,
    regime_prefix="TG",
    target_label="Synthetic NO3-N (mg/L)",
    save_path=IMG / "spatial_distribution.png",
)
plt.close("all")

print(f"[Taiwan synthetic] figures saved to {IMG}")


# ---------------------------------------------------------------------------
# 5. Quality metrics + Stage 2 + summary CSVs
# ---------------------------------------------------------------------------
metrics = compute_all_metrics(
    results, target=target,
    features_raw=features, attributions_raw=shap_values,
)
print(f"[Taiwan synthetic] {len(metrics)} quality metrics computed")
print(f"  eta^2 = {results.eta_sq:.3f}")

s2 = intensity_stratify_from_results(
    results, target=target,
    features_raw=features, feature_names=FEATURE_NAMES,
)
print(f"[Taiwan synthetic] Stage 2: {len(s2.split_groups)} split / {len(s2.retained_groups)} retained")

results.dci.to_csv(OUT / "dci_ranking.csv", index=False)
pd.DataFrame([{"metric": k, "value": v} for k, v in sorted(metrics.items())]).to_csv(
    OUT / "quality_metrics.csv", index=False,
)
pd.DataFrame({
    "synthetic_truth_regime": truth,
    "recovered_regime": results.labels,
    "target": target,
}).to_csv(OUT / "regime_assignments.csv", index=False)
s2.summary.to_csv(OUT / "stage2_verdict.csv", index=False)

print(f"[Taiwan synthetic] outputs saved to {OUT}")
