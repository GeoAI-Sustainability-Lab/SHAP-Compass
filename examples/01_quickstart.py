# -*- coding: utf-8 -*-
"""SHAP-Compass Quickstart — synthetic data demo.

A 3-context synthetic regression illustrating the full SHAP-Compass
pipeline end-to-end on an artificial dataset (no external data required).

This script is intentionally small (~500 samples, 8 features) so that it
finishes in a few seconds. For larger paper-style demonstrations see
``02_taiwan_synthetic.py`` and ``03_conus_synthetic.py``.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shap_compass import (
    SHAPCompass,
    compute_all_metrics,
)
# --- Advanced features (currently disabled in the public API):
# from shap_compass.consensus import check_consensus_from_results
# from shap_compass.stage2 import intensity_stratify_from_results
from shap_compass.plotting import (
    plot_dci_ranking,
    plot_bilayer_heatmap,
    plot_per_feature_unit_circle,
    plot_group_overview,
    plot_theta_heatmap,
    plot_spatial,
    plot_som_grid,
    plot_ward_dendrogram,
)


# ---------------------------------------------------------------------------
# 1. Synthetic data — three distinct environmental "contexts"
# ---------------------------------------------------------------------------
print("=" * 60)
print("  SHAP-Compass Quickstart — Synthetic Data")
print("=" * 60)

np.random.seed(42)
n_features = 8
feature_names = [
    "Temperature", "Elevation", "Rainfall", "Soil_Clay",
    "Landuse_Agri", "Population", "Vegetation", "Slope",
]

n1, n2, n3 = 180, 170, 150

X_a = np.random.randn(n1, n_features); X_a[:, 0] += 2.0; X_a[:, 2] += 1.5
shap_a = np.random.randn(n1, n_features) * 0.3
shap_a[:, 0] += X_a[:, 0] * 0.8
shap_a[:, 2] += X_a[:, 2] * 0.6
y_a = 8 + X_a[:, 0] * 1.2 + X_a[:, 2] * 0.8 + np.random.normal(0, 1, n1)

X_b = np.random.randn(n2, n_features); X_b[:, 4] += 1.5; X_b[:, 5] += 1.0
shap_b = np.random.randn(n2, n_features) * 0.3
shap_b[:, 4] += X_b[:, 4] * 0.7
shap_b[:, 5] -= X_b[:, 5] * 0.5
y_b = 4 + X_b[:, 4] * 0.6 - X_b[:, 5] * 0.4 + np.random.normal(0, 1, n2)

X_c = np.random.randn(n3, n_features); X_c[:, 1] += 2.5; X_c[:, 6] += 2.0
shap_c = np.random.randn(n3, n_features) * 0.3
shap_c[:, 1] -= X_c[:, 1] * 0.6
shap_c[:, 6] -= X_c[:, 6] * 0.4
y_c = 1 + np.random.normal(0, 0.5, n3)

features = np.vstack([X_a, X_b, X_c])
shap_values = np.vstack([shap_a, shap_b, shap_c])
target = np.concatenate([y_a, y_b, y_c])
coords_x = np.concatenate([
    np.random.uniform(0, 40, n1),
    np.random.uniform(30, 70, n2),
    np.random.uniform(60, 100, n3),
])
coords_y = np.concatenate([
    np.random.uniform(50, 100, n1),
    np.random.uniform(20, 70, n2),
    np.random.uniform(0, 50, n3),
])

print(f"  Samples: {len(target)}, Features: {n_features}")
print(f"  Target range: {target.min():.2f} - {target.max():.2f}")


# ---------------------------------------------------------------------------
# 2. Run SHAP-Compass
# ---------------------------------------------------------------------------
print("\n[1] Running SHAP-Compass (SOM 7x7, k=3)...")
compass = SHAPCompass(
    features=features,
    attributions=shap_values,
    feature_names=feature_names,
    target=target,
)
results = compass.fit(som_grid=(7, 7), n_regimes=3, random_state=42)
results.summary()


# ---------------------------------------------------------------------------
# 3. Generate figures
# ---------------------------------------------------------------------------
OUT = Path(__file__).parent / "quickstart_output"
IMG = OUT / "figures"
IMG.mkdir(parents=True, exist_ok=True)

n_groups = results.n_groups
ZF_g = np.zeros((n_groups, n_features))
ZS_g = np.zeros((n_groups, n_features))
for g in range(1, n_groups + 1):
    mask = results.labels == g
    if mask.any():
        ZF_g[g - 1] = results.ZF[mask].mean(axis=0)
        ZS_g[g - 1] = results.ZS[mask].mean(axis=0)

print("\n[2] Generating figures...")

neuron_t = np.zeros(len(results.neuron_ids))
for ni in range(len(results.neuron_ids)):
    m = results._sample_to_neuron == ni
    if m.any():
        neuron_t[ni] = target[m].mean()

plot_som_grid(
    results.neuron_ids, results.neuron_labels, results.neuron_sizes,
    results.som_grid, n_groups,
    neuron_target_means=neuron_t,
    target_label="Target Mean", figsize=(20, 6),
    save_path=IMG / "som_grid.png",
)
plt.close("all"); print("  som_grid.png")

plot_ward_dendrogram(
    results.neuron_cossin, results.neuron_labels, n_groups,
    save_path=IMG / "ward_dendrogram.png",
)
plt.close("all"); print("  ward_dendrogram.png")

plot_dci_ranking(results.dci, save_path=IMG / "dci_ranking.png")
plt.close("all"); print("  dci_ranking.png")

plot_group_overview(
    results.labels, target, feature_names, results.ZF, results.ZS, n_groups,
    save_path=IMG / "group_overview.png",
)
plt.close("all"); print("  group_overview.png")

plot_theta_heatmap(
    results.group_theta, feature_names, n_groups,
    save_path=IMG / "theta_heatmap.png",
)
plt.close("all"); print("  theta_heatmap.png")

plot_bilayer_heatmap(
    ZF_g, ZS_g, feature_names, n_groups,
    save_path=IMG / "bilayer_heatmap.png",
)
plt.close("all"); print("  bilayer_heatmap.png")

plot_per_feature_unit_circle(
    results.group_theta, results.dci, feature_names, n_groups,
    save_path=IMG / "per_feature_unit_circle.png",
)
plt.close("all"); print("  per_feature_unit_circle.png")

plot_spatial(
    results.labels, coords_x, coords_y, n_groups, target=target,
    convert_to_wgs84=False,
    save_path=IMG / "spatial_distribution.png",
)
plt.close("all"); print("  spatial_distribution.png")


# ---------------------------------------------------------------------------
# 4. Quality metrics + CSV outputs
# ---------------------------------------------------------------------------
print("\n[3] Quality assessment...")

metrics = compute_all_metrics(
    results, target=target,
    features_raw=features, attributions_raw=shap_values,
)
print(f"  Quality metrics computed: {len(metrics)}")

# --- Advanced features (currently disabled):
# consensus = check_consensus_from_results(results, top_n=3)
# print(f"  DCR = {consensus.dcr:.1%} ({consensus.quality})")
#
# s2 = intensity_stratify_from_results(
#     results, target=target,
#     features_raw=features, feature_names=feature_names,
# )
# print(f"  Stage 2: {len(s2.split_groups)} split / {len(s2.retained_groups)} retained")

import pandas as pd
results.dci.to_csv(OUT / "dci_ranking.csv", index=False)
pd.DataFrame([{"metric": k, "value": v} for k, v in sorted(metrics.items())]).to_csv(
    OUT / "quality_metrics.csv", index=False,
)
# --- Advanced-feature CSVs (disabled):
# if consensus.details is not None and len(consensus.details) > 0:
#     consensus.details.to_csv(OUT / "directional_consensus.csv", index=False)
# s2.summary.to_csv(OUT / "stage2_verdict.csv", index=False)

print(f"\n  All output saved to: {OUT}")
print("=" * 60)
