# -*- coding: utf-8 -*-
"""End-to-end SHAP-Compass demo on the full CONUS groundwater dataset.

Loads the bundled 12,082-well CONUS nitrate release, trains a
gradient-boosting regressor, computes SHAP attributions, runs
SHAP-Compass with the dimension-grouped feature schema, and writes the
full set of gallery figures (PNG) + summary tables (CSV) to
``examples/conus_output/``.

This is the script used to regenerate ``docs/figures/example_*.png``.

Run from the repository root:

    pip install -e ".[shap]"
    python examples/02_conus_full_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from shap_compass import (
    SHAPCompass,
    compute_all_metrics,
    load_conus_nitrate,
    CONUS_FEATURE_DIMENSIONS,
    CONUS_FEATURE_NAMES,
)
from shap_compass.plotting import (
    plot_bilayer_heatmap,
    plot_dci_ranking,
    plot_per_feature_unit_circle,
    plot_som_grid,
    plot_spatial,
)


OUT = Path(__file__).parent / "conus_output"
IMG = OUT / "figures"


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("  SHAP-Compass — CONUS groundwater nitrate (full pipeline)")
    print("=" * 64)

    df = load_conus_nitrate()
    print(f"  Samples: {len(df)}, Features: {len(CONUS_FEATURE_NAMES)}")
    X = df[CONUS_FEATURE_NAMES].to_numpy(dtype=float)
    y = df["NO3"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)

    # ---------------------------------------------------------------
    # 1) Train a regressor on log(NO3) and compute SHAP attributions.
    # ---------------------------------------------------------------
    print("\n[1] Training GradientBoostingRegressor on log(NO3) ...")
    log_y = np.log1p(y)
    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42,
    )
    model.fit(X, log_y)
    train_r2 = model.score(X, log_y)
    print(f"    train R^2 (log scale): {train_r2:.3f}")

    try:
        import shap
    except ImportError as exc:
        raise SystemExit(
            "shap is not installed. Run `pip install -e \".[shap]\"`."
        ) from exc

    print("[2] Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ---------------------------------------------------------------
    # 3) Run SHAP-Compass.
    # ---------------------------------------------------------------
    print("[3] Running SHAP-Compass (SOM 20x20, k=7) ...")
    compass = SHAPCompass(
        features=X,
        attributions=shap_values,
        feature_names=CONUS_FEATURE_NAMES,
        target=y,
    )
    results = compass.fit(som_grid=(20, 20), n_regimes=7, random_state=42)
    results.summary(regime_prefix="UG")

    # ---------------------------------------------------------------
    # 4) Gallery figures.
    # ---------------------------------------------------------------
    n_groups = results.n_groups
    ZF_g = np.zeros((n_groups, len(CONUS_FEATURE_NAMES)))
    ZS_g = np.zeros((n_groups, len(CONUS_FEATURE_NAMES)))
    for g in range(1, n_groups + 1):
        mask = results.labels == g
        if mask.any():
            ZF_g[g - 1] = results.ZF[mask].mean(axis=0)
            ZS_g[g - 1] = results.ZS[mask].mean(axis=0)

    neuron_t = np.zeros(len(results.neuron_ids))
    for ni in range(len(results.neuron_ids)):
        m = results._sample_to_neuron == ni
        if m.any():
            neuron_t[ni] = y[m].mean()

    print("\n[4] Writing gallery figures to", IMG, "...")

    plot_som_grid(
        results.neuron_ids, results.neuron_labels, results.neuron_sizes,
        results.som_grid, n_groups,
        neuron_target_means=neuron_t,
        target_label="NO3 mean (mg/L)",
        regime_prefix="UG",
        figsize=(26, 8),
        save_path=IMG / "som_grid.png",
    )
    plt.close("all"); print("  som_grid.png")

    plot_dci_ranking(
        results.dci, save_path=IMG / "dci_ranking.png", figsize=(9, 8),
    )
    plt.close("all"); print("  dci_ranking.png")

    plot_bilayer_heatmap(
        ZF_g, ZS_g, CONUS_FEATURE_NAMES, n_groups,
        feature_dimensions=CONUS_FEATURE_DIMENSIONS,
        regime_prefix="UG",
        save_path=IMG / "bilayer_feature_heatmap.png",
    )
    plt.close("all"); print("  bilayer_feature_heatmap.png")

    plot_per_feature_unit_circle(
        results.group_theta, results.dci, CONUS_FEATURE_NAMES, n_groups,
        regime_prefix="UG",
        save_path=IMG / "per_feature_unit_circle.png",
    )
    plt.close("all"); print("  per_feature_unit_circle.png")

    plot_spatial(
        results.labels, lon, lat, n_groups, target=y,
        convert_to_wgs84=False,
        regime_prefix="UG",
        target_label="NO3 (mg/L)",
        save_path=IMG / "spatial_distribution.png",
    )
    plt.close("all"); print("  spatial_distribution.png")

    # ---------------------------------------------------------------
    # 5) Tables.
    # ---------------------------------------------------------------
    metrics = compute_all_metrics(
        results, target=y, features_raw=X, attributions_raw=shap_values,
    )
    results.dci.to_csv(OUT / "dci_ranking.csv", index=False)
    pd.DataFrame(
        [{"metric": k, "value": v} for k, v in sorted(metrics.items())]
    ).to_csv(OUT / "quality_metrics.csv", index=False)
    pd.DataFrame({
        "lat": lat, "lon": lon, "NO3": y, "regime": results.labels,
    }).to_csv(OUT / "regime_assignments.csv", index=False)

    print(f"\n  All output saved under {OUT}")
    print("=" * 64)


if __name__ == "__main__":
    main()
