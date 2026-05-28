# -*- coding: utf-8 -*-
"""SHAP-Compass quickstart on the bundled CONUS groundwater dataset.

Three steps:
    1) Train a regressor on (features, target).
    2) Compute SHAP attributions for every sample.
    3) Run SHAP-Compass and print a one-screen summary.

Run from the repository root:

    pip install -e ".[shap]"
    python examples/01_quickstart.py

The script keeps the demo small (sample_size=2000) so it finishes in
under a minute. See examples/02_conus_full_pipeline.py for the full
12,082-sample analysis plus figure generation.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from shap_compass import (
    SHAPCompass,
    load_conus_nitrate,
)


def main() -> None:
    print("=" * 60)
    print("  SHAP-Compass quickstart — CONUS groundwater nitrate")
    print("=" * 60)

    df = load_conus_nitrate(sample_size=2000, random_state=42)
    feature_names = [c for c in df.columns if c not in {"lat", "lon", "NO3"}]
    X = df[feature_names].to_numpy(dtype=float)
    y = df["NO3"].to_numpy(dtype=float)

    print(f"  Samples: {len(y)}, Features: {len(feature_names)}")
    print(f"  Target NO3 range: {y.min():.2f} – {y.max():.2f} mg/L")

    # 1) Fit a regressor on log(NO3) — typical for skewed positive targets.
    print("\n[1] Training GradientBoostingRegressor on log(NO3) ...")
    log_y = np.log1p(y)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42,
    )
    model.fit(X, log_y)

    # 2) Compute SHAP attributions.
    try:
        import shap
    except ImportError as exc:
        raise SystemExit(
            "shap is not installed. Run `pip install -e \".[shap]\"` or "
            "`pip install shap` to enable this example."
        ) from exc

    print("[2] Computing SHAP values with TreeExplainer ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 3) Run SHAP-Compass.
    print("[3] Running SHAP-Compass (SOM 9x9, k=5) ...")
    compass = SHAPCompass(
        features=X,
        attributions=shap_values,
        feature_names=feature_names,
        target=y,
    )
    results = compass.fit(som_grid=(9, 9), n_regimes=5, random_state=42)
    results.summary()


if __name__ == "__main__":
    main()
