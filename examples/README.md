# SHAP-Compass examples

Two scripts, both running on the **public-domain CONUS groundwater
nitrate dataset** that ships with the package
(`shap_compass.load_conus_nitrate`). No external download is required.

| Script | Samples | Features | SOM | k | What it shows |
|---|---|---|---|---|---|
| `01_quickstart.py` | 2,000 | 29 | 9 × 9 | 5 | Smallest end-to-end demo. Trains a regressor, computes SHAP, runs SHAP-Compass, prints a one-screen summary. Finishes in under a minute. |
| `02_conus_full_pipeline.py` | 12,082 | 29 | 12 × 12 | 6 | Full pipeline with all gallery figures and CSV summaries written under `examples/conus_output/`. |

## Running

```bash
pip install -e ".[shap]"
python examples/01_quickstart.py
python examples/02_conus_full_pipeline.py
```

The `shap` extra is needed to compute attributions inside the example
scripts; SHAP-Compass itself does not depend on the `shap` library.

## Data source

The bundled dataset is a public-domain U.S. Geological Survey
groundwater nitrate release:

> Ransom, K.M., Nolan, B.T., Stackelberg, P.E., Belitz, K., Fram, M.S.
> (2021). *Machine learning predictions of nitrate in groundwater used
> for drinking supply in the conterminous United States: data release*.
> U.S. Geological Survey. <https://doi.org/10.5066/P9PQ622D>

The bundled subset keeps 29 of the original features (spanning
nitrogen inputs, land use, soil, drainage, hydrology, climate, well
construction and population), well coordinates, and the target `NO3`
in mg/L. The full release is available from USGS ScienceBase.

## Bringing your own data

Replace the `load_conus_nitrate()` call with your own `(X, y)` pair,
plus an attribution matrix from any explainer:

```python
from shap_compass import SHAPCompass

results = SHAPCompass(
    features=X,                # (n_samples, n_features)
    attributions=shap_values,  # (n_samples, n_features)
    feature_names=feature_names,
    target=y,
).fit(som_grid=(9, 9), n_regimes=6, random_state=42)
```

Any attribution method works — SHAP TreeExplainer / LinearExplainer /
KernelExplainer, LIME, Integrated Gradients, etc. — as long as the
output matrix has the same shape as `features`.
