"""One-shot script that bundles a clean subset of the Ransom 2021 CONUS
groundwater nitrate dataset into the package.

Usage
-----
    python scripts/prepare_conus_data.py path/to/training_and_holdout_data.txt

Or set the environment variable ``CONUS_SOURCE_TXT`` to that path and
run the script without arguments. The output CSV is small enough
(under ~2 MB compressed) to live in the repo.

The selection keeps the 25 most-cited features in the original Ransom
study (covering nitrogen inputs, land use, soil, drainage, hydrology,
climate, well depth, population), together with the target ``NO3``
(mg/L) and the well coordinates ``lat`` / ``lon``.

Data source
-----------
Ransom, K.M., Nolan, B.T., Stackelberg, P.E., Belitz, K., Fram, M.S.,
2021. Machine learning predictions of nitrate in groundwater used for
drinking supply in the conterminous United States: data release.
U.S. Geological Survey, https://doi.org/10.5066/P9PQ622D

The dataset is in the public domain (U.S. Geological Survey
information policy). Download ``training_and_holdout_data.txt`` from
the ScienceBase release linked above and pass its path to this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DST = Path(__file__).resolve().parents[1] / "shap_compass" / "data" / "conus_nitrate.csv.gz"


def _resolve_source() -> Path:
    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).expanduser().resolve()
    env = os.environ.get("CONUS_SOURCE_TXT")
    if env:
        return Path(env).expanduser().resolve()
    raise SystemExit(
        "No source file given. Pass the path to "
        "training_and_holdout_data.txt as the first argument, or set "
        "the CONUS_SOURCE_TXT environment variable."
    )


# Curated 25-feature subset (top SHAP importance from the Ransom analysis),
# spanning nitrogen inputs, land use, soil, drainage, hydrology, climate
# and well construction. Original column → readable short name.
FEATURE_RENAME = {
    "DEPTH":              "Well_Depth",
    "TOP":                "Screen_Top",
    "us_ppt1981_mmyr":    "Precip",
    "us_tave198_degC":    "Temperature",
    "PET_mmyr":           "PET",
    "ET_Reitz":           "ET",
    "rech48":             "Recharge",
    "runoff_Reitz":       "Runoff",
    "BFI48":              "Baseflow_Index",
    "StreamDensity":      "Stream_Density",
    "TWI":                "TWI",
    "WTDEPL_m":           "Water_Table_Depth",
    "dep_no3_1985":       "NO3_Deposition_1985",
    "dep_no3_1992":       "NO3_Deposition_1992",
    "dep_nh4_1992":       "NH4_Deposition_1992",
    "nfarm_1974":         "Farm_Count_1974",
    "X1974_LU50":         "NaturalCover_1974",
    "X1982_LU50":         "NaturalCover_1982",
    "X1992_LU50":         "NaturalCover_1992",
    "AVG_NO4_mean":       "Soil_NO4",
    "avg_bd_mean":        "Soil_BulkDensity",
    "avg_om_mean":        "Soil_OrganicMatter",
    "avg_awc_mean":       "Soil_AWC",
    "AWS25_mean":         "AWS_25cm",
    "DrnClass_9_mean":    "Poor_Drainage_Pct",
    "DrnClass_3_mean":    "Well_Drained_Pct",
    "HYDCLASS_mean":      "Hydrologic_Class",
    "LR_arsenic":         "Arsenic_LR",
    "pden_1990":          "Population_Density_1990",
}


def main() -> None:
    src = _resolve_source()
    print(f"reading {src}")
    df = pd.read_csv(src, sep="\t", low_memory=False)
    print(f"  raw shape: {df.shape}")

    src_cols = [c for c in FEATURE_RENAME if c in df.columns]
    missing = [c for c in FEATURE_RENAME if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns in source: {missing}")

    out = pd.DataFrame({
        "lat": df["DEC_LAT_VA"].astype(float),
        "lon": df["DEC_LONG_VA"].astype(float),
        "NO3": df["NO3"].astype(float),
    })
    for src, dst in FEATURE_RENAME.items():
        out[dst] = df[src].astype(float)

    # Fill missing feature values with column median (drop only rows where
    # target NO3 is missing).
    out = out.dropna(subset=["NO3"]).reset_index(drop=True)
    feature_cols = list(FEATURE_RENAME.values())
    out[feature_cols] = out[feature_cols].fillna(out[feature_cols].median())

    # Clip the target to a sensible range and drop any non-CONUS rows.
    out = out[(out["NO3"] >= 0) & (out["NO3"] <= 100)]
    out = out[(out["lat"] > 24) & (out["lat"] < 50)]
    out = out[(out["lon"] > -125) & (out["lon"] < -66)]
    out = out.reset_index(drop=True)

    print(f"  output shape: {out.shape}")
    print(f"  NO3 median: {out['NO3'].median():.3f} mg/L")

    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False, compression={"method": "gzip", "compresslevel": 9})
    size_mb = DST.stat().st_size / 1024 / 1024
    print(f"wrote {DST} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
