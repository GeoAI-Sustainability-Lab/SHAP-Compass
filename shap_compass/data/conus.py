"""CONUS groundwater nitrate dataset (Ransom et al. 2021, public domain).

The bundled CSV (``conus_nitrate.csv.gz``) is a 29-feature subset of the
USGS data release that ships with the package so the examples can run
without any external download.

Data source
-----------
Ransom, K.M., Nolan, B.T., Stackelberg, P.E., Belitz, K., Fram, M.S.,
2021. *Machine learning predictions of nitrate in groundwater used for
drinking supply in the conterminous United States: data release*.
U.S. Geological Survey. https://doi.org/10.5066/P9PQ622D

U.S. Geological Survey datasets are released as a work of the United
States Government and are in the public domain. The bundled subset
keeps the original target (``NO3`` in mg/L), well coordinates, and a
curated 29-feature subset spanning nitrogen inputs, land use, soil,
drainage, hydrology, climate, well construction and population.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

_DATA_PATH = Path(__file__).resolve().parent / "conus_nitrate.csv.gz"


# Functional dimensions (helpful for the bilayer-heatmap column grouping).
CONUS_FEATURE_DIMENSIONS: Mapping[str, list[str]] = {
    "Nitrogen input": [
        "NO3_Deposition_1985",
        "NO3_Deposition_1992",
        "NH4_Deposition_1992",
        "Farm_Count_1974",
    ],
    "Natural cover": [
        "NaturalCover_1974",
        "NaturalCover_1982",
        "NaturalCover_1992",
    ],
    "Soil": [
        "Soil_NO4",
        "Soil_BulkDensity",
        "Soil_OrganicMatter",
        "Soil_AWC",
        "AWS_25cm",
    ],
    "Drainage": [
        "Poor_Drainage_Pct",
        "Well_Drained_Pct",
        "Hydrologic_Class",
    ],
    "Hydrology": [
        "Recharge",
        "Runoff",
        "Baseflow_Index",
        "Stream_Density",
        "TWI",
        "Water_Table_Depth",
    ],
    "Climate": [
        "Precip",
        "Temperature",
        "PET",
        "ET",
    ],
    "Well construction": [
        "Well_Depth",
        "Screen_Top",
    ],
    "Other": [
        "Arsenic_LR",
        "Population_Density_1990",
    ],
}

CONUS_FEATURE_NAMES: list[str] = [
    f for feats in CONUS_FEATURE_DIMENSIONS.values() for f in feats
]


def load_conus_nitrate(
    *,
    as_frame: bool = True,
    return_X_y: bool = False,
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """Load the bundled CONUS groundwater nitrate dataset.

    Parameters
    ----------
    as_frame : bool, default True
        When True (default), return a :class:`pandas.DataFrame` with
        columns ``lat``, ``lon``, ``NO3`` and the 29 feature columns.
        When False, return a NumPy structured array of the same data.
    return_X_y : bool, default False
        When True, return ``(X, y)`` where ``X`` is the (n, 29) feature
        matrix and ``y`` is the (n,) target vector. Overrides
        ``as_frame``.
    sample_size : int, optional
        If given, return a uniform random sub-sample of this many rows.
        Useful for quick demos.
    random_state : int, optional
        Seed used when ``sample_size`` is set.

    Returns
    -------
    pandas.DataFrame, numpy.ndarray, or (X, y)
        Depending on the flags above.

    Notes
    -----
    The bundled subset contains 12,082 wells across the conterminous
    United States. Missing feature values were filled with the column
    median; rows with a missing ``NO3`` were dropped.

    See the module docstring for the data citation.
    """
    df = pd.read_csv(_DATA_PATH)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(
            n=sample_size,
            random_state=random_state,
        ).reset_index(drop=True)

    if return_X_y:
        X = df[CONUS_FEATURE_NAMES].to_numpy(dtype=float)
        y = df["NO3"].to_numpy(dtype=float)
        return X, y

    if as_frame:
        return df
    return df.to_records(index=False)
