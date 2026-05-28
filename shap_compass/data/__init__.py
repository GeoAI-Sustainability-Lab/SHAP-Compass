"""Bundled real-world datasets used by the examples.

Currently includes only the public-domain CONUS groundwater nitrate
release (Ransom et al. 2021). See :func:`load_conus_nitrate` for details.
"""

from .conus import (
    load_conus_nitrate,
    CONUS_FEATURE_DIMENSIONS,
    CONUS_FEATURE_NAMES,
)

__all__ = [
    "load_conus_nitrate",
    "CONUS_FEATURE_DIMENSIONS",
    "CONUS_FEATURE_NAMES",
]
