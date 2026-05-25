"""SHAP-Compass: Directional SHAP Attribution Clustering for GeoAI.

A Python toolkit for revealing how model attributions (e.g., SHAP values) vary
across spatial / environmental contexts. SHAP-Compass projects each
(feature value, attribution) pair onto the unit circle, groups samples whose
attribution directionalities are similar using a SOM + Ward two-stage
clustering, and quantifies cross-regime consistency with the Directional
Consistency Index (DCI).

Reference
---------
SHAP-Compass: A Regime Level Interpretability Framework for Revealing
Spatially Heterogeneous Attribution Mechanisms in GeoAI.
ISPRS Journal of Photogrammetry and Remote Sensing, 2026 (under review).
"""

__version__ = "0.2.0"

from .core import SHAPCompass, SHAPCompassResults
from .transform import (
    z_standardize,
    compute_theta_r,
    cossin_project,
    full_transform,
)
from .dci import (
    compute_dci,
    compute_dci_within_group,
    DCI_BANDS,
    DCI_BAND_COLORS,
    dci_band,
)
from .utils import circular_R, circular_R_axial, circular_mean
# --- Advanced features (kept in source but not part of the public API)
# These are not yet validated in the accompanying paper. The modules
# remain in the package so they can be re-enabled later, but they are
# intentionally not re-exported here. To use them anyway, import from
# the sub-module directly, e.g. `from shap_compass.consensus import ...`.
#
# from .consensus import (
#     check_consensus,
#     check_consensus_from_results,
#     ConsensusReport,
# )
# from .stage2 import (
#     intensity_stratify,
#     intensity_stratify_from_results,
#     Stage2Result,
# )
# from .multi import run_multi_target, MultiTargetResults
from .quality import (
    compute_all_metrics,
    METRIC_NAMES,
    METRIC_COLS,
)

__all__ = [
    "__version__",
    # Core
    "SHAPCompass",
    "SHAPCompassResults",
    # Transform
    "z_standardize",
    "compute_theta_r",
    "cossin_project",
    "full_transform",
    # DCI
    "compute_dci",
    "compute_dci_within_group",
    "DCI_BANDS",
    "DCI_BAND_COLORS",
    "dci_band",
    # Circular utilities
    "circular_R",
    "circular_R_axial",
    "circular_mean",
    # --- Advanced features (currently disabled, see note above)
    # "check_consensus",
    # "check_consensus_from_results",
    # "ConsensusReport",
    # "intensity_stratify",
    # "intensity_stratify_from_results",
    # "Stage2Result",
    # "run_multi_target",
    # "MultiTargetResults",
    # Quality metrics
    "compute_all_metrics",
    "METRIC_NAMES",
    "METRIC_COLS",
]
