"""SHAP-Compass transformation: Z-standardization, theta/r, unit-circle projection.

The SHAP-Compass vector for each sample is constructed as

    v_n = (cos theta_{n,1}, sin theta_{n,1}, ..., cos theta_{n,J}, sin theta_{n,J})

where theta_{n,j} = arctan2(Z^S_{n,j}, Z^F_{n,j}) is the direction of the
(feature value, attribution) pair in the standardized plane and Z^F / Z^S
are obtained by Z-standardizing the feature and attribution matrices
independently per feature.

The signal magnitude r_{n,j} = sqrt((Z^F_{n,j})^2 + (Z^S_{n,j})^2) is
intentionally dropped at the unit-circle projection step so that downstream
clustering is driven by directionality only and is not dominated by a few
extreme samples (see Section 3.1 ablation in the paper).
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def z_standardize(X: np.ndarray, scaler: StandardScaler | None = None):
    """Z-score standardize a matrix column-wise.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Raw values (feature values or SHAP values).
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None a new scaler is fit on ``X``.

    Returns
    -------
    Z : np.ndarray, shape (n_samples, n_features)
        Standardized values.
    scaler : StandardScaler
        Fitted scaler instance (reuse for held-out data).
    """
    if scaler is None:
        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
    else:
        Z = scaler.transform(X)
    return Z, scaler


def compute_theta_r(ZF: np.ndarray, ZS: np.ndarray):
    """Direction angle theta and signal magnitude r from Z^F and Z^S.

    theta_{n,j} = arctan2(Z^S_{n,j}, Z^F_{n,j})
    r_{n,j}    = sqrt((Z^F_{n,j})^2 + (Z^S_{n,j})^2)
    """
    theta = np.arctan2(ZS, ZF)
    r = np.sqrt(ZS ** 2 + ZF ** 2)
    return theta, r


def cossin_project(theta: np.ndarray) -> np.ndarray:
    """Project per-feature direction angles onto the unit circle.

    Returns the SHAP-Compass matrix in 2J dimensions, laid out as
    ``[cos(theta_{,1}), ..., cos(theta_{,J}), sin(theta_{,1}), ..., sin(theta_{,J})]``.

    Magnitude r is deliberately not retained — the projection isolates
    directional information and avoids extreme-sample dominance.
    """
    return np.hstack([np.cos(theta), np.sin(theta)])


def full_transform(features: np.ndarray, attributions: np.ndarray) -> dict:
    """Run the full SHAP-Compass transformation pipeline.

    Returns
    -------
    dict
        Keys: ``ZF``, ``ZS``, ``theta``, ``r``, ``cossin``,
        ``scaler_F``, ``scaler_S``.
    """
    ZF, scaler_F = z_standardize(features)
    ZS, scaler_S = z_standardize(attributions)
    theta, r = compute_theta_r(ZF, ZS)
    cossin = cossin_project(theta)

    return {
        "ZF": ZF,
        "ZS": ZS,
        "theta": theta,
        "r": r,
        "cossin": cossin,
        "scaler_F": scaler_F,
        "scaler_S": scaler_S,
    }
