"""Circular statistics utilities for SHAP-Compass."""

from __future__ import annotations

import numpy as np


def circular_R(angles: np.ndarray) -> float:
    """Mean resultant length of a set of angles (Fisher 1993, eq. 2.9).

    Returns a value in [0, 1] where 1 indicates perfect concentration and
    0 indicates uniform / cancelled directions.
    """
    angles = np.asarray(angles, dtype=float)
    C = float(np.mean(np.cos(angles)))
    S = float(np.mean(np.sin(angles)))
    return float(np.sqrt(C ** 2 + S ** 2))


def circular_R_axial(angles: np.ndarray) -> float:
    """Axial mean resultant length using the 2-theta doubling transform.

    Treats theta and theta + pi as the same axis (Mardia & Jupp 2000, §2.2),
    which is appropriate when the attribution direction's sign carries
    mechanistic meaning rather than absolute pointing direction.
    """
    angles = np.asarray(angles, dtype=float)
    C = float(np.mean(np.cos(2 * angles)))
    S = float(np.mean(np.sin(2 * angles)))
    return float(np.sqrt(C ** 2 + S ** 2))


def circular_mean(angles: np.ndarray) -> float:
    """Circular mean direction (Fisher 1993, eq. 2.7)."""
    angles = np.asarray(angles, dtype=float)
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def angle_diff(a: float, b: float) -> float:
    """Shortest angular difference between two angles, returned in [0, pi]."""
    d = abs(float(a) - float(b)) % (2 * np.pi)
    return float(min(d, 2 * np.pi - d))
