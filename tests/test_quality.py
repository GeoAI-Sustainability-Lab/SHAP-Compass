"""Tests for ``shap_compass.quality``."""

import numpy as np
import pytest

from shap_compass.quality import (
    compute_M01, compute_M02, compute_M05,
    compute_M12, compute_M15, compute_M16,
    compute_M19, compute_M19_min_frac,
    compute_M20, compute_M21,
    METRIC_COLS,
)


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    labels = np.repeat([1, 2, 3], [70, 70, 60])
    target = np.concatenate([
        np.random.normal(2, 0.5, 70),
        np.random.normal(5, 0.5, 70),
        np.random.normal(10, 0.5, 60),
    ])
    features = np.random.randn(200, 5)
    attributions = np.random.randn(200, 5)
    return labels, target, features, attributions


def test_metric_cols_is_21():
    assert len(METRIC_COLS) == 21
    assert METRIC_COLS[0] == "M01"
    assert METRIC_COLS[-1] == "M21"


def test_M01(synthetic_data):
    _, target, _, attributions = synthetic_data
    assert 0 <= compute_M01(attributions, target) <= 1


def test_M02(synthetic_data):
    _, _, features, attributions = synthetic_data
    from shap_compass.transform import z_standardize
    ZF, _ = z_standardize(features)
    ZS, _ = z_standardize(attributions)
    assert 0 <= compute_M02(ZF, ZS) <= 1


def test_M05():
    assert compute_M05(np.array([10, 20, 30, 15, 25]), (3, 3)) == pytest.approx(5 / 9)
    assert compute_M05(np.ones(9), (3, 3)) == pytest.approx(1.0)


def test_M12(synthetic_data):
    labels, target, _, _ = synthetic_data
    assert 0 < compute_M12(labels, target) <= 1


def test_M15(synthetic_data):
    labels, _, _, attributions = synthetic_data
    assert 0 <= compute_M15(attributions, labels, 3) <= 1


def test_M16(synthetic_data):
    labels, target, _, _ = synthetic_data
    assert 0 <= compute_M16(labels, target, 3) <= 1


def test_M19_balanced():
    labels = np.repeat([1, 2, 3], [100, 100, 100])
    assert compute_M19(labels, 3) > 0.9
    bad = np.repeat([1, 2, 3], [200, 95, 5])
    assert compute_M19(bad, 3) == 0.0


def test_M19_min_frac():
    labels = np.repeat([1, 2, 3], [150, 100, 50])
    assert compute_M19_min_frac(labels, 3) == pytest.approx(50 / 300)


def test_M20(synthetic_data):
    labels, target, _, _ = synthetic_data
    assert compute_M20(labels, target, 3) > 0


def test_M21_diverse_clusters():
    rng = np.random.default_rng(42)
    cossin = np.vstack([
        rng.standard_normal((20, 6)) + np.array([1, 0, 0, 0, 1, 0]),
        rng.standard_normal((20, 6)) + np.array([0, 1, 0, 0, 0, 1]),
        rng.standard_normal((20, 6)) + np.array([0, 0, 1, 1, 0, 0]),
    ])
    labels = np.repeat([1, 2, 3], 20)
    val = compute_M21(cossin, labels, 3)
    assert 0 <= val <= 1
    assert val > 0.1


