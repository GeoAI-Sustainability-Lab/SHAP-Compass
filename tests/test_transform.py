"""Tests for ``shap_compass.transform``."""

import numpy as np

from shap_compass.transform import (
    z_standardize,
    compute_theta_r,
    cossin_project,
    full_transform,
)


def test_z_standardize():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    Z, _ = z_standardize(X)
    assert Z.shape == (3, 2)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(Z.std(axis=0, ddof=0), 1, atol=1e-10)


def test_compute_theta_r():
    ZF = np.array([[1, 0], [0, 1], [-1, 0]])
    ZS = np.array([[0, 1], [1, 0], [0, -1]])
    theta, r = compute_theta_r(ZF, ZS)
    assert theta.shape == (3, 2)
    assert r.shape == (3, 2)
    np.testing.assert_allclose(theta[0, 0], 0, atol=1e-10)
    np.testing.assert_allclose(theta[0, 1], np.pi / 2, atol=1e-10)
    np.testing.assert_allclose(r[0, 0], 1.0, atol=1e-10)


def test_cossin_project():
    theta = np.array([[0, np.pi / 2], [np.pi, -np.pi / 2]])
    cossin = cossin_project(theta)
    assert cossin.shape == (2, 4)
    np.testing.assert_allclose(cossin[0, 0], 1.0, atol=1e-10)
    np.testing.assert_allclose(cossin[0, 2], 0.0, atol=1e-10)


def test_full_transform_on_unit_circle():
    np.random.seed(0)
    features = np.random.randn(50, 3)
    shap = np.random.randn(50, 3)
    out = full_transform(features, shap)
    assert out["ZF"].shape == (50, 3)
    assert out["ZS"].shape == (50, 3)
    assert out["theta"].shape == (50, 3)
    assert out["r"].shape == (50, 3)
    assert out["cossin"].shape == (50, 6)
    for j in range(3):
        cos_col = out["cossin"][:, j]
        sin_col = out["cossin"][:, j + 3]
        np.testing.assert_allclose(cos_col ** 2 + sin_col ** 2, 1.0, atol=1e-10)
