"""Tests for ``shap_compass.stage2``."""

import numpy as np
import pytest

from shap_compass.stage2 import intensity_stratify, Stage2Result


@pytest.fixture
def stratification_data():
    np.random.seed(42)
    r_g1_strong = np.random.uniform(2, 4, (75, 5))
    r_g1_weak = np.random.uniform(0.1, 1, (75, 5))
    target_g1_strong = np.random.normal(15, 2, 75)
    target_g1_weak = np.random.normal(3, 1, 75)
    feat_g1_strong = np.random.randn(75, 5) + 2
    feat_g1_weak = np.random.randn(75, 5) - 2

    r_g2 = np.random.uniform(1, 2, (150, 5))
    target_g2 = np.random.normal(7, 0.5, 150)
    feat_g2 = np.random.randn(150, 5)

    labels = np.array([1] * 150 + [2] * 150)
    r_matrix = np.vstack([r_g1_strong, r_g1_weak, r_g2])
    target = np.concatenate([target_g1_strong, target_g1_weak, target_g2])
    features = np.vstack([feat_g1_strong, feat_g1_weak, feat_g2])
    return labels, r_matrix, target, features, ["F1", "F2", "F3", "F4", "F5"]


def test_intensity_stratify_returns_result(stratification_data):
    labels, r_matrix, target, features, names = stratification_data
    result = intensity_stratify(
        labels, r_matrix, target, features_raw=features, feature_names=names,
    )
    assert isinstance(result, Stage2Result)
    assert result.summary is not None
    assert len(result.summary) == 2


def test_group1_splits(stratification_data):
    labels, r_matrix, target, features, names = stratification_data
    result = intensity_stratify(
        labels, r_matrix, target, features_raw=features, feature_names=names,
        delta_target_threshold=1.0,
    )
    assert 1 in result.split_groups


def test_group2_retains(stratification_data):
    labels, r_matrix, target, features, names = stratification_data
    result = intensity_stratify(
        labels, r_matrix, target, features_raw=features, feature_names=names,
    )
    assert 2 in result.retained_groups


def test_high_threshold_no_split(stratification_data):
    labels, r_matrix, target, features, names = stratification_data
    result = intensity_stratify(
        labels, r_matrix, target, features_raw=features, feature_names=names,
        delta_target_threshold=100,
    )
    assert len(result.split_groups) == 0


def test_summary_has_conditions(stratification_data):
    labels, r_matrix, target, features, names = stratification_data
    result = intensity_stratify(
        labels, r_matrix, target, features_raw=features, feature_names=names,
    )
    for col in ["cond_a", "cond_b", "cond_c", "cond_d", "verdict"]:
        assert col in result.summary.columns
