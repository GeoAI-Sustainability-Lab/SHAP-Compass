"""Tests for ``shap_compass.multi``."""

import numpy as np
import pytest

from shap_compass.multi import run_multi_target, MultiTargetResults


@pytest.fixture
def synthetic_multi_data():
    np.random.seed(42)
    n, p = 200, 5
    features = np.random.randn(n, p)
    feature_names = ["F1", "F2", "F3", "F4", "F5"]

    target_a = 2 * features[:, 0] + features[:, 1] + np.random.normal(0, 0.5, n)
    shap_a = np.random.randn(n, p)
    shap_a[:, 0] += features[:, 0] * 0.5

    target_b = features[:, 2] - 1.5 * features[:, 3] + np.random.normal(0, 0.5, n)
    shap_b = np.random.randn(n, p)
    shap_b[:, 2] += features[:, 2] * 0.5

    return {
        "features": features,
        "feature_names": feature_names,
        "attributions": {"TargetA": shap_a, "TargetB": shap_b},
        "targets": {"TargetA": target_a, "TargetB": target_b},
    }


def test_run_multi_target_returns_results(synthetic_multi_data):
    d = synthetic_multi_data
    result = run_multi_target(
        features=d["features"],
        attributions_dict=d["attributions"],
        targets_dict=d["targets"],
        feature_names=d["feature_names"],
        som_grid=(5, 5), n_regimes=3,
        som_iterations=500, verbose=False,
    )
    assert isinstance(result, MultiTargetResults)
    assert "TargetA" in result.results
    assert "TargetB" in result.results


def test_ari_matrix_shape(synthetic_multi_data):
    d = synthetic_multi_data
    result = run_multi_target(
        features=d["features"],
        attributions_dict=d["attributions"],
        targets_dict=d["targets"],
        feature_names=d["feature_names"],
        som_grid=(5, 5), n_regimes=3,
        som_iterations=500, verbose=False,
    )
    assert result.ari_matrix.shape == (2, 2)
    np.testing.assert_allclose(np.diag(result.ari_matrix.values), 1.0)


def test_dci_rank_correlation(synthetic_multi_data):
    d = synthetic_multi_data
    result = run_multi_target(
        features=d["features"],
        attributions_dict=d["attributions"],
        targets_dict=d["targets"],
        feature_names=d["feature_names"],
        som_grid=(5, 5), n_regimes=3,
        som_iterations=500, verbose=False,
    )
    assert result.dci_rank_corr is not None
    assert result.dci_rank_corr.shape == (2, 2)


def test_output_dir_structure(synthetic_multi_data, tmp_path):
    d = synthetic_multi_data
    out = tmp_path / "multi_output"
    run_multi_target(
        features=d["features"],
        attributions_dict=d["attributions"],
        targets_dict=d["targets"],
        feature_names=d["feature_names"],
        som_grid=(5, 5), n_regimes=3,
        som_iterations=500,
        output_dir=out, save_figures=False, verbose=False,
    )
    assert (out / "TargetA").is_dir()
    assert (out / "TargetB").is_dir()
    assert (out / "TargetA" / "dci_ranking.csv").exists()
    assert (out / "TargetA" / "quality_metrics.csv").exists()
    assert (out / "cross_comparison" / "ari_matrix.csv").exists()


def test_mismatched_keys_raises():
    features = np.random.randn(50, 3)
    with pytest.raises(ValueError, match="same keys"):
        run_multi_target(
            features=features,
            attributions_dict={"A": np.random.randn(50, 3)},
            targets_dict={"B": np.random.randn(50)},
            som_grid=(3, 3), n_regimes=2,
            verbose=False,
        )
