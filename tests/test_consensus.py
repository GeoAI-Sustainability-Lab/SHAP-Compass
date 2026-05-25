"""Tests for ``shap_compass.consensus``."""

import numpy as np
import pytest

from shap_compass.consensus import check_consensus, ConsensusReport


@pytest.fixture
def consensus_data():
    np.random.seed(42)
    n_features = 5

    theta_g1 = np.tile([0.5, 1.0, -0.5, 0.2, -1.0], (7, 1))
    theta_g1 += np.random.normal(0, 0.1, (7, n_features))

    theta_g2 = np.tile([2.0, -1.5, 0.8, -0.3, 1.5], (7, 1))
    theta_g2[5:, 0] *= -1
    theta_g2 += np.random.normal(0, 0.05, (7, n_features))

    neuron_theta = np.vstack([theta_g1, theta_g2])
    neuron_r = np.abs(neuron_theta) + 0.5
    neuron_labels = np.array([1] * 7 + [2] * 7)

    group_theta = np.array([theta_g1.mean(axis=0), theta_g2.mean(axis=0)])
    group_r = np.array([neuron_r[:7].mean(axis=0), neuron_r[7:].mean(axis=0)])
    feature_names = ["A", "B", "C", "D", "E"]
    return neuron_theta, neuron_r, group_theta, group_r, neuron_labels, feature_names


def test_check_consensus_returns_report(consensus_data):
    nt, nr, gt, gr, nl, names = consensus_data
    report = check_consensus(nt, nr, gt, gr, nl, n_groups=2, feature_names=names, top_n=3)
    assert isinstance(report, ConsensusReport)
    assert 0 <= report.dcr <= 1
    assert report.n_total > 0
    assert report.quality in ("excellent", "acceptable", "warning")


def test_consensus_dcr_range(consensus_data):
    nt, nr, gt, gr, nl, names = consensus_data
    report = check_consensus(nt, nr, gt, gr, nl, n_groups=2, feature_names=names, top_n=5)
    assert report.dcr >= 0.5


def test_consensus_details_columns(consensus_data):
    nt, nr, gt, gr, nl, names = consensus_data
    report = check_consensus(nt, nr, gt, gr, nl, n_groups=2, feature_names=names, top_n=3)
    expected = {"group", "feature", "status", "sin_consensus", "cos_consensus"}
    assert expected.issubset(set(report.details.columns))


def test_perfect_consensus():
    n_neurons = 10
    theta = np.tile([0.5, 1.0, -0.5], (n_neurons, 1))
    r = np.ones((n_neurons, 3))
    labels = np.ones(n_neurons, dtype=int)
    group_theta = theta[:1]; group_r = r[:1]
    report = check_consensus(theta, r, group_theta, group_r, labels,
                              n_groups=1, feature_names=["A", "B", "C"], top_n=3)
    assert report.dcr == 1.0
    assert report.n_split == 0
