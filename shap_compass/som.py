"""SOM training and neuron-level aggregation for SHAP-Compass.

The SOM is trained directly on the SHAP-Compass matrix (2J-dimensional
unit-circle projections), reproducing the two-stage SOM + Ward design of
Vesanto & Alhoniemi (2000). Each active neuron then aggregates the Z^F /
Z^S of its mapped samples; the resulting neuron-level (cos theta, sin theta)
vectors are what Ward clusters in the next stage.
"""

from __future__ import annotations

import numpy as np
from minisom import MiniSom


def train_som(
    data: np.ndarray,
    grid_size: tuple = (9, 9),
    sigma: float = 1.5,
    learning_rate: float = 0.5,
    num_iteration: int = 10000,
    random_seed: int = 42,
) -> MiniSom:
    """Train a MiniSom on the SHAP-Compass matrix.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, input_dim)
        Typically the 2J-dim SHAP-Compass matrix.
    grid_size : tuple of int
        SOM grid (rows, cols). Use ``(9, 9)`` for the Taiwan case study
        and ``(20, 20)`` for the CONUS case study in the paper.
    sigma, learning_rate, num_iteration : MiniSom training hyperparameters.
    random_seed : int
        Seed forwarded to MiniSom. The package uses ``42`` everywhere by
        default to keep results reproducible across runs.
    """
    rows, cols = grid_size
    input_dim = data.shape[1]

    som = MiniSom(
        rows, cols, input_dim,
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=random_seed,
    )
    som.random_weights_init(data)
    som.train_random(data, num_iteration=num_iteration, verbose=False)
    return som


def get_bmu_assignments(som: MiniSom, data: np.ndarray) -> np.ndarray:
    """(row, col) Best Matching Unit for each sample."""
    return np.array([som.winner(x) for x in data])


def neuron_aggregate(
    theta: np.ndarray,
    r: np.ndarray,
    ZF: np.ndarray,
    ZS: np.ndarray,
    bmu_indices: np.ndarray,
    grid_size: tuple = (9, 9),
):
    """Aggregate sample-level Z^F / Z^S to active SOM neurons.

    For each active neuron the mean Z^F and Z^S are taken across its
    mapped samples; the neuron-level theta is then ``arctan2(<Z^S>, <Z^F>)``.
    """
    rows, cols = grid_size
    n_features = theta.shape[1]

    neuron_key = bmu_indices[:, 0] * cols + bmu_indices[:, 1]
    unique_neurons = np.unique(neuron_key)

    neuron_ids: list[tuple[int, int]] = []
    neuron_theta_list = []
    neuron_r_list = []
    neuron_sizes: list[int] = []
    sample_to_neuron = np.full(len(theta), -1, dtype=int)

    for idx, nk in enumerate(unique_neurons):
        r_pos, c_pos = divmod(int(nk), cols)
        mask = neuron_key == nk

        mean_ZF = ZF[mask].mean(axis=0)
        mean_ZS = ZS[mask].mean(axis=0)

        n_theta = np.arctan2(mean_ZS, mean_ZF)
        n_r = np.sqrt(mean_ZS ** 2 + mean_ZF ** 2)

        neuron_ids.append((r_pos, c_pos))
        neuron_theta_list.append(n_theta)
        neuron_r_list.append(n_r)
        neuron_sizes.append(int(mask.sum()))
        sample_to_neuron[mask] = idx

    neuron_theta = np.array(neuron_theta_list)
    neuron_r = np.array(neuron_r_list)
    neuron_cossin = np.hstack([np.cos(neuron_theta), np.sin(neuron_theta)])

    return {
        "neuron_ids": neuron_ids,
        "neuron_theta": neuron_theta,
        "neuron_r": neuron_r,
        "neuron_cossin": neuron_cossin,
        "neuron_sizes": np.array(neuron_sizes),
        "sample_to_neuron": sample_to_neuron,
    }
