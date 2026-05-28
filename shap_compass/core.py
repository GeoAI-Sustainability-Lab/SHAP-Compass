"""SHAPCompass: main entry point for the SHAP-Compass analysis framework.

Pipeline (matches Section 2 of the ISPRS paper):

    1. Joint Z-standardisation of features (Z^F) and SHAP attributions (Z^S)
       column by column.
    2. Per-sample unit-circle projection
           theta_{n,j} = arctan2(Z^S_{n,j}, Z^F_{n,j})
           v_n         = (cos theta_{n,1}, sin theta_{n,1}, ...,
                          cos theta_{n,J}, sin theta_{n,J})
       producing the N x 2J SHAP-Compass matrix.
    3. Two-stage clustering (Vesanto & Alhoniemi 2000):
       3a. A Self-Organising Map of size M x M is trained on the
           SHAP-Compass matrix.
       3b. Each active neuron aggregates Z^F / Z^S of its mapped samples and
           yields a 2J-dimensional directional fingerprint.
       3c. Ward hierarchical clustering on the fingerprint matrix partitions
           neurons into K attribution regimes; samples inherit the label
           of their best-matching neuron.
    4. Regime-level direction angles -> Directional Consistency Index (DCI)
       per feature (see ``dci.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .transform import full_transform
from .som import train_som, get_bmu_assignments, neuron_aggregate
from .clustering import ward_cluster, remap_by_target, eta_squared
from .dci import compute_dci


@dataclass
class SHAPCompassResults:
    """Container for the result of a single SHAP-Compass run."""

    # Sample-level
    labels: Optional[np.ndarray] = None
    theta: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    cossin: Optional[np.ndarray] = None
    ZF: Optional[np.ndarray] = None
    ZS: Optional[np.ndarray] = None

    # Neuron-level
    neuron_theta: Optional[np.ndarray] = None
    neuron_r: Optional[np.ndarray] = None
    neuron_cossin: Optional[np.ndarray] = None
    neuron_labels: Optional[np.ndarray] = None
    neuron_sizes: Optional[np.ndarray] = None
    neuron_ids: list = field(default_factory=list)
    som_grid: tuple = (9, 9)
    _som: Optional[object] = None
    _sample_to_neuron: Optional[np.ndarray] = None

    # Regime-level
    group_theta: Optional[np.ndarray] = None
    n_groups: int = 0
    feature_names: list = field(default_factory=list)

    # Quality
    dci: Optional[pd.DataFrame] = None
    eta_sq: float = 0.0

    # ----------------------------------------------------------------
    # Convenience views
    # ----------------------------------------------------------------
    @property
    def regime_labels(self) -> np.ndarray:
        """Alias for ``labels`` using the paper's regime terminology."""
        return self.labels

    @property
    def n_regimes(self) -> int:
        return int(self.n_groups)

    def summary(self) -> None:
        """Print a concise text summary."""
        n_samples = len(self.labels) if self.labels is not None else 0
        n_features = len(self.feature_names)

        print("=" * 60)
        print("  SHAP-Compass Analysis Results")
        print("=" * 60)
        print(f"  Samples:    {n_samples}")
        print(f"  Features:   {n_features}")
        print(f"  Regimes:    {self.n_groups}")
        print(f"  eta^2 (target): {self.eta_sq:.4f}")
        print()

        if self.labels is not None:
            print("  Regime sizes:")
            for g in range(1, self.n_groups + 1):
                n = int((self.labels == g).sum())
                pct = n / n_samples * 100 if n_samples else 0.0
                print(f"    R{g}: {n:>5} ({pct:.1f}%)")

        if self.dci is not None:
            print()
            print("  DCI ranking (top 5):")
            for _, row in self.dci.head(5).iterrows():
                print(
                    f"    {row['rank']}. {row['feature']:<20} "
                    f"DCI={row['DCI']:.3f} ({row['band']})"
                )

        print("=" * 60)


class SHAPCompass:
    """SHAP-Compass: directional SHAP attribution clustering.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Raw feature values (any scale; the constructor will standardise).
    attributions : np.ndarray, shape (n_samples, n_features)
        Attribution values (e.g. SHAP) aligned with ``features``.
    feature_names : list of str, optional
        Feature labels used in plots and the DCI table.
    target : np.ndarray, shape (n_samples,), optional
        Target variable used to relabel regimes in descending mean order
        (matches the TG1 .. TGK convention of the paper).

    Examples
    --------
    >>> from shap_compass import SHAPCompass
    >>> compass = SHAPCompass(features=X, attributions=shap_values,
    ...                       feature_names=names, target=y)
    >>> results = compass.fit(som_grid=(9, 9), n_regimes=6)
    >>> results.summary()
    """

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------
    # --- Advanced feature (disabled): auto-compute SHAP values from a
    #     trained model. Kept in source for reference; not yet validated
    #     in the accompanying paper, so it is not part of the public API.
    # @classmethod
    # def from_model(
    #     cls,
    #     model,
    #     X: np.ndarray,
    #     feature_names: list | None = None,
    #     target: np.ndarray | None = None,
    #     explainer_type: str = "auto",
    # ) -> "SHAPCompass":
    #     """Build a ``SHAPCompass`` by computing SHAP values automatically.
    #
    #     Tries :class:`shap.TreeExplainer` first, then falls back to
    #     :class:`shap.LinearExplainer` / :class:`shap.KernelExplainer`.
    #     """
    #     try:
    #         import shap
    #     except ImportError as err:  # pragma: no cover - import guard
    #         raise ImportError(
    #             "shap is required for SHAPCompass.from_model(). "
    #             "Install it with: pip install shap"
    #         ) from err
    #
    #     if isinstance(X, pd.DataFrame):
    #         if feature_names is None:
    #             feature_names = list(X.columns)
    #         X_array = X.values.astype(np.float64)
    #     else:
    #         X_array = np.asarray(X, dtype=np.float64)
    #
    #     if explainer_type == "auto":
    #         try:
    #             explainer = shap.TreeExplainer(model)
    #             explainer_used = "TreeExplainer"
    #         except Exception:
    #             try:
    #                 explainer = shap.LinearExplainer(model, X_array)
    #                 explainer_used = "LinearExplainer"
    #             except Exception:
    #                 explainer = shap.KernelExplainer(
    #                     model.predict, shap.sample(X_array, 100)
    #                 )
    #                 explainer_used = "KernelExplainer"
    #     elif explainer_type == "tree":
    #         explainer = shap.TreeExplainer(model)
    #         explainer_used = "TreeExplainer"
    #     elif explainer_type == "linear":
    #         explainer = shap.LinearExplainer(model, X_array)
    #         explainer_used = "LinearExplainer"
    #     elif explainer_type == "kernel":
    #         explainer = shap.KernelExplainer(model.predict, shap.sample(X_array, 100))
    #         explainer_used = "KernelExplainer"
    #     else:
    #         raise ValueError(f"Unknown explainer_type: {explainer_type!r}")
    #
    #     print(f"[SHAP-Compass] Computing SHAP values with {explainer_used} ...")
    #     shap_values = explainer.shap_values(X_array)
    #     if isinstance(shap_values, list):
    #         shap_values = shap_values[0]
    #     shap_values = np.asarray(shap_values, dtype=np.float64)
    #     print(f"[SHAP-Compass] SHAP shape = {shap_values.shape}")
    #
    #     return cls(
    #         features=X_array,
    #         attributions=shap_values,
    #         feature_names=feature_names,
    #         target=target,
    #     )

    def __init__(
        self,
        features: np.ndarray,
        attributions: np.ndarray,
        feature_names: list | None = None,
        target: np.ndarray | None = None,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float64)
        self.attributions = np.asarray(attributions, dtype=np.float64)
        self.target = (
            np.asarray(target, dtype=np.float64) if target is not None else None
        )

        n_samples, n_features = self.features.shape
        if self.attributions.shape != (n_samples, n_features):
            raise ValueError(
                f"features shape {self.features.shape} != "
                f"attributions shape {self.attributions.shape}"
            )

        if feature_names is None:
            self.feature_names = [f"F{i+1}" for i in range(n_features)]
        else:
            self.feature_names = list(feature_names)

        self.results_: Optional[SHAPCompassResults] = None

    # ----------------------------------------------------------------
    # Pipeline
    # ----------------------------------------------------------------
    def fit(
        self,
        som_grid: tuple = (9, 9),
        n_regimes: int = 6,
        *,
        use_som: bool = True,
        som_sigma: float = 1.5,
        som_lr: float = 0.5,
        som_iterations: int = 10000,
        random_state: int = 42,
        pretrained_som=None,
        n_clusters: int | None = None,
    ) -> SHAPCompassResults:
        """Run the full SHAP-Compass pipeline.

        Parameters
        ----------
        som_grid : (rows, cols)
            SOM grid size. The paper uses ``(9, 9)`` for Taiwan (J=17)
            and ``(20, 20)`` for CONUS (J=76).
        n_regimes : int
            Number of attribution regimes (Ward's k). Default ``6``.
            ``n_clusters`` is accepted as an alias.
        use_som : bool
            If ``False``, Ward is applied directly to the sample-level
            SHAP-Compass matrix without the SOM step (only recommended
            for small datasets — see Section 2.3 of the paper).
        som_sigma, som_lr, som_iterations : MiniSom hyperparameters.
        random_state : int
            Seed propagated to MiniSom for reproducibility.
        pretrained_som : MiniSom, optional
            A previously trained SOM to reuse. ``som_grid`` is then
            inferred from its weights.
        """
        if n_clusters is not None:
            n_regimes = int(n_clusters)

        # Step 1: transformation
        tx = full_transform(self.features, self.attributions)

        results = SHAPCompassResults(
            theta=tx["theta"],
            r=tx["r"],
            cossin=tx["cossin"],
            ZF=tx["ZF"],
            ZS=tx["ZS"],
            feature_names=self.feature_names,
            n_groups=n_regimes,
        )

        if use_som:
            # Step 2a: SOM is trained on the SHAP-Compass matrix
            som_data = tx["cossin"]

            if pretrained_som is not None:
                som = pretrained_som
                w = som.get_weights()
                som_grid = (w.shape[0], w.shape[1])
            else:
                som = train_som(
                    som_data,
                    grid_size=som_grid,
                    sigma=som_sigma,
                    learning_rate=som_lr,
                    num_iteration=som_iterations,
                    random_seed=random_state,
                )
            bmu = get_bmu_assignments(som, som_data)

            # Step 2b: neuron aggregation
            agg = neuron_aggregate(
                tx["theta"], tx["r"], tx["ZF"], tx["ZS"], bmu, grid_size=som_grid
            )

            results.neuron_theta = agg["neuron_theta"]
            results.neuron_r = agg["neuron_r"]
            results.neuron_cossin = agg["neuron_cossin"]
            results.neuron_sizes = agg["neuron_sizes"]
            results.neuron_ids = agg["neuron_ids"]
            results.som_grid = som_grid
            results._som = som
            results._sample_to_neuron = agg["sample_to_neuron"]

            # Step 2c: Ward on neuron-level COSSIN fingerprints
            raw_labels = ward_cluster(agg["neuron_cossin"], n_clusters=n_regimes)
            sample_raw = raw_labels[agg["sample_to_neuron"]] + 1

            if self.target is not None:
                sample_labels = remap_by_target(sample_raw, self.target, descending=True)
                neuron_labels = np.array([
                    int(np.median(sample_labels[agg["sample_to_neuron"] == i]))
                    for i in range(len(agg["neuron_ids"]))
                ])
            else:
                sample_labels = sample_raw
                neuron_labels = raw_labels + 1

            results.neuron_labels = neuron_labels
            results.labels = sample_labels

        else:
            raw_labels = ward_cluster(tx["cossin"], n_clusters=n_regimes)
            if self.target is not None:
                results.labels = remap_by_target(raw_labels + 1, self.target, descending=True)
            else:
                results.labels = raw_labels + 1

        # Step 3: regime-level theta from sample-level Z^F / Z^S
        group_theta = np.zeros((n_regimes, tx["theta"].shape[1]))
        for g in range(1, n_regimes + 1):
            mask = results.labels == g
            if mask.any():
                mean_ZF = tx["ZF"][mask].mean(axis=0)
                mean_ZS = tx["ZS"][mask].mean(axis=0)
                group_theta[g - 1] = np.arctan2(mean_ZS, mean_ZF)
        results.group_theta = group_theta

        # Step 4: DCI per feature
        results.dci = compute_dci(group_theta, self.feature_names)

        # Step 5: eta^2 against the target if provided
        if self.target is not None:
            results.eta_sq = eta_squared(results.labels, self.target)

        self.results_ = results
        return results
