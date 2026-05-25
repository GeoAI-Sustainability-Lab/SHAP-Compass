"""Multi-target SHAP-Compass analysis.

Runs independent SHAP-Compass analyses for several target variables that
share the same feature matrix, then computes cross-target comparisons
(ARI between regime labels, Spearman correlation of DCI rankings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .core import SHAPCompass, SHAPCompassResults
from .quality import compute_all_metrics, METRIC_NAMES


@dataclass
class MultiTargetResults:
    results: Dict[str, SHAPCompassResults] = field(default_factory=dict)
    metrics: Dict[str, dict] = field(default_factory=dict)
    target_names: List[str] = field(default_factory=list)
    ari_matrix: pd.DataFrame = None
    dci_rank_corr: pd.DataFrame = None

    def summary(self) -> None:
        print("=" * 65)
        print("  Multi-Target SHAP-Compass Summary")
        print("=" * 65)
        print(f"  Targets: {len(self.target_names)}")
        for name in self.target_names:
            r = self.results[name]
            print(f"\n  [{name}]")
            print(f"    Regimes: {r.n_groups}, eta^2: {r.eta_sq:.4f}")
            if r.dci is not None:
                top3 = r.dci.head(3)
                feats = ", ".join(
                    f"{row['feature']}({row['DCI']:.2f})"
                    for _, row in top3.iterrows()
                )
                print(f"    DCI top-3: {feats}")

        if self.ari_matrix is not None:
            print("\n  Pairwise ARI between regime labels:")
            print(
                self.ari_matrix.to_string(float_format="%.3f", index=True)
                .replace("\n", "\n  ")
            )
        if self.dci_rank_corr is not None:
            print("\n  Pairwise Spearman rho of DCI rankings:")
            print(
                self.dci_rank_corr.to_string(float_format="%.3f", index=True)
                .replace("\n", "\n  ")
            )
        print("=" * 65)


def run_multi_target(
    features: np.ndarray,
    attributions_dict: Dict[str, np.ndarray],
    targets_dict: Dict[str, np.ndarray],
    feature_names: Optional[List[str]] = None,
    som_grid: tuple = (9, 9),
    n_regimes: int = 6,
    *,
    som_sigma: float = 1.5,
    som_lr: float = 0.5,
    som_iterations: int = 10000,
    random_state: int = 42,
    output_dir: Optional[str] = None,
    save_figures: bool = True,
    verbose: bool = True,
) -> MultiTargetResults:
    """Run SHAP-Compass independently for each target variable."""
    target_names = list(attributions_dict.keys())

    if set(target_names) != set(targets_dict.keys()):
        raise ValueError(
            "attributions_dict and targets_dict must have the same keys."
        )

    all_results: Dict[str, SHAPCompassResults] = {}
    all_metrics: Dict[str, dict] = {}

    for name in target_names:
        if verbose:
            print(f"\n{'='*60}\n  SHAP-Compass: {name}\n{'='*60}")

        compass = SHAPCompass(
            features=features,
            attributions=attributions_dict[name],
            feature_names=feature_names,
            target=targets_dict[name],
        )
        results = compass.fit(
            som_grid=som_grid,
            n_regimes=n_regimes,
            som_sigma=som_sigma,
            som_lr=som_lr,
            som_iterations=som_iterations,
            random_state=random_state,
        )
        if verbose:
            results.summary()
        all_results[name] = results

        metrics = compute_all_metrics(
            results,
            target=targets_dict[name],
            features_raw=features,
            attributions_raw=attributions_dict[name],
        )
        all_metrics[name] = metrics

        if output_dir is not None:
            target_dir = Path(output_dir) / name
            target_dir.mkdir(parents=True, exist_ok=True)
            _save_target_results(
                results, targets_dict[name], features, metrics,
                name, target_dir, feature_names, save_figures, verbose,
            )

    if verbose:
        print(f"\n{'='*60}\n  Cross-Comparison ({len(target_names)} targets)\n{'='*60}")

    ari_matrix = _compute_ari_matrix(all_results, target_names)
    dci_rank_corr = _compute_dci_rank_correlation(all_results, target_names)

    multi = MultiTargetResults(
        results=all_results,
        metrics=all_metrics,
        target_names=target_names,
        ari_matrix=ari_matrix,
        dci_rank_corr=dci_rank_corr,
    )

    if verbose:
        multi.summary()

    if output_dir is not None:
        cross_dir = Path(output_dir) / "cross_comparison"
        cross_dir.mkdir(parents=True, exist_ok=True)
        _save_cross_comparison(multi, cross_dir, verbose)

    return multi


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_target_results(
    results, target, features, metrics, name, target_dir,
    feature_names, save_figures, verbose,
) -> None:
    n_groups = results.n_groups
    pd.DataFrame({
        "sample_id": range(len(results.labels)),
        "regime": [f"R{g}" for g in results.labels],
    }).to_csv(target_dir / "regime_assignments.csv", index=False, encoding="utf-8-sig")

    if results.dci is not None:
        results.dci.to_csv(target_dir / "dci_ranking.csv", index=False, encoding="utf-8-sig")

    if feature_names is None:
        feature_names = results.feature_names
    pd.DataFrame(
        np.degrees(results.group_theta),
        columns=feature_names,
        index=[f"R{g}" for g in range(1, n_groups + 1)],
    ).to_csv(target_dir / "regime_theta_degrees.csv", encoding="utf-8-sig")

    pd.DataFrame([
        {"metric": k, "name": METRIC_NAMES.get(k, k), "value": v}
        for k, v in sorted(metrics.items())
    ]).to_csv(target_dir / "quality_metrics.csv", index=False, encoding="utf-8-sig")

    with open(target_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"SHAP-Compass Analysis: {name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Samples: {len(results.labels)}\n")
        f.write(f"Features: {len(results.feature_names)}\n")
        f.write(f"Regimes: {n_groups}\n")
        f.write(f"eta^2: {results.eta_sq:.4f}\n\n")
        f.write("Regime sizes:\n")
        for g in range(1, n_groups + 1):
            n = (results.labels == g).sum()
            mean_t = target[results.labels == g].mean()
            f.write(
                f"  R{g}: {n:>5} ({n/len(results.labels)*100:.1f}%), "
                f"target_mean={mean_t:.4f}\n"
            )
        f.write("\nDCI ranking:\n")
        if results.dci is not None:
            for _, row in results.dci.iterrows():
                f.write(
                    f"  {row['rank']:>2}. {row['feature']:<25} "
                    f"DCI={row['DCI']:.3f} ({row['band']})\n"
                )

    if verbose:
        print(f"  [{name}] saved to {target_dir}")

    if save_figures:
        fig_dir = target_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        _generate_target_figures(results, target, fig_dir, n_groups, feature_names)
        if verbose:
            print(f"  [{name}] figures saved to {fig_dir}")


def _generate_target_figures(results, target, fig_dir, n_groups, feature_names) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import (
        plot_dci_ranking,
        plot_bilayer_heatmap,
        plot_theta_heatmap,
        plot_per_feature_unit_circle,
    )

    n_features = len(feature_names)
    ZF_g = np.zeros((n_groups, n_features))
    ZS_g = np.zeros((n_groups, n_features))
    for g in range(1, n_groups + 1):
        mask = results.labels == g
        if mask.any():
            ZF_g[g - 1] = results.ZF[mask].mean(axis=0)
            ZS_g[g - 1] = results.ZS[mask].mean(axis=0)

    try:
        plot_dci_ranking(results.dci, save_path=fig_dir / "dci_ranking.png")
        plt.close("all")
    except Exception:
        pass

    try:
        plot_bilayer_heatmap(
            ZF_g, ZS_g, feature_names, n_groups,
            save_path=fig_dir / "bilayer_heatmap.png",
        )
        plt.close("all")
    except Exception:
        pass

    try:
        plot_theta_heatmap(
            results.group_theta, feature_names, n_groups,
            save_path=fig_dir / "theta_heatmap.png",
        )
        plt.close("all")
    except Exception:
        pass

    try:
        plot_per_feature_unit_circle(
            results.group_theta, results.dci, feature_names, n_groups,
            save_path=fig_dir / "per_feature_unit_circle.png",
        )
        plt.close("all")
    except Exception:
        pass


def _compute_ari_matrix(all_results, target_names) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score
    n = len(target_names)
    ari = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            val = adjusted_rand_score(
                all_results[target_names[i]].labels,
                all_results[target_names[j]].labels,
            )
            ari[i, j] = val
            ari[j, i] = val
    return pd.DataFrame(ari, index=target_names, columns=target_names)


def _compute_dci_rank_correlation(all_results, target_names) -> pd.DataFrame:
    from scipy.stats import spearmanr
    n = len(target_names)
    rho = np.eye(n)
    vectors = {}
    for name in target_names:
        dci_df = all_results[name].dci
        if dci_df is not None:
            vectors[name] = dci_df.set_index("feature")["DCI"]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = target_names[i], target_names[j]
            if a in vectors and b in vectors:
                common = vectors[a].index.intersection(vectors[b].index)
                if len(common) >= 3:
                    r, _ = spearmanr(vectors[a].loc[common], vectors[b].loc[common])
                    rho[i, j] = float(r)
                    rho[j, i] = float(r)
    return pd.DataFrame(rho, index=target_names, columns=target_names)


def _save_cross_comparison(multi, cross_dir, verbose) -> None:
    if multi.ari_matrix is not None:
        multi.ari_matrix.to_csv(cross_dir / "ari_matrix.csv", encoding="utf-8-sig")
    if multi.dci_rank_corr is not None:
        multi.dci_rank_corr.to_csv(
            cross_dir / "dci_rank_correlation.csv", encoding="utf-8-sig"
        )

    table = {
        name: multi.results[name].dci.set_index("feature")["DCI"]
        for name in multi.target_names
        if multi.results[name].dci is not None
    }
    if table:
        df = pd.DataFrame(table)
        for name in multi.target_names:
            df[f"{name}_rank"] = df[name].rank(ascending=False).astype(int)
        df.to_csv(cross_dir / "dci_comparison_table.csv", encoding="utf-8-sig")

    with open(cross_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Multi-Target SHAP-Compass Cross-Comparison\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Targets: {', '.join(multi.target_names)}\n\n")
        f.write("eta^2 (target separability):\n")
        for name in multi.target_names:
            f.write(f"  {name:<20} eta^2 = {multi.results[name].eta_sq:.4f}\n")
        if multi.ari_matrix is not None:
            f.write("\nPairwise ARI:\n")
            for i, ni in enumerate(multi.target_names):
                for j, nj in enumerate(multi.target_names):
                    if j > i:
                        f.write(
                            f"  {ni} vs {nj}: ARI = "
                            f"{multi.ari_matrix.values[i, j]:.4f}\n"
                        )
        if multi.dci_rank_corr is not None:
            f.write("\nDCI rank correlation (Spearman):\n")
            for i, ni in enumerate(multi.target_names):
                for j, nj in enumerate(multi.target_names):
                    if j > i:
                        f.write(
                            f"  {ni} vs {nj}: rho = "
                            f"{multi.dci_rank_corr.values[i, j]:.4f}\n"
                        )

    if verbose:
        print(f"  Cross-comparison saved to {cross_dir}")
