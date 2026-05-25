"""Stage 2: per-regime intensity stratification.

After SHAP-Compass clustering captures directional regimes (Stage 1), each
regime can optionally be split on the r-vector (signal strength) to surface
"Strong" and "Weak" sub-regimes when warranted. A four-condition decision
rule guards the split:

    (a) target-mean delta > threshold (e.g. 1.0 mg/L for nitrate)
    (b) R-vector difference significant (Mann-Whitney p < 0.01)
    (c) >= ``min_sig_features`` raw features differ (p < 0.05)
    (d) smallest sub-regime >= ``min_subgroup_size`` samples

All four conditions must hold for a "split" verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans


@dataclass
class Stage2Result:
    summary: pd.DataFrame = None
    subgroup_labels: np.ndarray = None
    split_groups: list = field(default_factory=list)
    retained_groups: list = field(default_factory=list)
    feature_comparison: pd.DataFrame = None

    def print_summary(self) -> None:
        n_split = len(self.split_groups)
        n_retain = len(self.retained_groups)
        print("=" * 60)
        print(
            f"  Stage 2 Intensity Stratification: "
            f"{n_split} split / {n_retain} retained"
        )
        print("=" * 60)
        if self.summary is not None:
            for _, r in self.summary.iterrows():
                icon = "SPLIT" if r["verdict"] == "split" else "retain"
                print(f"  R{r['group']} (n={r['n_samples']}): [{icon}]")
                print(f"    Strong n={r['n_strong']}, target={r['target_strong']:.3f}")
                print(f"    Weak   n={r['n_weak']}, target={r['target_weak']:.3f}")
                print(
                    f"    (a) delta={r['delta_target']:.3f} "
                    f"{'PASS' if r['cond_a'] else 'FAIL'}"
                )
                print(
                    f"    (b) p_R={r['p_R']:.2e} "
                    f"{'PASS' if r['cond_b'] else 'FAIL'}"
                )
                print(
                    f"    (c) sig_features={r['n_sig_features']} "
                    f"{'PASS' if r['cond_c'] else 'FAIL'}"
                )
                print(
                    f"    (d) min_n={r['min_subgroup']} "
                    f"{'PASS' if r['cond_d'] else 'FAIL'}"
                )
        print("=" * 60)


def intensity_stratify(
    labels: np.ndarray,
    r_matrix: np.ndarray,
    target: np.ndarray,
    features_raw: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    n_groups: int | None = None,
    delta_target_threshold: float = 1.0,
    p_R_threshold: float = 0.01,
    p_feature_threshold: float = 0.05,
    min_sig_features: int = 3,
    min_subgroup_size: int = 30,
    random_state: int = 42,
) -> Stage2Result:
    if n_groups is None:
        n_groups = int(labels.max())
    n_samples = len(labels)
    n_features = r_matrix.shape[1]
    R_norm = np.linalg.norm(r_matrix, axis=1)

    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(n_features)]

    results = []
    subgroup_labels = np.array([f"R{labels[i]}" for i in range(n_samples)], dtype="U20")
    split_groups: list[int] = []
    retained_groups: list[int] = []
    feat_compare_rows: list[dict] = []

    for g in range(1, n_groups + 1):
        mask = labels == g
        n_g = int(mask.sum())

        if n_g < 2 * min_subgroup_size:
            retained_groups.append(g)
            results.append({
                "group": g, "n_samples": n_g,
                "n_strong": 0, "n_weak": n_g,
                "target_strong": np.nan,
                "target_weak": float(target[mask].mean()) if n_g > 0 else np.nan,
                "delta_target": 0.0,
                "R_strong": np.nan,
                "R_weak": float(R_norm[mask].mean()) if n_g > 0 else np.nan,
                "p_R": 1.0,
                "n_sig_features": 0, "sig_features": "",
                "min_subgroup": 0,
                "cond_a": False, "cond_b": False,
                "cond_c": False, "cond_d": False,
                "verdict": "retain",
            })
            continue

        r_g = r_matrix[mask]
        R_g = R_norm[mask]
        target_g = target[mask]

        km = KMeans(n_clusters=2, random_state=random_state, n_init=20)
        sub_labels = km.fit_predict(r_g)
        if R_g[sub_labels == 0].mean() > R_g[sub_labels == 1].mean():
            sub_labels = 1 - sub_labels

        strong = sub_labels == 1
        weak = sub_labels == 0
        n_strong = int(strong.sum()); n_weak = int(weak.sum())

        target_strong = float(target_g[strong].mean()) if n_strong > 0 else np.nan
        target_weak = float(target_g[weak].mean()) if n_weak > 0 else np.nan
        R_strong = float(R_g[strong].mean()) if n_strong > 0 else np.nan
        R_weak = float(R_g[weak].mean()) if n_weak > 0 else np.nan
        delta_target = (
            abs(target_strong - target_weak)
            if not (np.isnan(target_strong) or np.isnan(target_weak))
            else 0.0
        )

        cond_a = delta_target > delta_target_threshold

        if n_strong >= 2 and n_weak >= 2:
            _, p_R = stats.mannwhitneyu(R_g[strong], R_g[weak], alternative="two-sided")
        else:
            p_R = 1.0
        cond_b = p_R < p_R_threshold

        n_sig_feat = 0
        sig_feats: list[str] = []
        if features_raw is not None and n_strong >= 2 and n_weak >= 2:
            feat_g = features_raw[mask]
            for j in range(n_features):
                try:
                    _, p_f = stats.mannwhitneyu(
                        feat_g[strong, j], feat_g[weak, j], alternative="two-sided"
                    )
                    if p_f < p_feature_threshold:
                        n_sig_feat += 1
                        sig_feats.append(feature_names[j])
                except Exception:
                    pass
        cond_c = n_sig_feat >= min_sig_features

        min_n = min(n_strong, n_weak)
        cond_d = min_n >= min_subgroup_size

        split = cond_a and cond_b and cond_c and cond_d
        verdict = "split" if split else "retain"
        (split_groups if split else retained_groups).append(g)

        idx_g = np.where(mask)[0]
        for i, idx in enumerate(idx_g):
            if split:
                subgroup_labels[idx] = (
                    f"R{g}-Strong" if sub_labels[i] == 1 else f"R{g}-Weak"
                )

        results.append({
            "group": g, "n_samples": n_g,
            "n_strong": n_strong, "n_weak": n_weak,
            "target_strong": round(target_strong, 4),
            "target_weak": round(target_weak, 4),
            "delta_target": round(delta_target, 4),
            "R_strong": round(R_strong, 4),
            "R_weak": round(R_weak, 4),
            "p_R": float(p_R),
            "n_sig_features": n_sig_feat,
            "sig_features": ", ".join(sig_feats[:8]),
            "min_subgroup": min_n,
            "cond_a": cond_a, "cond_b": cond_b,
            "cond_c": cond_c, "cond_d": cond_d,
            "verdict": verdict,
        })

        if split and features_raw is not None:
            for j in range(n_features):
                s_vals = features_raw[mask][strong, j]
                w_vals = features_raw[mask][weak, j]
                try:
                    _, p = stats.mannwhitneyu(s_vals, w_vals, alternative="two-sided")
                except Exception:
                    p = 1.0
                sig = (
                    "***" if p < 0.001
                    else "**" if p < 0.01
                    else "*" if p < 0.05
                    else "ns"
                )
                feat_compare_rows.append({
                    "group": g,
                    "feature": feature_names[j],
                    "strong_mean": round(float(s_vals.mean()), 4),
                    "weak_mean": round(float(w_vals.mean()), 4),
                    "diff_pct": round(
                        (float(s_vals.mean()) - float(w_vals.mean()))
                        / (abs(float(w_vals.mean())) + 1e-10) * 100, 1
                    ),
                    "p_value": f"{p:.2e}",
                    "significance": sig,
                })

    summary_df = pd.DataFrame(results)
    feat_df = pd.DataFrame(feat_compare_rows) if feat_compare_rows else None

    return Stage2Result(
        summary=summary_df,
        subgroup_labels=subgroup_labels,
        split_groups=split_groups,
        retained_groups=retained_groups,
        feature_comparison=feat_df,
    )


def intensity_stratify_from_results(
    results,
    target: np.ndarray,
    features_raw: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    **kwargs,
) -> Stage2Result:
    """Convenience wrapper that extracts data from ``SHAPCompassResults``."""
    if results.ZF is None or results.ZS is None:
        raise ValueError(
            "SHAPCompassResults must contain ZF and ZS. "
            "These are computed during fit()."
        )
    r_matrix = np.sqrt(results.ZS ** 2 + results.ZF ** 2)
    if features_raw is None:
        features_raw = results.ZF
    if feature_names is None:
        feature_names = results.feature_names
    return intensity_stratify(
        labels=results.labels,
        r_matrix=r_matrix,
        target=target,
        features_raw=features_raw,
        feature_names=feature_names,
        n_groups=results.n_groups,
        **kwargs,
    )
