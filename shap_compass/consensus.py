"""Within-regime directional consensus check.

After clustering, each regime's top-N features (ranked by signal strength r)
are inspected for directional agreement at the neuron level. If most
neurons within a regime share the same sign of ``sin(theta)``, the regime
is considered to have a consensus direction for that feature; otherwise
the (regime, feature) pair is labelled "split".

The Directional Consensus Rate (DCR) = 1 - n_split / n_total summarises
overall clustering quality from the directional perspective. It is a
diagnostic, complementary to the cross-regime DCI defined in ``dci.py``
(which lives at a different layer of the analysis).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConsensusReport:
    """Within-regime directional consensus summary."""

    dcr: float = 1.0
    n_split: int = 0
    n_total: int = 0
    details: pd.DataFrame = None
    quality: str = "excellent"

    def summary(self) -> str:
        lines = [
            f"DCR = {self.dcr:.1%}  (split {self.n_split}/{self.n_total})",
            "=" * 55,
        ]
        if self.details is not None and len(self.details) > 0:
            for g in sorted(self.details["group"].unique()):
                g_res = self.details[self.details["group"] == g]
                splits = g_res[g_res["status"] == "split"]
                if len(splits) > 0:
                    feats = ", ".join(
                        f"{r['feature']}({r['sin_pos']}/{r['sin_neg']})"
                        for _, r in splits.iterrows()
                    )
                    lines.append(f"  ! R{g}: split -> {feats}")
                else:
                    lines.append(f"  * R{g}: all consensus/majority")
        lines.append("=" * 55)
        if self.dcr >= 0.9:
            lines.append("Verdict: Excellent (split <= 10%)")
        elif self.dcr >= 0.8:
            lines.append("Verdict: Acceptable (split 10-20%)")
        else:
            lines.append("Verdict: Warning (split > 20%) -- consider increasing k")
        return "\n".join(lines)


def check_consensus(
    neuron_theta: np.ndarray,
    neuron_r: np.ndarray,
    group_theta: np.ndarray,
    group_r: np.ndarray,
    neuron_labels: np.ndarray,
    n_groups: int,
    feature_names: list[str],
    top_n: int = 5,
    consensus_threshold: float = 6 / 7,
    majority_threshold: float = 5 / 7,
) -> ConsensusReport:
    """Directional consensus check on the top-N features of each regime."""
    records: list[dict] = []
    n_split = 0
    n_total = 0

    for g in range(1, n_groups + 1):
        idx_list = np.where(neuron_labels == g)[0]
        n_neurons = len(idx_list)
        if n_neurons == 0:
            continue

        if group_r is not None:
            top_idx = np.argsort(-group_r[g - 1])[:top_n]
        else:
            top_idx = np.arange(min(top_n, neuron_theta.shape[1]))

        for j in top_idx:
            n_total += 1
            sins = [np.sin(neuron_theta[ni, j]) for ni in idx_list]
            sin_pos = sum(1 for s in sins if s > 0)
            sin_neg = n_neurons - sin_pos
            cos_pos = int(sum(1 for c in (np.cos(neuron_theta[ni, j]) for ni in idx_list) if c > 0))
            cos_neg = n_neurons - cos_pos

            sin_consensus = max(sin_pos, sin_neg) / n_neurons
            cos_consensus = max(cos_pos, cos_neg) / n_neurons

            if sin_consensus >= consensus_threshold:
                status = "consensus"
            elif sin_consensus >= majority_threshold:
                status = "majority"
            else:
                status = "split"
                n_split += 1

            records.append({
                "group": g,
                "feature": feature_names[j] if j < len(feature_names) else f"F{j+1}",
                "r_rank": int(np.where(top_idx == j)[0][0]) + 1,
                "r": round(float(group_r[g - 1, j]), 4) if group_r is not None else None,
                "L3_cos": round(float(np.cos(group_theta[g - 1, j])), 4),
                "L3_sin": round(float(np.sin(group_theta[g - 1, j])), 4),
                "n_neurons": n_neurons,
                "sin_pos": sin_pos,
                "sin_neg": sin_neg,
                "sin_consensus": round(sin_consensus, 4),
                "cos_pos": cos_pos,
                "cos_neg": cos_neg,
                "cos_consensus": round(cos_consensus, 4),
                "status": status,
            })

    dcr = 1 - n_split / n_total if n_total > 0 else 1.0
    quality = "excellent" if dcr >= 0.9 else "acceptable" if dcr >= 0.8 else "warning"

    return ConsensusReport(
        dcr=dcr,
        n_split=n_split,
        n_total=n_total,
        details=pd.DataFrame(records) if records else pd.DataFrame(),
        quality=quality,
    )


def check_consensus_from_results(results, top_n: int = 5) -> ConsensusReport:
    """Convenience wrapper that extracts data from a ``SHAPCompassResults``."""
    if results.neuron_theta is None or results.neuron_labels is None:
        raise ValueError(
            "SHAPCompassResults must contain neuron-level data. "
            "Run fit() with use_som=True."
        )

    n_groups = results.n_groups
    n_features = results.neuron_theta.shape[1]
    group_r = np.zeros((n_groups, n_features))

    for g in range(1, n_groups + 1):
        mask = results.labels == g
        if mask.any():
            if results.ZF is not None and results.ZS is not None:
                mean_ZF = results.ZF[mask].mean(axis=0)
                mean_ZS = results.ZS[mask].mean(axis=0)
                group_r[g - 1] = np.sqrt(mean_ZS ** 2 + mean_ZF ** 2)
            else:
                n_mask = results.neuron_labels == g
                if n_mask.any():
                    group_r[g - 1] = results.neuron_r[n_mask].mean(axis=0)

    return check_consensus(
        neuron_theta=results.neuron_theta,
        neuron_r=results.neuron_r,
        group_theta=results.group_theta,
        group_r=group_r,
        neuron_labels=results.neuron_labels,
        n_groups=n_groups,
        feature_names=results.feature_names,
        top_n=top_n,
    )
