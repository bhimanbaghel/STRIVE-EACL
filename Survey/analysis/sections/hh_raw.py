"""Section B: Raw human-human agreement (4-point ordinal)."""

import numpy as np
from itertools import combinations
from ..config import COND_ORDER
from ..metrics import weighted_cohens_kappa, krippendorffs_alpha


def run(plaus_valid, annotator_ids, **kwargs):
    print("\n" + "=" * 70)
    print("B. RAW H-H AGREEMENT (4-point ordinal)")
    print("=" * 70)

    pivot_raw = plaus_valid.pivot_table(
        index=["item", "condition"], columns="annotator",
        values="rating", aggfunc="first"
    )
    alpha = krippendorffs_alpha(pivot_raw.values.astype(float))
    print(f"\n  Krippendorff's alpha: {alpha:.4f}")

    raw_kappas = {}
    for a1, a2 in combinations(annotator_ids, 2):
        if a1 in pivot_raw.columns and a2 in pivot_raw.columns:
            k = weighted_cohens_kappa(
                pivot_raw[a1].values.astype(float),
                pivot_raw[a2].values.astype(float),
                n_categories=5,
            )
            raw_kappas[f"{a1}-{a2}"] = float(k)
            print(f"    {a1}-{a2}: {k:.4f}")

    mean_k = np.nanmean(list(raw_kappas.values()))
    sd_k = np.nanstd(list(raw_kappas.values()))
    print(f"  Mean: {mean_k:.4f} +/- {sd_k:.4f}")

    print(f"\n  Per-condition median:")
    cond_medians = {}
    for cond in COND_ORDER:
        med = float(plaus_valid[plaus_valid["condition"] == cond]["rating"].median())
        cond_medians[cond] = {"median": med}
        print(f"    {cond:20s}: median={med:.1f}")

    return {
        "hh_agreement_raw": {
            "krippendorff_alpha": float(alpha),
            "pairwise_kappa": raw_kappas,
            "mean_kappa": float(mean_k),
            "sd_kappa": float(sd_k),
            "min_kappa": float(np.nanmin(list(raw_kappas.values()))),
            "max_kappa": float(np.nanmax(list(raw_kappas.values()))),
            "per_condition_median": cond_medians,
        },
        # Pass through for downstream sections
        "_pivot_raw": pivot_raw,
    }
