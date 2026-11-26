"""
Section C: Strict 4-point comparison (sentence-level).

NEW: Includes 4x4 confusion matrix (LLM category x human majority category)
to visualize the systematic pattern of humans pulling hard conditions toward
easy poles while the LLM preserves the generator's gradient.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from ..config import (
    COND_ORDER, CATEGORY_TO_RATING, RATING_TO_CATEGORY,
)
from ..metrics import weighted_cohens_kappa, cohens_kappa


def _build_confusion_matrix(ss_df):
    """
    Build a 4x4 confusion matrix: rows = LLM category, cols = human majority category.

    This is the most informative single summary of LLM-human agreement at the
    4-point level. It directly shows:
    - Diagonal = exact agreement
    - Off-diagonal patterns reveal systematic biases
    - Whether humans pull hard conditions toward easy poles
    """
    cats = list(CATEGORY_TO_RATING.keys())  # ordered pe, ph, ih, ie
    matrix = np.zeros((4, 4), dtype=int)
    for _, row in ss_df.iterrows():
        if row["llm_category"] in cats and row["human_category"] in cats:
            li = cats.index(row["llm_category"])
            hi = cats.index(row["human_category"])
            matrix[li, hi] += 1

    # Also build a normalized version (row-normalized: per LLM category)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    return cats, matrix, matrix_norm


def run(plaus_valid, diff_df, llm, annotator_ids, item_list, pivot_raw, **kwargs):
    print("\n" + "=" * 70)
    print("C. STRICT 4-POINT COMPARISON (sentence-level)")
    print("  LLM assigns a 4-point category (from actual_* columns).")
    print("  Human majority assigns a 4-point rating.")
    print("  We compare these two judgments directly.")
    print("=" * 70)

    strict_sentences = []
    for i in item_list:
        if i not in llm:
            continue
        for cond in COND_ORDER:
            j = llm[i]["judgments"][cond]
            llm_rating = j["llm_rating"]
            llm_category = j["llm_category"]

            cd = plaus_valid[
                (plaus_valid["item"] == i) & (plaus_valid["condition"] == cond)
            ]
            if len(cd) == 0 or llm_rating is None:
                continue
            majority = cd["rating"].mode()
            if len(majority) == 0:
                continue
            human_rating = int(majority.iloc[0])
            human_category = RATING_TO_CATEGORY.get(human_rating)

            strict_sentences.append({
                "item": i, "condition": cond,
                "llm_rating": llm_rating, "llm_category": llm_category,
                "human_rating": human_rating, "human_category": human_category,
                "exact_match": int(llm_rating == human_rating),
                "llm_verdict": j["verdict"],
            })

    ss_df = pd.DataFrame(strict_sentences, columns=[
        "item", "condition", "llm_rating", "llm_category",
        "human_rating", "human_category", "exact_match", "llm_verdict",
    ])
    n_strict = len(ss_df)
    if n_strict == 0:
        raise ValueError(
            "No valid LLM 4-point judgments were available for strict comparison; "
            "check evaluator success and actual_* columns before running AAT analysis."
        )
    exact_rate = ss_df["exact_match"].mean()

    print(f"\n  LLM vs Human majority (n={n_strict} sentences):")
    print(f"    Exact match: {exact_rate:.1%}")

    # ---- 4x4 Confusion matrix (NEW) ----
    cats, conf_matrix, conf_norm = _build_confusion_matrix(ss_df)
    print(f"\n  4x4 Confusion Matrix (LLM rows x Human columns):")
    header = "             " + "  ".join(f"{c[:7]:>7s}" for c in cats)
    print(f"  {header}")
    for i, cat in enumerate(cats):
        row_str = "  ".join(f"{conf_matrix[i, j]:>7d}" for j in range(4))
        print(f"  {cat[:12]:12s} {row_str}")

    print(f"\n  Row-normalized (what humans say when LLM says X):")
    print(f"  {header}")
    for i, cat in enumerate(cats):
        row_str = "  ".join(f"{conf_norm[i, j]:>7.0%}" for j in range(4))
        print(f"  {cat[:12]:12s} {row_str}")

    # ---- Per condition ----
    print(f"\n  Per condition:")
    strict_per_cond = {}
    for cond in COND_ORDER:
        cd = ss_df[ss_df["condition"] == cond]
        if len(cd) == 0:
            continue
        exact = float(cd["exact_match"].mean())
        llm_dist = cd["llm_category"].value_counts().to_dict()
        human_dist = cd["human_category"].value_counts().to_dict()
        strict_per_cond[cond] = {
            "exact_match": exact, "n": len(cd),
            "llm_distribution": llm_dist, "human_distribution": human_dist,
        }
        print(f"    {cond:20s}: exact={exact:.0%} (n={len(cd)})")
        print(f"      LLM assigns:   {llm_dist}")
        print(f"      Human assigns: {human_dist}")

    # ---- Kappas ----
    llm_ratings = ss_df["llm_rating"].values.astype(float) - 1
    human_ratings = ss_df["human_rating"].values.astype(float) - 1
    strict_kappa_w = weighted_cohens_kappa(llm_ratings, human_ratings, n_categories=4)
    strict_kappa_uw = cohens_kappa(llm_ratings, human_ratings)

    print(f"\n  LLM vs Human majority (4-point):")
    print(f"    Weighted kappa:   {strict_kappa_w:.4f}")
    print(f"    Unweighted kappa: {strict_kappa_uw:.4f}")

    hh_strict_kappas = []
    for a1, a2 in combinations(annotator_ids, 2):
        if a1 in pivot_raw.columns and a2 in pivot_raw.columns:
            y1 = pivot_raw[a1].values.astype(float) - 1
            y2 = pivot_raw[a2].values.astype(float) - 1
            k = cohens_kappa(y1, y2)
            hh_strict_kappas.append(k)
    hh_strict_mean = float(np.nanmean(hh_strict_kappas))
    strict_delta = float(strict_kappa_uw) - hh_strict_mean

    print(f"\n  H-H unweighted kappa: {hh_strict_mean:.4f}")
    print(f"  L-H unweighted kappa: {strict_kappa_uw:.4f}")
    print(f"  Delta: {strict_delta:+.4f}")

    # ---- Human rating spread ----
    spread_data = {}
    print(f"\n  Human rating spread per condition:")
    for cond in COND_ORDER:
        cd = plaus_valid[plaus_valid["condition"] == cond]
        spread_data[cond] = {
            "std_rating": float(cd["rating"].std()),
            "mean_rating": float(cd["rating"].mean()),
            "median_rating": float(cd["rating"].median()),
            "rating_counts": {
                str(v): int(c)
                for v, c in cd["rating"].value_counts().sort_index().items()
            },
        }
        print(f"    {cond:20s}: sigma={cd['rating'].std():.2f}, median={cd['rating'].median():.1f}")

    # ---- Difficulty evidence ----
    merged = plaus_valid.merge(
        diff_df[["annotator", "item", "condition", "rating"]],
        on=["annotator", "item", "condition"], suffixes=("_plaus", "_diff"),
    )
    diff_by_cond = {}
    print(f"\n  Difficulty by condition:")
    for cond in COND_ORDER:
        cd = merged[merged["condition"] == cond]["rating_diff"]
        diff_by_cond[cond] = {"mean": float(cd.mean()), "std": float(cd.std())}
        print(f"    {cond:20s}: mean={cd.mean():.2f}")

    print(f"\n  --> Hard conditions: high spread, low exact match, higher difficulty")
    print(f"  --> Motivates binary (plausible/implausible) comparison")

    # ---- Serialize confusion matrix ----
    conf_dict = {
        "labels": cats,
        "counts": conf_matrix.tolist(),
        "row_normalized": [[round(v, 4) for v in row] for row in conf_norm.tolist()],
    }

    return {
        "strict_comparison": {
            "overall_exact": float(exact_rate),
            "n_sentences": n_strict,
            "confusion_matrix": conf_dict,
            "lh_kappa_weighted": float(strict_kappa_w),
            "lh_kappa_unweighted": float(strict_kappa_uw),
            "hh_strict_kappa_mean": hh_strict_mean,
            "strict_delta": strict_delta,
            "per_condition": strict_per_cond,
            "difficulty_by_condition": diff_by_cond,
            "human_spread": spread_data,
        },
        # Pass through for downstream
        "_ss_df": ss_df,
        "_spread_data": spread_data,
        "_diff_by_cond": diff_by_cond,
        "_merged_plaus_diff": merged,
    }
