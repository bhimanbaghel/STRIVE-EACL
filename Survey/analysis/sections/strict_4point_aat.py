"""
Section C2 (NEW): Proper 4-point AAT — LLM vs. individual annotators.

Section C compared LLM vs. human *majority* (apples-to-oranges).
This section does the correct AAT comparison: LLM vs. each individual
annotator (8 L-H pairs) against annotator vs. annotator (28 H-H pairs),
with bootstrapped non-inferiority on the delta.

Two tests:
  - Unweighted kappa AAT (primary): Does the LLM match humans on the exact
    4-point category? This matches the exact-category tier assignment gate.
  - Weighted kappa AAT (supporting): Does the LLM match humans on plausibility
    ordering? Adjacent-category disagreements are penalized less.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from ..config import (
    COND_ORDER, CATEGORY_TO_RATING, BOOTSTRAP_N, RANDOM_SEED, DELTAS,
)
from ..metrics import (
    weighted_cohens_kappa, cohens_kappa, summarize_bootstrap,
)


def run(plaus_valid, llm, annotator_ids, item_list, pivot_raw, **kwargs):
    print("\n" + "=" * 70)
    print("C2. PRIMARY: 4-POINT AAT (LLM vs. individual annotators)")
    print("  Primary metric: Does the LLM evaluator agree with humans as well")
    print("  as humans agree with each other? Tested via non-inferiority AAT.")
    print("  Unweighted kappa: exact 4-point category (primary)")
    print("  Weighted kappa: plausibility ordering (supporting)")
    print("=" * 70)

    actual_anns = [a for a in annotator_ids if a in pivot_raw.columns]
    n_sentences = len(pivot_raw)

    # ---- Build LLM rating vector aligned to pivot_raw index ----
    llm_ratings = []
    for item, cond in pivot_raw.index:
        if item in llm:
            j = llm[item]["judgments"][cond]
            r = j["llm_rating"]
            llm_ratings.append(float(r) if r is not None else np.nan)
        else:
            llm_ratings.append(np.nan)

    llm_vec = np.array(llm_ratings)  # 1-4 scale, NaN where LLM has no judgment
    human_matrix = pivot_raw[actual_anns].values.astype(float)

    n_valid = int(np.sum(~np.isnan(llm_vec)))
    print(f"\n  Sentences with LLM 4-point judgment: {n_valid}/{n_sentences}")

    # ================================================================
    # POINT ESTIMATES: Weighted kappa
    # ================================================================
    print(f"\n  --- Weighted Cohen's Kappa (quadratic weights) ---")

    hh_wk = {}
    for a1, a2 in combinations(range(len(actual_anns)), 2):
        y1 = human_matrix[:, a1]
        y2 = human_matrix[:, a2]
        k = weighted_cohens_kappa(y1 - 1, y2 - 1, n_categories=4)
        pair_name = f"{actual_anns[a1]}-{actual_anns[a2]}"
        hh_wk[pair_name] = float(k)
    hh_wk_mean = float(np.nanmean(list(hh_wk.values())))
    print(f"  H-H weighted kappa: mean = {hh_wk_mean:.4f} (n = {len(hh_wk)} pairs)")

    lh_wk = {}
    for i, aid in enumerate(actual_anns):
        y_h = human_matrix[:, i]
        k = weighted_cohens_kappa(llm_vec - 1, y_h - 1, n_categories=4)
        lh_wk[f"LLM-{aid}"] = float(k)
        print(f"    LLM-{aid}: wk = {k:.4f}")
    lh_wk_mean = float(np.nanmean(list(lh_wk.values())))
    wk_delta = lh_wk_mean - hh_wk_mean
    print(f"  L-H weighted kappa: mean = {lh_wk_mean:.4f}")
    print(f"  Delta (weighted): {wk_delta:+.4f}")

    # ================================================================
    # POINT ESTIMATES: Unweighted kappa
    # ================================================================
    print(f"\n  --- Unweighted Cohen's Kappa (exact agreement) ---")

    hh_uk = {}
    for a1, a2 in combinations(range(len(actual_anns)), 2):
        y1 = human_matrix[:, a1] - 1
        y2 = human_matrix[:, a2] - 1
        k = cohens_kappa(y1, y2)
        pair_name = f"{actual_anns[a1]}-{actual_anns[a2]}"
        hh_uk[pair_name] = float(k)
    hh_uk_mean = float(np.nanmean(list(hh_uk.values())))
    print(f"  H-H unweighted kappa: mean = {hh_uk_mean:.4f} (n = {len(hh_uk)} pairs)")

    lh_uk = {}
    for i, aid in enumerate(actual_anns):
        y_h = human_matrix[:, i] - 1
        k = cohens_kappa(llm_vec - 1, y_h)
        lh_uk[f"LLM-{aid}"] = float(k)
        print(f"    LLM-{aid}: uk = {k:.4f}")
    lh_uk_mean = float(np.nanmean(list(lh_uk.values())))
    uk_delta = lh_uk_mean - hh_uk_mean
    print(f"  L-H unweighted kappa: mean = {lh_uk_mean:.4f}")
    print(f"  Delta (unweighted): {uk_delta:+.4f}")

    # ================================================================
    # BOOTSTRAP
    # ================================================================
    print(f"\n  Bootstrap non-inferiority (n = {BOOTSTRAP_N})...")

    rng = np.random.RandomState(RANDOM_SEED + 100)
    all_idx = np.arange(n_sentences)
    n_anns = len(actual_anns)

    boot_wk_deltas = []
    boot_uk_deltas = []

    for _ in range(BOOTSTRAP_N):
        sidx = rng.choice(all_idx, size=len(all_idx), replace=True)

        b_hh_wk, b_hh_uk = [], []
        for a1, a2 in combinations(range(n_anns), 2):
            wk = weighted_cohens_kappa(
                human_matrix[sidx, a1] - 1, human_matrix[sidx, a2] - 1, n_categories=4
            )
            uk = cohens_kappa(human_matrix[sidx, a1] - 1, human_matrix[sidx, a2] - 1)
            if not np.isnan(wk):
                b_hh_wk.append(wk)
            if not np.isnan(uk):
                b_hh_uk.append(uk)

        b_lh_wk, b_lh_uk = [], []
        for i in range(n_anns):
            wk = weighted_cohens_kappa(
                llm_vec[sidx] - 1, human_matrix[sidx, i] - 1, n_categories=4
            )
            uk = cohens_kappa(llm_vec[sidx] - 1, human_matrix[sidx, i] - 1)
            if not np.isnan(wk):
                b_lh_wk.append(wk)
            if not np.isnan(uk):
                b_lh_uk.append(uk)

        if b_hh_wk and b_lh_wk:
            boot_wk_deltas.append(np.nanmean(b_lh_wk) - np.nanmean(b_hh_wk))
        if b_hh_uk and b_lh_uk:
            boot_uk_deltas.append(np.nanmean(b_lh_uk) - np.nanmean(b_hh_uk))

    boot_wk_deltas = np.array(boot_wk_deltas)
    boot_uk_deltas = np.array(boot_uk_deltas)

    print(f"\n  Weighted kappa AAT:")
    wk_boot_results = summarize_bootstrap(boot_wk_deltas, "  4-pt weighted kappa AAT")

    print(f"\n  Unweighted kappa AAT:")
    uk_boot_results = summarize_bootstrap(boot_uk_deltas, "  4-pt unweighted kappa AAT")

    # ================================================================
    # INTERPRETATION
    # ================================================================
    wk_pass = wk_boot_results["noninferiority"].get("delta_0.05", False)
    uk_pass = uk_boot_results["noninferiority"].get("delta_0.05", False)

    print(f"\n  --- Interpretation ---")
    print(f"  Unweighted kappa (exact):  LLM {'matches' if uk_pass else 'does NOT match'} "
          f"humans at exact 4-point category (delta=0.05: {'PASS' if uk_pass else 'FAIL'})")
    print(f"  Weighted kappa (ordering): LLM {'matches' if wk_pass else 'does NOT match'} "
          f"humans at plausibility ordering (delta=0.05: {'PASS' if wk_pass else 'FAIL'})")

    if wk_pass and not uk_pass:
        print(f"  -> LLM preserves the ordering but not exact categories.")
        print(f"  -> This reinforces binary collapse: the 4-point distinction")
        print(f"     is not reliably made by humans OR the LLM.")
    elif wk_pass and uk_pass:
        print(f"  -> LLM matches humans at both ordering AND exact category level.")

    return {
        "strict_4point_aat": {
            "n_sentences": n_sentences,
            "n_valid_llm": n_valid,
            "weighted_kappa": {
                "hh_kappas": hh_wk,
                "hh_mean": hh_wk_mean,
                "lh_kappas": lh_wk,
                "lh_mean": lh_wk_mean,
                "delta": wk_delta,
                "bootstrap": wk_boot_results,
            },
            "unweighted_kappa": {
                "hh_kappas": hh_uk,
                "hh_mean": hh_uk_mean,
                "lh_kappas": lh_uk,
                "lh_mean": lh_uk_mean,
                "delta": uk_delta,
                "bootstrap": uk_boot_results,
            },
        },
    }
