"""
Section D: Primary binary plausibility AAT (sentence-level).

NEW: Per-condition AAT using Gwet's AC1 and percent agreement with
bootstrapped CIs, addressing the kappa paradox for skewed conditions.

The kappa paradox: when prevalence is extreme (e.g., plausible_easy
sentences are nearly all rated PLAUSIBLE), observed agreement is high
but so is chance agreement, crushing kappa even when raters genuinely
agree. AC1 and percent agreement avoid this artifact.
"""

import numpy as np
from itertools import combinations
from ..config import COND_ORDER, BOOTSTRAP_N, RANDOM_SEED, rating_to_binary
from ..metrics import (
    cohens_kappa, gwets_ac1, percent_agreement, summarize_bootstrap,
)


def run(plaus_valid, llm, annotator_ids, item_list, **kwargs):
    print("\n" + "=" * 70)
    print("D. SUPPLEMENTARY: BINARY PLAUSIBILITY (sentence-level, n=120)")
    print("  Human: rating 1-2 = PLAUSIBLE, 3-4 = IMPLAUSIBLE")
    print("  LLM: from actual_* category mapped to binary")
    print("  NOTE: Binary binarization sits at the PH/IH boundary where human")
    print("  disagreement is highest. Reported descriptively, not as primary AAT.")
    print("=" * 70)

    # ---- Prepare binary data ----
    pv = plaus_valid.copy()
    pv["binary_human"] = pv["rating"].map(rating_to_binary)
    plaus_bin = pv.dropna(subset=["binary_human"]).copy()

    llm_binary_map = {}
    for i in item_list:
        if i not in llm:
            continue
        for cond in COND_ORDER:
            j = llm[i]["judgments"][cond]
            llm_binary_map[(i, cond)] = j["llm_binary"]

    bin_pivot = plaus_bin.pivot_table(
        index=["item", "condition"], columns="annotator",
        values="binary_human", aggfunc="first",
    )
    bin_num = bin_pivot.copy()
    for col in bin_num.columns:
        bin_num[col] = bin_num[col].map({"PLAUSIBLE": 1, "IMPLAUSIBLE": 0})
    bin_num = bin_num.astype(float)

    llm_bin_vec = np.array([
        1.0 if llm_binary_map.get((item, cond)) == "PLAUSIBLE"
        else (0.0 if llm_binary_map.get((item, cond)) == "IMPLAUSIBLE" else np.nan)
        for item, cond in bin_pivot.index
    ])

    actual_anns = [a for a in annotator_ids if a in bin_num.columns]
    human_matrix = bin_num[actual_anns].values.astype(float)

    # ---- Overall H-H binary kappa ----
    print(f"\n  H-H Binary Agreement (pairwise, {len(bin_num)} sentences):")
    bin_hh_kappas = {}
    for a1, a2 in combinations(annotator_ids, 2):
        if a1 in bin_num.columns and a2 in bin_num.columns:
            k = cohens_kappa(bin_num[a1].values, bin_num[a2].values)
            bin_hh_kappas[f"{a1}-{a2}"] = float(k)
            print(f"    {a1}-{a2}: kappa={k:.4f}")
    bin_hh_mean = float(np.nanmean(list(bin_hh_kappas.values())))
    print(f"  Mean H-H binary kappa: {bin_hh_mean:.4f}")

    # ---- Overall L-H binary kappa ----
    print(f"\n  L-H Binary Agreement (pairwise, {len(bin_num)} sentences):")
    bin_lh_kappas = {}
    for aid in annotator_ids:
        if aid in bin_num.columns:
            k = cohens_kappa(llm_bin_vec, bin_num[aid].values)
            bin_lh_kappas[f"LLM-{aid}"] = float(k)
            print(f"    LLM-{aid}: kappa={k:.4f}")
    bin_lh_mean = float(np.nanmean(list(bin_lh_kappas.values())))
    bin_delta = bin_lh_mean - bin_hh_mean
    print(f"  Mean L-H binary kappa: {bin_lh_mean:.4f}")
    print(f"  Delta = {bin_delta:+.4f}")

    # ---- Overall bootstrap AAT ----
    print(f"\n  Bootstrap non-inferiority (n={BOOTSTRAP_N})...")
    rng = np.random.RandomState(RANDOM_SEED)
    all_idx = np.arange(len(bin_num))
    bin_boot = []
    for _ in range(BOOTSTRAP_N):
        sidx = rng.choice(all_idx, size=len(all_idx), replace=True)
        hh = [
            cohens_kappa(human_matrix[sidx, i], human_matrix[sidx, j])
            for i, j in combinations(range(len(actual_anns)), 2)
        ]
        hh = [k for k in hh if not np.isnan(k)]
        lh = [
            cohens_kappa(llm_bin_vec[sidx], human_matrix[sidx, i])
            for i in range(len(actual_anns))
        ]
        lh = [k for k in lh if not np.isnan(k)]
        if hh and lh:
            bin_boot.append(np.nanmean(lh) - np.nanmean(hh))
    bin_boot = np.array(bin_boot)
    bin_boot_results = summarize_bootstrap(bin_boot, "Binary AAT (kappa)")

    # ================================================================
    # PER-CONDITION AAT (NEW)
    # Uses AC1 + percent agreement instead of kappa to avoid
    # the kappa paradox with skewed prevalence.
    # ================================================================
    print(f"\n  --- Per-Condition AAT (AC1 + %agree) ---")
    print(f"  (AC1 avoids kappa paradox when prevalence is extreme)")

    per_cond_aat = {}
    for cond in COND_ORDER:
        cond_mask = np.array([c == cond for _, c in bin_pivot.index])
        cond_idx = np.where(cond_mask)[0]
        if len(cond_idx) == 0:
            continue

        # Point estimates
        hh_ac1s, hh_pcts = [], []
        for i, j in combinations(range(len(actual_anns)), 2):
            ac = gwets_ac1(human_matrix[cond_idx, i], human_matrix[cond_idx, j])
            pa = percent_agreement(human_matrix[cond_idx, i], human_matrix[cond_idx, j])
            if not np.isnan(ac):
                hh_ac1s.append(ac)
            if not np.isnan(pa):
                hh_pcts.append(pa)

        lh_ac1s, lh_pcts = [], []
        for i in range(len(actual_anns)):
            ac = gwets_ac1(llm_bin_vec[cond_idx], human_matrix[cond_idx, i])
            pa = percent_agreement(llm_bin_vec[cond_idx], human_matrix[cond_idx, i])
            if not np.isnan(ac):
                lh_ac1s.append(ac)
            if not np.isnan(pa):
                lh_pcts.append(pa)

        hh_ac1_mean = float(np.nanmean(hh_ac1s)) if hh_ac1s else np.nan
        lh_ac1_mean = float(np.nanmean(lh_ac1s)) if lh_ac1s else np.nan
        ac1_delta = lh_ac1_mean - hh_ac1_mean

        hh_pct_mean = float(np.nanmean(hh_pcts)) if hh_pcts else np.nan
        lh_pct_mean = float(np.nanmean(lh_pcts)) if lh_pcts else np.nan
        pct_delta = lh_pct_mean - hh_pct_mean

        # Bootstrap per-condition
        rng_cond = np.random.RandomState(RANDOM_SEED + hash(cond) % 10000)
        boot_ac1, boot_pct = [], []
        for _ in range(BOOTSTRAP_N):
            sidx = rng_cond.choice(cond_idx, size=len(cond_idx), replace=True)
            b_hh_ac1 = [
                gwets_ac1(human_matrix[sidx, i], human_matrix[sidx, j])
                for i, j in combinations(range(len(actual_anns)), 2)
            ]
            b_hh_ac1 = [k for k in b_hh_ac1 if not np.isnan(k)]
            b_lh_ac1 = [
                gwets_ac1(llm_bin_vec[sidx], human_matrix[sidx, i])
                for i in range(len(actual_anns))
            ]
            b_lh_ac1 = [k for k in b_lh_ac1 if not np.isnan(k)]

            b_hh_pct = [
                percent_agreement(human_matrix[sidx, i], human_matrix[sidx, j])
                for i, j in combinations(range(len(actual_anns)), 2)
            ]
            b_hh_pct = [k for k in b_hh_pct if not np.isnan(k)]
            b_lh_pct = [
                percent_agreement(llm_bin_vec[sidx], human_matrix[sidx, i])
                for i in range(len(actual_anns))
            ]
            b_lh_pct = [k for k in b_lh_pct if not np.isnan(k)]

            if b_hh_ac1 and b_lh_ac1:
                boot_ac1.append(np.nanmean(b_lh_ac1) - np.nanmean(b_hh_ac1))
            if b_hh_pct and b_lh_pct:
                boot_pct.append(np.nanmean(b_lh_pct) - np.nanmean(b_hh_pct))

        boot_ac1 = np.array(boot_ac1)
        boot_pct = np.array(boot_pct)

        ac1_boot_res = summarize_bootstrap(boot_ac1, f"  {cond} AC1") if len(boot_ac1) > 0 else {}
        pct_boot_res = summarize_bootstrap(boot_pct, f"  {cond} %agree") if len(boot_pct) > 0 else {}

        # Prevalence info for interpretation
        llm_prev = float(np.nanmean(llm_bin_vec[cond_idx]))
        human_prev = float(np.nanmean(human_matrix[cond_idx]))

        per_cond_aat[cond] = {
            "n_sentences": len(cond_idx),
            "prevalence_plausible": {"llm": llm_prev, "human_mean": human_prev},
            "ac1": {
                "hh_mean": hh_ac1_mean, "lh_mean": lh_ac1_mean,
                "delta": ac1_delta, "bootstrap": ac1_boot_res,
            },
            "percent_agreement": {
                "hh_mean": hh_pct_mean, "lh_mean": lh_pct_mean,
                "delta": pct_delta, "bootstrap": pct_boot_res,
            },
        }

        print(f"\n    {cond} (n={len(cond_idx)}, prev={human_prev:.0%} PLAUSIBLE):")
        print(f"      AC1: HH={hh_ac1_mean:.4f}, LH={lh_ac1_mean:.4f}, delta={ac1_delta:+.4f}")
        print(f"      %ag: HH={hh_pct_mean:.4f}, LH={lh_pct_mean:.4f}, delta={pct_delta:+.4f}")

    # ---- Per-condition LLM-Human binary agreement (individual ratings) ----
    print(f"\n  Per-condition LLM-Human binary agreement (all individual ratings):")
    bin_per_cond_raw = {}
    for cond in COND_ORDER:
        cond_mask = np.array([c == cond for _, c in bin_pivot.index])
        cond_idx = np.where(cond_mask)[0]
        if len(cond_idx) == 0:
            continue
        agrees = 0
        total = 0
        for idx in cond_idx:
            for aid in actual_anns:
                h = bin_num.iloc[idx][aid]
                l = llm_bin_vec[idx]
                if not np.isnan(h) and not np.isnan(l):
                    if h == l:
                        agrees += 1
                    total += 1
        acc = agrees / total if total > 0 else 0
        bin_per_cond_raw[cond] = {
            "agreement": float(acc), "n_ratings": total,
            "n_sentences": len(cond_idx),
        }
        print(f"    {cond:20s}: {acc:.1%} (n_ratings={total})")

    return {
        "binary_sentence_level": {
            "per_condition_agreement": bin_per_cond_raw,
            "hh_kappas": bin_hh_kappas,
            "hh_mean": bin_hh_mean,
            "lh_kappas": bin_lh_kappas,
            "lh_mean": bin_lh_mean,
            "delta": float(bin_delta),
            "bootstrap": bin_boot_results,
            "per_condition_aat": per_cond_aat,
        },
        # Pass through
        "_plaus_bin": plaus_bin,
        "_bin_num": bin_num,
        "_llm_bin_vec": llm_bin_vec,
        "_human_matrix": human_matrix,
        "_actual_anns": actual_anns,
        "_llm_binary_map": llm_binary_map,
    }