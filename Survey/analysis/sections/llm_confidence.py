"""
Section E: LLM confidence analysis (descriptive) — 4-point primary.

For each sentence, the LLM evaluator issues a verdict (CORRECT/INCORRECT/BORDERLINE)
and assigns a 4-point category. We check whether the LLM's 4-point category matches
the human majority's 4-point rating.

Improvements:
1. Agreement checked at 4-point exact match (primary) AND binary (supplementary)
2. Distinguishes CORRECT-aligned vs INCORRECT-aligned cases
3. Correlates disagreements with human difficulty ratings
4. Reports adjacency analysis: when LLM and humans disagree, how far apart are they?
"""

import numpy as np
from collections import Counter
from scipy import stats as sp_stats
from ..config import (
    COND_ORDER, N_ITEMS, CATEGORY_TO_RATING, RATING_TO_CATEGORY,
    CATEGORY_TO_BINARY,
)


def run(llm, plaus_valid, plaus_bin, diff_df, item_list, **kwargs):
    print("\n" + "=" * 70)
    print("E. LLM CONFIDENCE ANALYSIS (4-point primary)")
    print("  When LLM is confident (CORRECT/INCORRECT), does its 4-point")
    print("  category match human majority's 4-point rating?")
    print("  BORDERLINE = acknowledged uncertainty.")
    print("=" * 70)

    total_confident = 0
    total_borderline = 0

    # 4-point agreement tracking
    confident_4pt_agree = 0
    confident_4pt_disagree = 0

    # Binary agreement tracking (supplementary)
    confident_bin_agree = 0
    confident_bin_disagree = 0

    # Verdict-type breakdown (4-point)
    correct_4pt_agree = 0
    correct_4pt_disagree = 0
    incorrect_4pt_agree = 0
    incorrect_4pt_disagree = 0

    # Adjacency tracking: when they disagree, how far apart?
    disagree_distances = []

    # Difficulty correlation
    agree_difficulties = []
    disagree_difficulties = []

    per_item_detail = {}

    for i in item_list:
        if i not in llm:
            continue
        item_conf = 0
        item_4pt_agree = 0
        item_border = 0
        cond_details = {}

        for cond in COND_ORDER:
            j = llm[i]["judgments"][cond]
            llm_category = j["llm_category"]
            llm_rating = j["llm_rating"]
            llm_binary = j["llm_binary"]
            verdict = j["verdict"]

            cd = plaus_valid[
                (plaus_valid["item"] == i) & (plaus_valid["condition"] == cond)
            ]
            if len(cd) == 0 or llm_rating is None:
                continue

            # Human majority at 4-point level
            majority = cd["rating"].mode()
            if len(majority) == 0:
                continue
            human_rating = int(majority.iloc[0])
            human_category = RATING_TO_CATEGORY.get(human_rating)

            # Human majority at binary level
            cd_bin = plaus_bin[
                (plaus_bin["item"] == i) & (plaus_bin["condition"] == cond)
            ] if plaus_bin is not None else cd
            h_plaus = (cd_bin["binary_human"] == "PLAUSIBLE").sum() if "binary_human" in cd_bin.columns else 0
            h_implaus = (cd_bin["binary_human"] == "IMPLAUSIBLE").sum() if "binary_human" in cd_bin.columns else 0
            human_binary = "PLAUSIBLE" if h_plaus > h_implaus else "IMPLAUSIBLE"

            # 4-point exact match
            is_4pt_match = (llm_rating == human_rating)

            # Binary match
            is_bin_match = (llm_binary == human_binary)

            # Distance when they disagree
            distance = abs(llm_rating - human_rating)

            # Difficulty
            dd = diff_df[
                (diff_df["item"] == i) & (diff_df["condition"] == cond)
            ]
            mean_diff = float(dd["rating"].mean()) if len(dd) > 0 else None

            if verdict == "BORDERLINE":
                item_border += 1
                total_borderline += 1
                agreement_4pt = "BORDERLINE_SKIP"
                agreement_bin = "BORDERLINE_SKIP"
            elif verdict in ("CORRECT", "INCORRECT"):
                item_conf += 1
                total_confident += 1

                # 4-point agreement
                if is_4pt_match:
                    item_4pt_agree += 1
                    confident_4pt_agree += 1
                    agreement_4pt = "AGREE"
                    if mean_diff is not None:
                        agree_difficulties.append(mean_diff)
                    if verdict == "CORRECT":
                        correct_4pt_agree += 1
                    else:
                        incorrect_4pt_agree += 1
                else:
                    confident_4pt_disagree += 1
                    agreement_4pt = "DISAGREE"
                    disagree_distances.append(distance)
                    if mean_diff is not None:
                        disagree_difficulties.append(mean_diff)
                    if verdict == "CORRECT":
                        correct_4pt_disagree += 1
                    else:
                        incorrect_4pt_disagree += 1

                # Binary agreement (supplementary)
                if is_bin_match:
                    confident_bin_agree += 1
                    agreement_bin = "AGREE"
                else:
                    confident_bin_disagree += 1
                    agreement_bin = "DISAGREE"
            else:
                agreement_4pt = "UNKNOWN"
                agreement_bin = "UNKNOWN"

            cond_details[cond] = {
                "llm_category": llm_category,
                "llm_rating": llm_rating,
                "human_category": human_category,
                "human_rating": human_rating,
                "llm_binary": llm_binary,
                "human_binary": human_binary,
                "verdict": verdict,
                "agreement_4pt": agreement_4pt,
                "agreement_bin": agreement_bin,
                "distance": distance,
                "mean_difficulty": mean_diff,
            }

        per_item_detail[i] = {
            "n_confident": item_conf,
            "n_4pt_agree": item_4pt_agree,
            "n_borderline": item_border,
            "all_confident_4pt_agree": item_4pt_agree == item_conf if item_conf > 0 else True,
            "conditions": cond_details,
        }

    n_sets_all_agree = sum(
        1 for v in per_item_detail.values() if v["all_confident_4pt_agree"]
    )
    confident_4pt_acc = confident_4pt_agree / total_confident if total_confident > 0 else 0
    confident_bin_acc = confident_bin_agree / total_confident if total_confident > 0 else 0

    # ---- Print 4-point results (primary) ----
    print(f"\n  Verdict breakdown ({N_ITEMS * 4} sentences):")
    print(f"    Confident: {total_confident}, BORDERLINE: {total_borderline}")

    print(f"\n  === 4-POINT AGREEMENT (primary) ===")
    print(f"  On confident verdicts:")
    print(f"    4-pt exact match: {confident_4pt_agree}/{total_confident} ({confident_4pt_acc:.1%})")
    print(f"    4-pt disagree:    {confident_4pt_disagree}/{total_confident}")
    print(f"  Sets with full 4-pt agreement: {n_sets_all_agree}/{N_ITEMS}")

    # ---- Binary (supplementary) ----
    print(f"\n  === BINARY AGREEMENT (supplementary, task-relevant) ===")
    print(f"    Binary match: {confident_bin_agree}/{total_confident} ({confident_bin_acc:.1%})")

    # ---- Verdict-type breakdown (4-point) ----
    print(f"\n  --- Verdict-type breakdown (4-point) ---")
    n_correct = correct_4pt_agree + correct_4pt_disagree
    n_incorrect = incorrect_4pt_agree + incorrect_4pt_disagree
    print(f"  CORRECT verdicts (evaluator endorsed generator's assignment):")
    if n_correct > 0:
        print(f"    4-pt match with human: {correct_4pt_agree}/{n_correct} ({correct_4pt_agree/n_correct:.0%})")
        print(f"    4-pt mismatch:         {correct_4pt_disagree}/{n_correct} ({correct_4pt_disagree/n_correct:.0%})")
    print(f"  INCORRECT verdicts (evaluator corrected generator's assignment):")
    if n_incorrect > 0:
        print(f"    Correction matches human: {incorrect_4pt_agree}/{n_incorrect} ({incorrect_4pt_agree/n_incorrect:.0%})")
        print(f"    Correction diverges:      {incorrect_4pt_disagree}/{n_incorrect} ({incorrect_4pt_disagree/n_incorrect:.0%})")

    print(f"\n  4-cell summary:")
    print(f"    CORRECT+AGREE:     {correct_4pt_agree} (evaluator endorsed, human confirms)")
    print(f"    INCORRECT+AGREE:   {incorrect_4pt_agree} (evaluator corrected, correction matches human)")
    print(f"    CORRECT+DISAGREE:  {correct_4pt_disagree} (evaluator endorsed, human disagrees)")
    print(f"    INCORRECT+DISAGREE:{incorrect_4pt_disagree} (evaluator corrected, human disagrees with correction)")

    # ---- Adjacency analysis ----
    print(f"\n  --- Adjacency analysis (4-point disagreements only) ---")
    if disagree_distances:
        dist_counts = Counter(disagree_distances)
        print(f"    Total disagreements: {len(disagree_distances)}")
        for d in sorted(dist_counts.keys()):
            pct = dist_counts[d] / len(disagree_distances)
            label = {1: "adjacent (1 step)", 2: "two steps", 3: "opposite ends (3 steps)"}
            print(f"    Distance {d} ({label.get(d, '?')}): {dist_counts[d]} ({pct:.0%})")
        mean_dist = float(np.mean(disagree_distances))
        print(f"    Mean distance: {mean_dist:.2f}")
        print(f"    Adjacent (1-step) rate: {dist_counts.get(1, 0)}/{len(disagree_distances)} "
              f"({dist_counts.get(1, 0)/len(disagree_distances):.0%})")
    else:
        print(f"    No disagreements to analyze")

    # ---- Difficulty correlation ----
    print(f"\n  --- Difficulty-disagreement correlation ---")
    diff_corr = {}
    if agree_difficulties and disagree_difficulties:
        mean_agree_diff = float(np.mean(agree_difficulties))
        mean_disagree_diff = float(np.mean(disagree_difficulties))
        print(f"    Mean difficulty (4-pt agree):    {mean_agree_diff:.3f} (n={len(agree_difficulties)})")
        print(f"    Mean difficulty (4-pt disagree): {mean_disagree_diff:.3f} (n={len(disagree_difficulties)})")

        if len(disagree_difficulties) >= 3:
            stat, p_val = sp_stats.mannwhitneyu(
                disagree_difficulties, agree_difficulties,
                alternative="greater",
            )
            print(f"    Mann-Whitney U (disagree > agree): U={stat:.1f}, p={p_val:.4f}")
            diff_corr = {
                "mean_difficulty_agree": mean_agree_diff,
                "mean_difficulty_disagree": mean_disagree_diff,
                "n_agree": len(agree_difficulties),
                "n_disagree": len(disagree_difficulties),
                "mannwhitney_U": float(stat),
                "mannwhitney_p": float(p_val),
                "interpretation": (
                    "Disagreements concentrate in higher-difficulty items"
                    if p_val < 0.05
                    else "No significant difficulty difference between agree/disagree"
                ),
            }
        else:
            diff_corr = {
                "mean_difficulty_agree": mean_agree_diff,
                "mean_difficulty_disagree": mean_disagree_diff,
                "n_agree": len(agree_difficulties),
                "n_disagree": len(disagree_difficulties),
                "note": "Too few disagreements for statistical test",
            }

    # ---- Per-verdict summary ----
    verdict_counts = Counter()
    verdict_4pt_agree = Counter()
    verdict_bin_agree = Counter()
    for detail in per_item_detail.values():
        for cd in detail["conditions"].values():
            verdict_counts[cd["verdict"]] += 1
            if cd["agreement_4pt"] in ("AGREE", "BORDERLINE_SKIP"):
                verdict_4pt_agree[cd["verdict"]] += 1
            if cd["agreement_bin"] in ("AGREE", "BORDERLINE_SKIP"):
                verdict_bin_agree[cd["verdict"]] += 1
    print(f"\n  Per verdict type (4-pt | binary):")
    for v in ["CORRECT", "INCORRECT", "BORDERLINE"]:
        t = verdict_counts.get(v, 0)
        a4 = verdict_4pt_agree.get(v, 0)
        ab = verdict_bin_agree.get(v, 0)
        if t > 0:
            print(f"    {v:12s}: {t} total, 4pt-aligned={a4} ({a4/t:.0%}), "
                  f"bin-aligned={ab} ({ab/t:.0%})")

    return {
        "llm_confidence": {
            "total_confident": total_confident,
            "total_borderline": total_borderline,
            "fourpoint_agreement": {
                "agree": confident_4pt_agree,
                "disagree": confident_4pt_disagree,
                "accuracy": confident_4pt_acc,
                "sets_all_agree": n_sets_all_agree,
            },
            "binary_agreement_supplementary": {
                "agree": confident_bin_agree,
                "disagree": confident_bin_disagree,
                "accuracy": confident_bin_acc,
            },
            "verdict_type_breakdown": {
                "correct_4pt_agree": correct_4pt_agree,
                "correct_4pt_disagree": correct_4pt_disagree,
                "incorrect_4pt_agree": incorrect_4pt_agree,
                "incorrect_4pt_disagree": incorrect_4pt_disagree,
            },
            "adjacency_analysis": {
                "distances": dict(Counter(disagree_distances)) if disagree_distances else {},
                "mean_distance": float(np.mean(disagree_distances)) if disagree_distances else None,
                "n_adjacent": int(Counter(disagree_distances).get(1, 0)) if disagree_distances else 0,
                "n_total_disagree": len(disagree_distances),
            },
            "difficulty_correlation": diff_corr,
            "per_verdict": {
                v: {
                    "total": verdict_counts.get(v, 0),
                    "fourpoint_aligned": verdict_4pt_agree.get(v, 0),
                    "binary_aligned": verdict_bin_agree.get(v, 0),
                }
                for v in ["CORRECT", "INCORRECT", "BORDERLINE"]
            },
            "per_item": {str(i): per_item_detail[i] for i in item_list},
        },
    }
