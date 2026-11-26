"""
Section I: Per-sentence example export for qualitative review.

Generates a CSV with every sentence showing:
- The sentence text, verb, agent, generator-assigned condition
- Each annotator's individual 4-point rating
- Human majority rating, rating spread (SD), self-reported difficulty
- LLM's 4-point category, LLM verdict
- Whether LLM and human majority agree (4-point and binary)
- Disagreement type (cross-polarity vs within-polarity vs agree)

Sorted by human rating spread (highest disagreement first) so collaborators
can immediately see the most contested sentences.

Also prints summary examples of high-disagreement and low-disagreement sentences.
"""

import numpy as np
import pandas as pd
from ..config import (
    COND_ORDER, CATEGORY_TO_RATING, RATING_TO_CATEGORY,
    CATEGORY_TO_BINARY, N_ITEMS,
)


def run(plaus_valid, diff_df, llm, items, annotator_ids, item_list, eval_prefix="evalv3", **kwargs):
    print("\n" + "=" * 70)
    print("I. PER-SENTENCE EXAMPLES EXPORT")
    print("=" * 70)

    actual_anns = sorted([a for a in annotator_ids if a in plaus_valid["annotator"].unique()])
    rows = []

    for i in item_list:
        if i not in items:
            continue
        for cond in COND_ORDER:
            if cond not in items[i]["conditions"]:
                continue

            cond_info = items[i]["conditions"][cond]
            sentence = cond_info.get("sentence", "")
            agent = cond_info.get("agent", "")
            verb = items[i].get("verb", "")
            tier = items[i].get("tier", "")
            model = items[i].get("model", "")

            # Individual human ratings
            cd = plaus_valid[
                (plaus_valid["item"] == i) & (plaus_valid["condition"] == cond)
            ]
            rating_map = {}
            for _, r in cd.iterrows():
                rating_map[r["annotator"]] = int(r["rating"])

            ratings_list = [rating_map.get(a, None) for a in actual_anns]
            valid_ratings = [r for r in ratings_list if r is not None]

            if not valid_ratings:
                continue

            # Human majority
            majority = cd["rating"].mode()
            human_majority = int(majority.iloc[0]) if len(majority) > 0 else None
            human_category = RATING_TO_CATEGORY.get(human_majority)
            human_binary = CATEGORY_TO_BINARY.get(human_category)
            rating_sd = float(cd["rating"].std())
            rating_mean = float(cd["rating"].mean())

            # Rating distribution
            rating_label = {1: "PE", 2: "PH", 3: "IH", 4: "IE"}
            dist = cd["rating"].value_counts().sort_index().to_dict()
            dist_str = " | ".join(f"{rating_label.get(k, k)}:{v}" for k, v in sorted(dist.items()))

            # Difficulty
            dd = diff_df[
                (diff_df["item"] == i) & (diff_df["condition"] == cond)
            ]
            mean_diff = float(dd["rating"].mean()) if len(dd) > 0 else None

            # LLM judgment
            llm_category = None
            llm_rating = None
            llm_binary = None
            llm_verdict = None
            if i in llm:
                j = llm[i]["judgments"].get(cond, {})
                llm_category = j.get("llm_category")
                llm_rating = j.get("llm_rating")
                llm_binary = j.get("llm_binary")
                llm_verdict = j.get("verdict")

            # Agreement classification
            if llm_rating is not None and human_majority is not None:
                is_4pt_match = llm_rating == human_majority
                is_bin_match = llm_binary == human_binary
                distance = abs(llm_rating - human_majority)

                if is_4pt_match:
                    disagree_type = "AGREE"
                elif is_bin_match:
                    disagree_type = "WITHIN_POLARITY"
                else:
                    disagree_type = "CROSS_POLARITY"
            else:
                is_4pt_match = None
                is_bin_match = None
                distance = None
                disagree_type = "NO_LLM"

            row = {
                "item": i,
                "condition_generator": cond,
                "verb": verb,
                "agent": agent,
                "sentence": sentence,
                "tier": tier,
                "model": model,
            }
            # Add individual annotator ratings (as category abbreviations)
            rating_label = {1: "PE", 2: "PH", 3: "IH", 4: "IE"}
            for a_idx, a in enumerate(actual_anns):
                raw = rating_map.get(a, None)
                row[a] = rating_label.get(raw, "") if raw else ""

            row.update({
                "rating_distribution": dist_str,
                "human_majority_rating": human_majority,
                "human_majority_abbrev": rating_label.get(human_majority, ""),
                "human_majority_category": human_category,
                "human_binary": human_binary,
                "rating_mean": round(rating_mean, 2),
                "rating_SD": round(rating_sd, 2),
                "mean_difficulty": round(mean_diff, 2) if mean_diff else "",
                "llm_category": llm_category or "",
                "llm_rating": llm_rating or "",
                "llm_binary": llm_binary or "",
                "llm_verdict": llm_verdict or "",
                "4pt_match": is_4pt_match,
                "binary_match": is_bin_match,
                "distance": distance,
                "disagree_type": disagree_type,
            })
            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by rating_SD descending (most disagreement first)
    df = df.sort_values("rating_SD", ascending=False).reset_index(drop=True)

    # Save CSV
    from ..config import sentence_examples_path
    out_path = sentence_examples_path(eval_prefix)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} sentences to {out_path}")
    print(f"  Sorted by rating SD (highest human disagreement first)")

    # ---- Print example summaries ----
    rating_label = {1: "PE", 2: "PH", 3: "IH", 4: "IE"}

    print(f"\n  --- HIGH H-H DISAGREEMENT (top 10 by SD) ---")
    print(f"  (PE=Plausible Easy, PH=Plausible Hard, IH=Implausible Hard, IE=Implausible Easy)")
    top = df.head(10)
    for _, r in top.iterrows():
        ann_ratings = " ".join(str(r.get(a, "?")) for a in actual_anns)
        human_cat = rating_label.get(r['human_majority_rating'], "?")
        print(f"    Item {r['item']:2d} {r['condition_generator']:20s} "
              f"SD={r['rating_SD']:.2f} ratings=[{ann_ratings}] "
              f"majority={human_cat} "
              f"LLM={r['llm_category']} ({r['disagree_type']})")
        if r['sentence']:
            print(f"      \"{r['sentence'][:100]}{'...' if len(str(r['sentence'])) > 100 else ''}\"")

    print(f"\n  --- LOW H-H DISAGREEMENT (bottom 10 by SD) ---")
    bottom = df.tail(10)
    for _, r in bottom.iterrows():
        ann_ratings = " ".join(str(r.get(a, "?")) for a in actual_anns)
        human_cat = rating_label.get(r['human_majority_rating'], "?")
        print(f"    Item {r['item']:2d} {r['condition_generator']:20s} "
              f"SD={r['rating_SD']:.2f} ratings=[{ann_ratings}] "
              f"majority={human_cat} "
              f"LLM={r['llm_category']} ({r['disagree_type']})")

    # ---- Summary stats ----
    n_agree = (df["disagree_type"] == "AGREE").sum()
    n_within = (df["disagree_type"] == "WITHIN_POLARITY").sum()
    n_cross = (df["disagree_type"] == "CROSS_POLARITY").sum()

    # Per-condition SD summary
    print(f"\n  --- Human disagreement summary ---")
    print(f"  Overall: {n_agree} agree, {n_within} within-polarity, {n_cross} cross-polarity")
    for cond in COND_ORDER:
        cd = df[df["condition_generator"] == cond]
        print(f"    {cond:20s}: mean SD={cd['rating_SD'].mean():.2f}, "
              f"max SD={cd['rating_SD'].max():.2f}, "
              f"sentences with SD>1.0: {(cd['rating_SD'] > 1.0).sum()}/{len(cd)}")

    # Identify consensus sentences (SD=0 or near-zero)
    perfect = df[df["rating_SD"] < 0.01]
    near_perfect = df[df["rating_SD"] < 0.5]
    print(f"\n  Perfect consensus (SD=0): {len(perfect)} sentences")
    print(f"  Near-consensus (SD<0.5): {len(near_perfect)} sentences")
    print(f"  High disagreement (SD>1.0): {(df['rating_SD'] > 1.0).sum()} sentences")

    return {
        "example_export": {
            "csv_path": str(out_path),
            "n_sentences": len(df),
            "disagree_summary": {
                "agree": int(n_agree),
                "within_polarity": int(n_within),
                "cross_polarity": int(n_cross),
            },
            "sd_summary": {
                "perfect_consensus": int(len(perfect)),
                "near_consensus": int(len(near_perfect)),
                "high_disagreement": int((df["rating_SD"] > 1.0).sum()),
            },
        },
    }