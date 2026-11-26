"""
Section H: Item-level diagnostics — 4-point primary.

Identifies which specific items drive 4-point LLM-human disagreements.
Reports both 4-point mismatches (primary) and binary mismatches (supplementary).
"""

import numpy as np
from ..config import COND_ORDER, N_ITEMS, RATING_TO_CATEGORY, CATEGORY_TO_BINARY


def run(llm_confidence_results, items, llm, plaus_valid, diff_df, **kwargs):
    print("\n" + "=" * 70)
    print("H. ITEM-LEVEL DIAGNOSTICS (4-point primary)")
    print("=" * 70)

    per_item = llm_confidence_results["per_item"]

    # ---- Identify 4-point disagreement items ----
    disagree_items_4pt = []
    disagree_items_bin = []

    for item_str, detail in per_item.items():
        item_num = int(item_str)
        for cond, cd in detail["conditions"].items():
            if cd["agreement_4pt"] == "DISAGREE":
                dd = diff_df[
                    (diff_df["item"] == item_num) & (diff_df["condition"] == cond)
                ]
                mean_diff = float(dd["rating"].mean()) if len(dd) > 0 else None

                hr = plaus_valid[
                    (plaus_valid["item"] == item_num) &
                    (plaus_valid["condition"] == cond)
                ]
                rating_spread = float(hr["rating"].std()) if len(hr) > 0 else None

                verb = items[item_num]["verb"] if item_num in items else "?"
                agent = (
                    items[item_num]["conditions"][cond]["agent"]
                    if item_num in items and cond in items[item_num]["conditions"]
                    else "?"
                )

                entry = {
                    "item": item_num,
                    "condition": cond,
                    "verb": verb,
                    "agent": agent,
                    "llm_category": cd["llm_category"],
                    "llm_rating": cd["llm_rating"],
                    "human_category": cd["human_category"],
                    "human_rating": cd["human_rating"],
                    "distance": cd["distance"],
                    "llm_verdict": cd["verdict"],
                    "mean_difficulty": mean_diff,
                    "rating_spread": rating_spread,
                    "also_binary_disagree": cd["agreement_bin"] == "DISAGREE",
                }
                disagree_items_4pt.append(entry)

                if cd["agreement_bin"] == "DISAGREE":
                    disagree_items_bin.append(entry)

    # ---- 4-point disagreements (primary) ----
    print(f"\n  === 4-POINT DISAGREEMENTS (primary) ===")
    print(f"  Total: {len(disagree_items_4pt)}")

    # By condition
    cond_counts = {}
    for d in disagree_items_4pt:
        cond_counts[d["condition"]] = cond_counts.get(d["condition"], 0) + 1
    print(f"  By condition: {cond_counts}")

    # By distance
    dist_counts = {}
    for d in disagree_items_4pt:
        dist_counts[d["distance"]] = dist_counts.get(d["distance"], 0) + 1
    print(f"  By distance: {dist_counts}")

    # By direction: LLM more plausible vs more implausible
    llm_more_plausible = sum(1 for d in disagree_items_4pt if d["llm_rating"] < d["human_rating"])
    llm_more_implausible = sum(1 for d in disagree_items_4pt if d["llm_rating"] > d["human_rating"])
    print(f"  Direction: LLM rates MORE plausible than human: {llm_more_plausible}")
    print(f"             LLM rates MORE implausible than human: {llm_more_implausible}")

    # By verdict type
    verdict_counts = {}
    for d in disagree_items_4pt:
        verdict_counts[d["llm_verdict"]] = verdict_counts.get(d["llm_verdict"], 0) + 1
    print(f"  By verdict type: {verdict_counts}")

    # Which also disagree at binary level
    n_also_binary = sum(1 for d in disagree_items_4pt if d["also_binary_disagree"])
    n_only_4pt = len(disagree_items_4pt) - n_also_binary
    print(f"\n  Of {len(disagree_items_4pt)} 4-point disagreements:")
    print(f"    Also disagree at binary level:     {n_also_binary} (cross-polarity errors)")
    print(f"    Agree at binary, disagree at 4-pt: {n_only_4pt} (within-polarity errors, e.g. easy vs hard)")

    # ---- Detail listing ----
    print(f"\n  Disagreement details (sorted by distance, then condition):")
    for d in sorted(disagree_items_4pt, key=lambda x: (-x["distance"], x["condition"], x["item"])):
        diff_str = f"diff={d['mean_difficulty']:.2f}" if d["mean_difficulty"] else "diff=?"
        spread_str = f"spread={d['rating_spread']:.2f}" if d["rating_spread"] else "spread=?"
        bin_flag = " [BINARY DISAGREE]" if d["also_binary_disagree"] else " [4pt only]"
        print(
            f"    Item {d['item']:2d} ({d['verb']:12s}) {d['condition']:20s}: "
            f"LLM={d['llm_category']:20s} Human={d['human_category']:20s} "
            f"dist={d['distance']} {d['llm_verdict']:10s} {diff_str} {spread_str} "
            f"agent={d['agent']}{bin_flag}"
        )

    # ---- Items with most disagreements ----
    item_disagree_counts = {}
    for d in disagree_items_4pt:
        item_disagree_counts[d["item"]] = item_disagree_counts.get(d["item"], 0) + 1
    multi_disagree = {k: v for k, v in item_disagree_counts.items() if v > 1}
    if multi_disagree:
        print(f"\n  Items with multiple 4-point disagreements:")
        for item_num, count in sorted(multi_disagree.items(), key=lambda x: -x[1]):
            verb = items[item_num]["verb"] if item_num in items else "?"
            print(f"    Item {item_num} ({verb}): {count} disagreements")

    # ---- Concentration ----
    unique_items = set(d["item"] for d in disagree_items_4pt)
    print(f"\n  {len(disagree_items_4pt)} disagreements across {len(unique_items)} unique items "
          f"(out of {N_ITEMS})")

    # ---- Binary-only summary (supplementary) ----
    print(f"\n  === BINARY DISAGREEMENTS (supplementary) ===")
    print(f"  Total: {n_also_binary} (subset of 4-point disagreements that cross the plausible/implausible boundary)")

    return {
        "item_diagnostics": {
            "fourpoint_disagreements": {
                "n_total": len(disagree_items_4pt),
                "by_condition": cond_counts,
                "by_distance": {str(k): v for k, v in dist_counts.items()},
                "by_direction": {
                    "llm_more_plausible": llm_more_plausible,
                    "llm_more_implausible": llm_more_implausible,
                },
                "by_verdict_type": verdict_counts,
                "n_also_binary_disagree": n_also_binary,
                "n_only_4pt_disagree": n_only_4pt,
                "n_unique_items": len(unique_items),
                "items_with_multiple": multi_disagree,
                "details": disagree_items_4pt,
            },
            "binary_disagreements_supplementary": {
                "n_total": n_also_binary,
                "details": disagree_items_bin,
            },
        },
    }
