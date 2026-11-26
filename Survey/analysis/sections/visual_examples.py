"""
Section J: Visual examples export for qualitative review.

Generates two CSVs:
1. visual_identifiability_examples.csv
   Per-agent: each annotator's Yes/Maybe/No, LLM rating, agreement status
   Sorted by human disagreement (most contested agents first)

2. visual_pairwise_examples.csv
   Per-pair: each annotator's flag, LLM judgment, human flagging rate
   Sorted by human flagging rate (most-flagged pairs first)
"""

import json
import os
import numpy as np
import pandas as pd
from ..config import (
    COND_ORDER, N_ITEMS, EVAL_CSV, QUALTRICS_CSV,
    evaluation_json_path,
    visual_identifiability_examples_path,
    visual_pairwise_examples_path,
)


COND_ABBREV = {
    "plausible_easy": "pe", "plausible_hard": "ph",
    "implausible_hard": "ih", "implausible_easy": "ie",
}
PAIR_ORDER = ["pe", "ph", "ih", "ie"]
ALL_PAIR_KEYS = [
    f"{PAIR_ORDER[i]}_vs_{PAIR_ORDER[j]}"
    for i in range(4) for j in range(i+1, 4)
]

LLM_TO_HUMAN = {"IDENTIFIABLE": "Yes", "AMBIGUOUS": "Maybe", "UNIDENTIFIABLE": "No"}


def _load_followup_reasons(n_items=30):
    """Load follow-up text reasons from Qualtrics embedded data columns."""
    import pandas as pd
    try:
        raw = pd.read_csv(QUALTRICS_CSV, header=None, dtype=str)
        # Find data start row
        data_start = 1
        for i in range(1, min(5, len(raw))):
            first_val = str(raw.iloc[i, 0]) if pd.notna(raw.iloc[i, 0]) else ""
            if first_val.startswith("{") or "plausible" in first_val.lower() or "ImportId" in first_val:
                data_start = i + 1
            elif first_val.startswith("20"):
                data_start = i
                break
        df = raw.iloc[data_start:].copy()
        df.columns = raw.iloc[0]

        # Collect per-item: all unique non-empty follow-up reasons
        item_reasons = {}
        for item_num in range(1, n_items + 1):
            reasons = set()
            for choice in range(1, 7):
                col = f"Item{item_num}_FollowUp_{choice}"
                if col in df.columns:
                    for val in df[col]:
                        if pd.notna(val) and str(val).strip():
                            reasons.add(str(val).strip())
            item_reasons[item_num] = sorted(reasons) if reasons else []
        return item_reasons
    except Exception as e:
        print(f"  Warning: Could not load follow-up reasons: {e}")
        return {}


def run(ident_df, pair_df, items, llm, annotator_ids, item_list, eval_prefix="evalv3", **kwargs):
    print("\n" + "=" * 70)
    print("J. VISUAL EXAMPLES EXPORT")
    print("=" * 70)

    actual_anns = sorted([a for a in annotator_ids if a in ident_df["annotator"].unique()])

    # Load follow-up reasons from Qualtrics
    followup_reasons = _load_followup_reasons(N_ITEMS)

    # Load LLM identifiability from merged CSV
    evals_df = pd.read_csv(EVAL_CSV, dtype=str)
    llm_ident = {}
    for i in item_list:
        if i not in items:
            continue
        match = evals_df[
            (evals_df["eval_run"].str.strip() == items[i]["source_run"]) &
            (evals_df["verb"].str.strip() == items[i]["verb"])
        ]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        for cond in COND_ORDER:
            col = f"ident_{cond}"
            val = str(row.get(col, "")).strip().upper() if pd.notna(row.get(col)) else ""
            if val:
                llm_ident[(i, cond)] = val

    # Load LLM pairwise from JSONs
    llm_pairs = {}
    for i in item_list:
        if i not in items:
            continue
        source_run = items[i]["source_run"]
        if eval_prefix != "evalv3" and source_run.startswith("evalv3_"):
            source_run = eval_prefix + "_" + source_run[len("evalv3_"):]
        verb = items[i]["verb"]
        json_path = evaluation_json_path(source_run, verb)
        if not json_path.exists():
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
            ve = data.get("evaluation", data).get("visual_evaluation", {})
            pc = ve.get("pairwise_checks", {})
            llm_pairs[i] = {
                pk: bool(pc[pk].get("distinguishable", True))
                for pk in ALL_PAIR_KEYS if pk in pc
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    # ================================================================
    # CSV 1: IDENTIFIABILITY
    # ================================================================
    ident_rows = []
    for i in item_list:
        if i not in items:
            continue
        for cond in COND_ORDER:
            if cond not in items[i]["conditions"]:
                continue
            agent = items[i]["conditions"][cond]["agent"]
            verb = items[i]["verb"]

            row = {
                "item": i, "verb": verb, "condition": cond, "agent": agent,
            }

            # Each annotator's rating
            ratings = {}
            for aid in actual_anns:
                ad = ident_df[
                    (ident_df["item"] == i) & (ident_df["annotator"] == aid) &
                    (ident_df["agent"] == agent)
                ]
                if len(ad) > 0:
                    ratings[aid] = ad.iloc[0]["rating"]
                    row[aid] = ad.iloc[0]["rating"]
                else:
                    row[aid] = ""

            # Distribution
            vals = list(ratings.values())
            yes_count = vals.count("Yes")
            maybe_count = vals.count("Maybe")
            no_count = vals.count("No")
            total = len(vals)
            row["n_Yes"] = yes_count
            row["n_Maybe"] = maybe_count
            row["n_No"] = no_count
            row["human_majority"] = max(["Yes", "Maybe", "No"],
                                         key=lambda x: vals.count(x)) if vals else ""

            # Disagreement measure: entropy-like
            if total > 0:
                proportions = [yes_count/total, maybe_count/total, no_count/total]
                row["agreement_rate"] = round(max(proportions), 2)
                row["n_categories_used"] = sum(1 for p in proportions if p > 0)
            else:
                row["agreement_rate"] = ""
                row["n_categories_used"] = ""

            # LLM
            lval = llm_ident.get((i, cond), "")
            row["llm_rating"] = lval
            row["llm_rating_mapped"] = LLM_TO_HUMAN.get(lval, "")

            # Agreement
            human_maj = row["human_majority"]
            llm_mapped = row["llm_rating_mapped"]
            if human_maj and llm_mapped:
                row["llm_human_agree"] = "AGREE" if human_maj == llm_mapped else "DISAGREE"
                row["llm_human_direction"] = ""
                if human_maj != llm_mapped:
                    order = ["No", "Maybe", "Yes"]
                    if order.index(llm_mapped) > order.index(human_maj):
                        row["llm_human_direction"] = "LLM_MORE_OPTIMISTIC"
                    else:
                        row["llm_human_direction"] = "LLM_MORE_PESSIMISTIC"
            else:
                row["llm_human_agree"] = ""
                row["llm_human_direction"] = ""

            ident_rows.append(row)

    ident_df_out = pd.DataFrame(ident_rows)
    # Sort by agreement rate ascending (most disagreement first), then by n_No descending
    ident_df_out = ident_df_out.sort_values(
        ["agreement_rate", "n_No"], ascending=[True, False]
    ).reset_index(drop=True)

    ident_path = visual_identifiability_examples_path(eval_prefix)
    ident_df_out.to_csv(ident_path, index=False)
    print(f"\n  Saved {len(ident_df_out)} agent identifiability records to {ident_path}")

    # Print top disagreements
    disagrees = ident_df_out[ident_df_out["llm_human_agree"] == "DISAGREE"]
    print(f"\n  --- LLM-Human Identifiability Disagreements ({len(disagrees)} total) ---")
    for _, r in disagrees.head(15).iterrows():
        ann_str = " ".join(f"{r.get(a, '?')}" for a in actual_anns)
        print(f"    Item {r['item']:2d} {r['condition']:20s} [{ann_str}] "
              f"majority={r['human_majority']:5s} LLM={r['llm_rating_mapped']:5s} "
              f"({r['llm_human_direction']}) agent={r['agent']}")

    # Most problematic agents (No-heavy)
    print(f"\n  --- Agents with highest 'No' rate (least identifiable per humans) ---")
    no_heavy = ident_df_out[ident_df_out["n_No"] >= 2].sort_values("n_No", ascending=False)
    for _, r in no_heavy.head(10).iterrows():
        print(f"    Item {r['item']:2d} {r['condition']:20s} "
              f"Yes={r['n_Yes']} Maybe={r['n_Maybe']} No={r['n_No']} "
              f"LLM={r['llm_rating_mapped']:5s} agent={r['agent']}")

    # ================================================================
    # CSV 2: PAIRWISE DISTINGUISHABILITY
    # ================================================================
    pair_rows = []

    for i in item_list:
        if i not in items:
            continue
        verb = items[i]["verb"]

        # Build agent lookup for this item
        cond_to_agent = {cond: items[i]["conditions"][cond]["agent"]
                         for cond in COND_ORDER if cond in items[i]["conditions"]}
        agent_to_cond = {v.strip().lower(): k for k, v in cond_to_agent.items()}

        for pk in ALL_PAIR_KEYS:
            # Decode pair key to agent names
            parts = pk.split("_vs_")
            abbrev_to_cond = {v: k for k, v in COND_ABBREV.items()}
            c1 = abbrev_to_cond.get(parts[0], "")
            c2 = abbrev_to_cond.get(parts[1], "")
            a1 = cond_to_agent.get(c1, "?")
            a2 = cond_to_agent.get(c2, "?")

            row = {
                "item": i, "verb": verb, "pair_key": pk,
                "condition_1": c1, "agent_1": a1,
                "condition_2": c2, "agent_2": a2,
                "pair_label": f"{a1} vs {a2}",
            }

            # Parse each annotator's response for this pair
            flagged_count = 0
            total_responses = 0
            for aid in actual_anns:
                responses = pair_df[
                    (pair_df["item"] == i) & (pair_df["annotator"] == aid)
                ]
                if len(responses) == 0:
                    row[aid] = ""
                    continue

                response_text = str(responses.iloc[0]["response"]).strip()
                total_responses += 1

                if "none" in response_text.lower():
                    row[aid] = 0  # all distinguishable
                    continue

                # Check if this pair was flagged
                selected = [p.strip().lower() for p in response_text.split(",")]
                pair_flagged = False
                for sel in selected:
                    sel_parts = sel.split(" vs ")
                    if len(sel_parts) == 2:
                        s1, s2 = sel_parts[0].strip(), sel_parts[1].strip()
                        sc1 = agent_to_cond.get(s1)
                        sc2 = agent_to_cond.get(s2)
                        if sc1 is None:
                            for ak, cv in agent_to_cond.items():
                                if s1 in ak or ak in s1:
                                    sc1 = cv
                                    break
                        if sc2 is None:
                            for ak, cv in agent_to_cond.items():
                                if s2 in ak or ak in s2:
                                    sc2 = cv
                                    break
                        if sc1 and sc2:
                            ab1 = COND_ABBREV.get(sc1, "")
                            ab2 = COND_ABBREV.get(sc2, "")
                            if PAIR_ORDER.index(ab1) <= PAIR_ORDER.index(ab2):
                                check_pk = f"{ab1}_vs_{ab2}"
                            else:
                                check_pk = f"{ab2}_vs_{ab1}"
                            if check_pk == pk:
                                pair_flagged = True
                                break

                row[aid] = 1 if pair_flagged else 0
                if pair_flagged:
                    flagged_count += 1

            row["n_flagged"] = flagged_count
            row["n_responded"] = total_responses
            row["human_flag_rate"] = round(flagged_count / total_responses, 2) if total_responses > 0 else ""

            # LLM
            llm_dist = None
            if i in llm_pairs and pk in llm_pairs[i]:
                llm_dist = llm_pairs[i][pk]
            row["llm_distinguishable"] = "" if llm_dist is None else ("Yes" if llm_dist else "No")
            row["llm_flagged"] = "" if llm_dist is None else (0 if llm_dist else 1)

            # Agreement
            human_majority_flag = 1 if flagged_count > total_responses / 2 else 0
            row["human_majority_distinguishable"] = "No" if human_majority_flag else "Yes"
            row["human_majority_flag"] = human_majority_flag
            # Add aggregated follow-up reasons for this item
            item_followups = followup_reasons.get(i, [])
            row["human_reasons_why_hard"] = " | ".join(item_followups) if item_followups else ""
            if llm_dist is not None:
                llm_flag = 0 if llm_dist else 1
                row["agree"] = "AGREE" if llm_flag == human_majority_flag else "DISAGREE"
                if llm_flag != human_majority_flag:
                    if llm_flag == 0 and human_majority_flag == 1:
                        row["error_type"] = "LLM_MISSED (humans flag, LLM says fine)"
                    else:
                        row["error_type"] = "LLM_OVERFLAG (LLM flags, humans say fine)"
                else:
                    row["error_type"] = ""
            else:
                row["agree"] = ""
                row["error_type"] = ""

            pair_rows.append(row)

    pair_df_out = pd.DataFrame(pair_rows)
    pair_df_out = pair_df_out.sort_values(
        ["human_flag_rate", "n_flagged"], ascending=[False, False]
    ).reset_index(drop=True)

    pair_path = visual_pairwise_examples_path(eval_prefix)
    pair_df_out.to_csv(pair_path, index=False)
    print(f"\n  Saved {len(pair_df_out)} pair records to {pair_path}")

    # Most-flagged pairs
    flagged_pairs = pair_df_out[pair_df_out["n_flagged"] >= 2].copy()
    print(f"\n  --- Most-Flagged Pairs (flagged by 2+ annotators, n={len(flagged_pairs)}) ---")
    for _, r in flagged_pairs.head(20).iterrows():
        ann_str = " ".join(str(r.get(a, "?")) for a in actual_anns)
        print(f"    Item {r['item']:2d} [{ann_str}] rate={r['human_flag_rate']:.0%} "
              f"LLM={r['llm_distinguishable']:3s} ({r.get('agree', ''):8s}) "
              f"{r['pair_label']}")

    # LLM misses (humans flag, LLM says fine)
    misses = pair_df_out[pair_df_out["error_type"].str.contains("MISSED", na=False)]
    print(f"\n  --- LLM Misses ({len(misses)} pairs where humans flag but LLM says fine) ---")
    for _, r in misses.head(15).iterrows():
        print(f"    Item {r['item']:2d} flag_rate={r['human_flag_rate']:.0%} "
              f"({r['n_flagged']}/{r['n_responded']}) "
              f"{r['pair_label']}")

    # Summary
    n_agree = (pair_df_out["agree"] == "AGREE").sum()
    n_disagree = (pair_df_out["agree"] == "DISAGREE").sum()
    n_missed = len(misses)
    n_overflag = (pair_df_out["error_type"].str.contains("OVERFLAG", na=False)).sum()
    print(f"\n  --- Summary ---")
    print(f"  Total pairs: {len(pair_df_out)}")
    print(f"  LLM-Human agree: {n_agree}, disagree: {n_disagree}")
    print(f"  Error breakdown: {n_missed} LLM misses, {n_overflag} LLM overflags")

    return {
        "visual_examples": {
            "identifiability_csv": str(ident_path),
            "pairwise_csv": str(pair_path),
            "n_ident_disagrees": int(len(disagrees)),
            "n_pair_misses": int(n_missed),
        },
    }