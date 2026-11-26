"""
Section F: Visual Distinctiveness AAT — Decomposed.

F1: Agent Identifiability AAT (per-agent, n=120)
    Human: Yes/Maybe/No per agent
    LLM: IDENTIFIABLE/AMBIGUOUS/UNIDENTIFIABLE per agent
    Primary metric: 3-point unweighted kappa (classification)
    Supplementary: binary kappa (Yes vs Maybe+No)

F2: Pairwise Distinguishability AAT (per-pair, n=180)
    Human: checked pairs = indistinguishable, unchecked = distinguishable
    LLM: distinguishable true/false per pair from evaluation JSON
    Metric: binary kappa + AC1 (prevalence may be skewed)

F3: Combined Verdict (per-set, n=30)
    Derived holistic summary: ALL_DISTINCT / MOSTLY_DISTINCT / PROBLEMATIC
    Kept for backward compatibility but no longer the primary visual analysis.
"""

import json
import os
import numpy as np
import pandas as pd
from itertools import combinations
from ..config import (
    COND_ORDER, N_ITEMS, BOOTSTRAP_N, RANDOM_SEED, DELTAS,
    EVAL_CSV, evaluation_json_path,
)
from ..metrics import (
    weighted_cohens_kappa, cohens_kappa, gwets_ac1,
    percent_agreement, krippendorffs_alpha, summarize_bootstrap,
)


# ---- Condition pair key helpers ----
COND_ABBREV = {
    "plausible_easy": "pe", "plausible_hard": "ph",
    "implausible_hard": "ih", "implausible_easy": "ie",
}
PAIR_ORDER = ["pe", "ph", "ih", "ie"]
ALL_PAIR_KEYS = [
    f"{PAIR_ORDER[i]}_vs_{PAIR_ORDER[j]}"
    for i in range(4) for j in range(i+1, 4)
]  # pe_vs_ph, pe_vs_ih, pe_vs_ie, ph_vs_ih, ph_vs_ie, ih_vs_ie


def _load_llm_pairwise(items, eval_csv_path):
    """
    Load per-pair LLM judgments from evaluation JSON files.
    Returns dict: item_num -> {pair_key: True/False (distinguishable)}
    """
    # Read the merged CSV to find eval_run for each item
    evals_df = pd.read_csv(eval_csv_path, dtype=str)
    llm_pairs = {}

    for item_num, item in items.items():
        source_run = item["source_run"]
        verb = item["verb"]

        # Find matching eval row to confirm the eval_run
        match = evals_df[
            (evals_df["eval_run"].str.strip() == source_run) &
            (evals_df["verb"].str.strip() == verb)
        ]
        if len(match) == 0:
            continue

        eval_run = str(match.iloc[0]["eval_run"]).strip()

        # Try to load the JSON
        json_path = evaluation_json_path(eval_run, verb)
        if not json_path.exists():
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            ve = data.get("evaluation", data).get("visual_evaluation", {})
            pc = ve.get("pairwise_checks", {})

            pair_judgments = {}
            for pk in ALL_PAIR_KEYS:
                if pk in pc:
                    pair_judgments[pk] = bool(pc[pk].get("distinguishable", True))
                else:
                    pair_judgments[pk] = None

            llm_pairs[item_num] = pair_judgments
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return llm_pairs


def _parse_human_pairs(pair_df, items, annotator_ids):
    """
    Parse human pairwise responses into per-pair binary judgments.
    Returns dict: (annotator, item, pair_key) -> 0 (distinguishable) or 1 (indistinguishable)
    """
    human_pairs = {}

    for item_num in range(1, N_ITEMS + 1):
        if item_num not in items:
            continue

        # Build agent -> condition mapping for this item
        agent_to_cond = {}
        for cond, info in items[item_num]["conditions"].items():
            agent_name = info["agent"].strip().lower()
            agent_to_cond[agent_name] = cond

        for aid in annotator_ids:
            responses = pair_df[
                (pair_df["item"] == item_num) & (pair_df["annotator"] == aid)
            ]

            # Default: all pairs are distinguishable
            for pk in ALL_PAIR_KEYS:
                human_pairs[(aid, item_num, pk)] = 0  # distinguishable

            if len(responses) == 0:
                # No response = missing data
                for pk in ALL_PAIR_KEYS:
                    human_pairs[(aid, item_num, pk)] = None
                continue

            response_text = str(responses.iloc[0]["response"]).strip()

            if "none" in response_text.lower():
                # All pairs distinguishable (already set to 0)
                continue

            # Parse comma-separated pair selections
            selected_pairs = [p.strip() for p in response_text.split(",") if p.strip()]

            for pair_text in selected_pairs:
                # Parse "Agent A vs Agent B"
                parts = pair_text.split(" vs ")
                if len(parts) != 2:
                    continue

                a1_name = parts[0].strip().lower()
                a2_name = parts[1].strip().lower()

                # Find conditions for these agents
                c1 = agent_to_cond.get(a1_name)
                c2 = agent_to_cond.get(a2_name)

                if c1 is None or c2 is None:
                    # Try fuzzy matching (agent name might have slight variations)
                    for agent_key, cond_val in agent_to_cond.items():
                        if c1 is None and (a1_name in agent_key or agent_key in a1_name):
                            c1 = cond_val
                        if c2 is None and (a2_name in agent_key or agent_key in a2_name):
                            c2 = cond_val

                if c1 is None or c2 is None:
                    continue

                # Build pair key in canonical order
                abbr1 = COND_ABBREV[c1]
                abbr2 = COND_ABBREV[c2]
                if PAIR_ORDER.index(abbr1) <= PAIR_ORDER.index(abbr2):
                    pk = f"{abbr1}_vs_{abbr2}"
                else:
                    pk = f"{abbr2}_vs_{abbr1}"

                if pk in ALL_PAIR_KEYS:
                    human_pairs[(aid, item_num, pk)] = 1  # indistinguishable

    return human_pairs


def _derive_visual_verdict(ident_df_item, pair_df_item):
    """Original combined verdict derivation (kept for F3)."""
    all_yes = True
    all_yes_maybe = True
    for agent in ident_df_item["agent"].unique():
        ratings = ident_df_item[ident_df_item["agent"] == agent]["rating"]
        if len(ratings) > 0:
            majority = ratings.mode()
            if len(majority) > 0:
                if majority.iloc[0] != "Yes":
                    all_yes = False
                if majority.iloc[0] not in ("Yes", "Maybe"):
                    all_yes_maybe = False
            else:
                all_yes = all_yes_maybe = False
    n_resp = len(pair_df_item)
    none_majority = False
    if n_resp > 0:
        none_count = sum(
            1 for _, r in pair_df_item.iterrows()
            if "none" in str(r.get("response", "")).lower()
        )
        none_majority = none_count > n_resp / 2
    if none_majority and all_yes:
        return "ALL_DISTINCT"
    elif all_yes_maybe:
        return "MOSTLY_DISTINCT"
    return "PROBLEMATIC"


def run(llm, ident_df, pair_df, annotator_ids, item_list, items, **kwargs):
    print("\n" + "=" * 70)
    print("F. VISUAL DISTINCTIVENESS AAT (Decomposed)")
    print("  F1: Agent Identifiability (per-agent, n=120)")
    print("  F2: Pairwise Distinguishability (per-pair, n=180)")
    print("  F3: Combined Verdict (per-set, n=30, derived)")
    print("=" * 70)

    actual_anns = [a for a in annotator_ids if a in ident_df["annotator"].unique()]
    n_anns = len(actual_anns)

    # ================================================================
    # F1: AGENT IDENTIFIABILITY AAT
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  F1: AGENT IDENTIFIABILITY (per-agent, n=120)")
    print(f"  {'='*60}")

    IDENT_MAP = {"Yes": 2, "Maybe": 1, "No": 0}
    LLM_IDENT_MAP = {"IDENTIFIABLE": 2, "AMBIGUOUS": 1, "UNIDENTIFIABLE": 0}
    IDENT_BIN_MAP = {"Yes": 1, "Maybe": 0, "No": 0}
    LLM_IDENT_BIN_MAP = {"IDENTIFIABLE": 1, "AMBIGUOUS": 0, "UNIDENTIFIABLE": 0}

    # Build per-agent data: (item, condition) as rows
    ident_rows = []
    for i in item_list:
        for cond in COND_ORDER:
            if i not in items or cond not in items[i]["conditions"]:
                continue
            agent = items[i]["conditions"][cond]["agent"]

            # Human ratings
            h_ratings = {}
            for aid in actual_anns:
                ad = ident_df[
                    (ident_df["item"] == i) & (ident_df["annotator"] == aid) &
                    (ident_df["agent"] == agent)
                ]
                if len(ad) > 0:
                    h_ratings[aid] = ad.iloc[0]["rating"]

            # LLM rating from merged CSV columns
            llm_rating = None
            if i in llm:
                # Get from the scores/identifiability in the llm dict
                # The merged CSV has ident_{condition} columns
                pass

            ident_rows.append({
                "item": i, "condition": cond, "agent": agent,
                "human_ratings": h_ratings,
            })

    # Get LLM identifiability from merged CSV
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
            if val in LLM_IDENT_MAP:
                llm_ident[(i, cond)] = val

    # Build matrices for AAT
    # 3-point analysis
    ident_index = [(i, cond) for i in item_list for cond in COND_ORDER
                   if i in items and cond in items[i].get("conditions", {})]

    human_ident_matrix = np.full((len(ident_index), n_anns), np.nan)
    llm_ident_vec = np.full(len(ident_index), np.nan)
    llm_ident_bin_vec = np.full(len(ident_index), np.nan)
    human_ident_bin_matrix = np.full((len(ident_index), n_anns), np.nan)

    for idx, (i, cond) in enumerate(ident_index):
        agent = items[i]["conditions"][cond]["agent"]
        for a_idx, aid in enumerate(actual_anns):
            ad = ident_df[
                (ident_df["item"] == i) & (ident_df["annotator"] == aid) &
                (ident_df["agent"] == agent)
            ]
            if len(ad) > 0:
                rating = ad.iloc[0]["rating"]
                human_ident_matrix[idx, a_idx] = IDENT_MAP.get(rating, np.nan)
                human_ident_bin_matrix[idx, a_idx] = IDENT_BIN_MAP.get(rating, np.nan)

        lval = llm_ident.get((i, cond))
        if lval:
            llm_ident_vec[idx] = LLM_IDENT_MAP.get(lval, np.nan)
            llm_ident_bin_vec[idx] = LLM_IDENT_BIN_MAP.get(lval, np.nan)

    n_valid_ident = int(np.sum(~np.isnan(llm_ident_vec)))
    print(f"\n  Agents with LLM identifiability rating: {n_valid_ident}/{len(ident_index)}")

    # Distribution
    if len(ident_df) > 0:
        human_dist = {r: float((ident_df["rating"] == r).mean()) for r in ["Yes", "Maybe", "No"]}
        print(f"  Human identifiability distribution: {human_dist}")
    llm_dist = {}
    for val in ["IDENTIFIABLE", "AMBIGUOUS", "UNIDENTIFIABLE"]:
        llm_dist[val] = sum(1 for v in llm_ident.values() if v == val)
    print(f"  LLM identifiability distribution: {llm_dist}")

    # H-H 3-point kappa
    print(f"\n  --- F1a: 3-point unweighted kappa (primary) ---")
    hh_ident_kappas = {}
    for a1, a2 in combinations(range(n_anns), 2):
        k = cohens_kappa(human_ident_matrix[:, a1], human_ident_matrix[:, a2])
        pair_name = f"{actual_anns[a1]}-{actual_anns[a2]}"
        hh_ident_kappas[pair_name] = float(k) if not np.isnan(k) else np.nan
    hh_ident_mean = float(np.nanmean(list(hh_ident_kappas.values())))
    print(f"  H-H 3pt kappa: mean = {hh_ident_mean:.4f}")

    # L-H 3-point kappa
    lh_ident_kappas = {}
    for a_idx, aid in enumerate(actual_anns):
        k = cohens_kappa(llm_ident_vec, human_ident_matrix[:, a_idx])
        lh_ident_kappas[f"LLM-{aid}"] = float(k) if not np.isnan(k) else np.nan
        print(f"    LLM-{aid}: k={k:.4f}")
    lh_ident_mean = float(np.nanmean(list(lh_ident_kappas.values())))
    ident_delta = lh_ident_mean - hh_ident_mean
    print(f"  L-H 3pt kappa: mean = {lh_ident_mean:.4f}")
    print(f"  Delta: {ident_delta:+.4f}")

    # Bootstrap 3-point
    print(f"\n  Bootstrap (n={BOOTSTRAP_N})...")
    rng = np.random.RandomState(RANDOM_SEED + 200)
    all_idx = np.arange(len(ident_index))
    ident_boot = []
    for _ in range(BOOTSTRAP_N):
        sidx = rng.choice(all_idx, size=len(all_idx), replace=True)
        hh = [cohens_kappa(human_ident_matrix[sidx, i], human_ident_matrix[sidx, j])
              for i, j in combinations(range(n_anns), 2)]
        hh = [k for k in hh if not np.isnan(k)]
        lh = [cohens_kappa(llm_ident_vec[sidx], human_ident_matrix[sidx, i])
              for i in range(n_anns)]
        lh = [k for k in lh if not np.isnan(k)]
        if hh and lh:
            ident_boot.append(np.nanmean(lh) - np.nanmean(hh))
    ident_boot = np.array(ident_boot)
    ident_boot_results = summarize_bootstrap(ident_boot, "  F1 Identifiability (3pt)")

    # Binary supplementary
    print(f"\n  --- F1b: Binary kappa (Yes vs Maybe+No, supplementary) ---")
    hh_ident_bin_kappas = {}
    for a1, a2 in combinations(range(n_anns), 2):
        k = cohens_kappa(human_ident_bin_matrix[:, a1], human_ident_bin_matrix[:, a2])
        pair_name = f"{actual_anns[a1]}-{actual_anns[a2]}"
        hh_ident_bin_kappas[pair_name] = float(k) if not np.isnan(k) else np.nan
    hh_ident_bin_mean = float(np.nanmean(list(hh_ident_bin_kappas.values())))

    lh_ident_bin_kappas = {}
    for a_idx, aid in enumerate(actual_anns):
        k = cohens_kappa(llm_ident_bin_vec, human_ident_bin_matrix[:, a_idx])
        lh_ident_bin_kappas[f"LLM-{aid}"] = float(k) if not np.isnan(k) else np.nan
    lh_ident_bin_mean = float(np.nanmean(list(lh_ident_bin_kappas.values())))
    ident_bin_delta = lh_ident_bin_mean - hh_ident_bin_mean
    print(f"  H-H binary kappa: {hh_ident_bin_mean:.4f}")
    print(f"  L-H binary kappa: {lh_ident_bin_mean:.4f}")
    print(f"  Delta: {ident_bin_delta:+.4f}")

    # ================================================================
    # F2: PAIRWISE DISTINGUISHABILITY AAT
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  F2: PAIRWISE DISTINGUISHABILITY (per-pair, n=180)")
    print(f"  {'='*60}")

    # Load LLM pairwise from JSONs
    llm_pair_data = _load_llm_pairwise(items, EVAL_CSV)
    print(f"\n  Loaded LLM pairwise data for {len(llm_pair_data)} items")

    # Parse human pairwise
    human_pair_data = _parse_human_pairs(pair_df, items, actual_anns)

    # Build matrices: rows = (item, pair_key), columns = annotators
    pair_index = [(i, pk) for i in item_list for pk in ALL_PAIR_KEYS
                  if i in items]

    human_pair_matrix = np.full((len(pair_index), n_anns), np.nan)
    llm_pair_vec = np.full(len(pair_index), np.nan)

    for idx, (i, pk) in enumerate(pair_index):
        for a_idx, aid in enumerate(actual_anns):
            val = human_pair_data.get((aid, i, pk))
            if val is not None:
                human_pair_matrix[idx, a_idx] = float(val)

        if i in llm_pair_data and pk in llm_pair_data[i]:
            llm_val = llm_pair_data[i][pk]
            if llm_val is not None:
                # LLM: distinguishable=True -> 0 (no problem), distinguishable=False -> 1 (problem)
                llm_pair_vec[idx] = 0.0 if llm_val else 1.0

    n_valid_pairs = int(np.sum(~np.isnan(llm_pair_vec)))
    print(f"  Valid LLM pair judgments: {n_valid_pairs}/{len(pair_index)}")

    # Prevalence check
    human_indist_rate = float(np.nanmean(human_pair_matrix))
    llm_indist_rate = float(np.nanmean(llm_pair_vec[~np.isnan(llm_pair_vec)]))
    print(f"  Human indistinguishable rate: {human_indist_rate:.1%}")
    print(f"  LLM indistinguishable rate: {llm_indist_rate:.1%}")

    # H-H binary kappa
    print(f"\n  --- F2a: Binary kappa ---")
    hh_pair_kappas = {}
    for a1, a2 in combinations(range(n_anns), 2):
        k = cohens_kappa(human_pair_matrix[:, a1], human_pair_matrix[:, a2])
        pair_name = f"{actual_anns[a1]}-{actual_anns[a2]}"
        hh_pair_kappas[pair_name] = float(k) if not np.isnan(k) else np.nan
    hh_pair_mean = float(np.nanmean(list(hh_pair_kappas.values())))
    print(f"  H-H binary kappa: mean = {hh_pair_mean:.4f}")

    lh_pair_kappas = {}
    for a_idx, aid in enumerate(actual_anns):
        k = cohens_kappa(llm_pair_vec, human_pair_matrix[:, a_idx])
        lh_pair_kappas[f"LLM-{aid}"] = float(k) if not np.isnan(k) else np.nan
        print(f"    LLM-{aid}: k={k:.4f}")
    lh_pair_mean = float(np.nanmean(list(lh_pair_kappas.values())))
    pair_delta = lh_pair_mean - hh_pair_mean
    print(f"  L-H binary kappa: mean = {lh_pair_mean:.4f}")
    print(f"  Delta: {pair_delta:+.4f}")

    # AC1 (for prevalence imbalance)
    print(f"\n  --- F2b: Gwet's AC1 (handles prevalence imbalance) ---")
    hh_pair_ac1s = {}
    for a1, a2 in combinations(range(n_anns), 2):
        k = gwets_ac1(human_pair_matrix[:, a1], human_pair_matrix[:, a2])
        pair_name = f"{actual_anns[a1]}-{actual_anns[a2]}"
        hh_pair_ac1s[pair_name] = float(k) if not np.isnan(k) else np.nan
    hh_pair_ac1_mean = float(np.nanmean(list(hh_pair_ac1s.values())))

    lh_pair_ac1s = {}
    for a_idx, aid in enumerate(actual_anns):
        k = gwets_ac1(llm_pair_vec, human_pair_matrix[:, a_idx])
        lh_pair_ac1s[f"LLM-{aid}"] = float(k) if not np.isnan(k) else np.nan
    lh_pair_ac1_mean = float(np.nanmean(list(lh_pair_ac1s.values())))
    pair_ac1_delta = lh_pair_ac1_mean - hh_pair_ac1_mean
    print(f"  H-H AC1: {hh_pair_ac1_mean:.4f}")
    print(f"  L-H AC1: {lh_pair_ac1_mean:.4f}")
    print(f"  Delta: {pair_ac1_delta:+.4f}")

    # Bootstrap pairwise
    print(f"\n  Bootstrap (n={BOOTSTRAP_N})...")
    rng2 = np.random.RandomState(RANDOM_SEED + 300)
    pair_idx_arr = np.arange(len(pair_index))
    pair_boot_kappa = []
    pair_boot_ac1 = []
    for _ in range(BOOTSTRAP_N):
        sidx = rng2.choice(pair_idx_arr, size=len(pair_idx_arr), replace=True)
        hh_k = [cohens_kappa(human_pair_matrix[sidx, i], human_pair_matrix[sidx, j])
                 for i, j in combinations(range(n_anns), 2)]
        hh_k = [k for k in hh_k if not np.isnan(k)]
        lh_k = [cohens_kappa(llm_pair_vec[sidx], human_pair_matrix[sidx, i])
                 for i in range(n_anns)]
        lh_k = [k for k in lh_k if not np.isnan(k)]
        if hh_k and lh_k:
            pair_boot_kappa.append(np.nanmean(lh_k) - np.nanmean(hh_k))

        hh_a = [gwets_ac1(human_pair_matrix[sidx, i], human_pair_matrix[sidx, j])
                 for i, j in combinations(range(n_anns), 2)]
        hh_a = [k for k in hh_a if not np.isnan(k)]
        lh_a = [gwets_ac1(llm_pair_vec[sidx], human_pair_matrix[sidx, i])
                 for i in range(n_anns)]
        lh_a = [k for k in lh_a if not np.isnan(k)]
        if hh_a and lh_a:
            pair_boot_ac1.append(np.nanmean(lh_a) - np.nanmean(hh_a))

    pair_boot_kappa = np.array(pair_boot_kappa)
    pair_boot_ac1 = np.array(pair_boot_ac1)

    pair_kappa_boot_results = summarize_bootstrap(pair_boot_kappa, "  F2 Pairwise (kappa)") if len(pair_boot_kappa) > 0 else {}
    pair_ac1_boot_results = summarize_bootstrap(pair_boot_ac1, "  F2 Pairwise (AC1)") if len(pair_boot_ac1) > 0 else {}

    # ================================================================
    # F3: COMBINED VERDICT (per-set, backward compatible)
    # ================================================================
    print(f"\n  {'='*60}")
    print(f"  F3: COMBINED VERDICT (per-set, n={N_ITEMS}, derived)")
    print(f"  {'='*60}")

    VIS_NUM = {"ALL_DISTINCT": 2, "MOSTLY_DISTINCT": 1, "PROBLEMATIC": 0}

    ann_visual = {}
    for aid in actual_anns:
        ann_visual[aid] = {}
        for i in item_list:
            ai = ident_df[(ident_df["item"] == i) & (ident_df["annotator"] == aid)]
            ap = (pair_df[(pair_df["item"] == i) & (pair_df["annotator"] == aid)]
                  if len(pair_df) > 0 else pd.DataFrame())
            ann_visual[aid][i] = _derive_visual_verdict(ai, ap)

    # H-H combined
    vis_hh_kappas = {}
    for a1, a2 in combinations(actual_anns, 2):
        y1 = np.array([VIS_NUM.get(ann_visual[a1][i], np.nan) for i in item_list], dtype=float)
        y2 = np.array([VIS_NUM.get(ann_visual[a2][i], np.nan) for i in item_list], dtype=float)
        k = weighted_cohens_kappa(y1, y2, n_categories=3)
        vis_hh_kappas[f"{a1}-{a2}"] = float(k)
    vis_hh_mean = float(np.nanmean(list(vis_hh_kappas.values())))
    print(f"\n  H-H combined weighted kappa: {vis_hh_mean:.4f}")

    # L-H combined
    llm_vis_vec = np.array([
        VIS_NUM.get(llm[i]["visual_verdict"], np.nan) if i in llm else np.nan
        for i in item_list
    ], dtype=float)
    vis_lh_kappas = {}
    for aid in actual_anns:
        y_h = np.array([VIS_NUM.get(ann_visual[aid][i], np.nan) for i in item_list], dtype=float)
        k = weighted_cohens_kappa(llm_vis_vec, y_h, n_categories=3)
        vis_lh_kappas[f"LLM-{aid}"] = float(k)
    vis_lh_mean = float(np.nanmean(list(vis_lh_kappas.values())))
    vis_delta = vis_lh_mean - vis_hh_mean
    print(f"  L-H combined weighted kappa: {vis_lh_mean:.4f}")
    print(f"  Delta: {vis_delta:+.4f}")

    # ---- Summary ----
    print(f"\n  {'='*60}")
    print(f"  VISUAL AAT SUMMARY")
    print(f"  {'='*60}")
    print(f"  F1 Identifiability (3pt):  HH={hh_ident_mean:.4f} LH={lh_ident_mean:.4f} "
          f"delta={ident_delta:+.4f}")
    print(f"  F1 Identifiability (bin):  HH={hh_ident_bin_mean:.4f} LH={lh_ident_bin_mean:.4f} "
          f"delta={ident_bin_delta:+.4f}")
    print(f"  F2 Pairwise (kappa):       HH={hh_pair_mean:.4f} LH={lh_pair_mean:.4f} "
          f"delta={pair_delta:+.4f}")
    print(f"  F2 Pairwise (AC1):         HH={hh_pair_ac1_mean:.4f} LH={lh_pair_ac1_mean:.4f} "
          f"delta={pair_ac1_delta:+.4f}")
    print(f"  F3 Combined (wk):          HH={vis_hh_mean:.4f} LH={vis_lh_mean:.4f} "
          f"delta={vis_delta:+.4f}")

    return {
        "visual_distinctiveness": {
            "f1_identifiability": {
                "n_agents": len(ident_index),
                "threepoint": {
                    "hh_kappas": hh_ident_kappas, "hh_mean": hh_ident_mean,
                    "lh_kappas": lh_ident_kappas, "lh_mean": lh_ident_mean,
                    "delta": ident_delta,
                    "bootstrap": ident_boot_results,
                },
                "binary": {
                    "hh_mean": hh_ident_bin_mean, "lh_mean": lh_ident_bin_mean,
                    "delta": ident_bin_delta,
                },
                "distributions": {
                    "human": human_dist if len(ident_df) > 0 else {},
                    "llm": llm_dist,
                },
            },
            "f2_pairwise": {
                "n_pairs": len(pair_index),
                "prevalence": {
                    "human_indistinguishable_rate": human_indist_rate,
                    "llm_indistinguishable_rate": llm_indist_rate,
                },
                "kappa": {
                    "hh_kappas": hh_pair_kappas, "hh_mean": hh_pair_mean,
                    "lh_kappas": lh_pair_kappas, "lh_mean": lh_pair_mean,
                    "delta": pair_delta,
                    "bootstrap": pair_kappa_boot_results,
                },
                "ac1": {
                    "hh_mean": hh_pair_ac1_mean, "lh_mean": lh_pair_ac1_mean,
                    "delta": pair_ac1_delta,
                    "bootstrap": pair_ac1_boot_results,
                },
            },
            "f3_combined": {
                "hh_kappas": vis_hh_kappas, "hh_mean": vis_hh_mean,
                "lh_kappas": vis_lh_kappas, "lh_mean": vis_lh_mean,
                "delta": vis_delta,
            },
        },
    }
