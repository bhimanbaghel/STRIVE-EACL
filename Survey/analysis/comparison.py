"""
Comparison: side-by-side analysis of multiple evaluator versions.

Usage (from project root):
    python -m analysis.comparison --prefixes evalv3 evalv4
    python -m analysis.comparison --prefixes evalv3 evalv4 evalv4_qwen4b

Prerequisites: run `python -m analysis.main --eval-prefix <prefix>` for each
prefix first, so that sentence_examples_{prefix}.csv files exist.

Outputs:
    - sentence_comparison_{p1}_vs_{p2}.csv  (merged per-sentence comparison)
    - Printed summary of improvements, regressions, scene gate firings
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from .config import COND_ORDER, sentence_examples_path, results_json_path, sentence_comparison_path


def load_sentence_examples(prefix):
    """Load sentence_examples_{prefix}.csv."""
    path = sentence_examples_path(prefix)
    if not path.exists():
        print(f"ERROR: {path} not found. Run: python -m analysis.main --eval-prefix {prefix}")
        sys.exit(1)
    return pd.read_csv(path)


def load_analysis_results(prefix):
    """Load analysis_results_{prefix}.json."""
    path = results_json_path(prefix)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def merge_sentence_examples(df_base, df_new, base_prefix, new_prefix):
    """
    Merge two sentence_examples DataFrames on (item, condition_generator).

    Keeps human columns from base (they're identical), adds LLM columns
    from both with prefix suffixes.
    """
    # Columns that are the same across evaluator versions (human + stimulus metadata)
    shared_cols = [
        "item", "condition_generator", "verb", "agent", "sentence",
        "tier", "model",
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
        "rating_distribution", "human_majority_rating",
        "human_majority_abbrev", "human_majority_category",
        "human_binary", "rating_mean", "rating_SD", "mean_difficulty",
    ]

    # LLM columns that differ between versions
    llm_cols = [
        "llm_category", "llm_rating", "llm_binary", "llm_verdict",
        "4pt_match", "binary_match", "distance", "disagree_type",
    ]

    # Filter to existing columns
    shared_cols = [c for c in shared_cols if c in df_base.columns]
    llm_cols = [c for c in llm_cols if c in df_base.columns]

    # Rename LLM columns with prefix
    base_renamed = df_base[shared_cols + llm_cols].copy()
    for col in llm_cols:
        base_renamed = base_renamed.rename(columns={col: f"{col}_{base_prefix}"})

    new_renamed = df_new[["item", "condition_generator"] + llm_cols].copy()
    for col in llm_cols:
        new_renamed = new_renamed.rename(columns={col: f"{col}_{new_prefix}"})

    # Merge on item + condition
    merged = base_renamed.merge(
        new_renamed, on=["item", "condition_generator"], how="outer",
    )

    # Add comparison columns
    m4_base = f"4pt_match_{base_prefix}"
    m4_new = f"4pt_match_{new_prefix}"
    mb_base = f"binary_match_{base_prefix}"
    mb_new = f"binary_match_{new_prefix}"

    if m4_base in merged.columns and m4_new in merged.columns:
        merged["v4_improved_4pt"] = (~merged[m4_base].astype(bool)) & (merged[m4_new].astype(bool))
        merged["v4_regressed_4pt"] = (merged[m4_base].astype(bool)) & (~merged[m4_new].astype(bool))
        merged["v4_changed_4pt"] = merged[m4_base].astype(bool) != merged[m4_new].astype(bool)

    if mb_base in merged.columns and mb_new in merged.columns:
        merged["v4_improved_binary"] = (~merged[mb_base].astype(bool)) & (merged[mb_new].astype(bool))
        merged["v4_regressed_binary"] = (merged[mb_base].astype(bool)) & (~merged[mb_new].astype(bool))

    return merged


def print_comparison_summary(merged, base_prefix, new_prefix):
    """Print a summary of changes between evaluator versions."""
    n = len(merged)
    m4_base = f"4pt_match_{base_prefix}"
    m4_new = f"4pt_match_{new_prefix}"
    mb_base = f"binary_match_{base_prefix}"
    mb_new = f"binary_match_{new_prefix}"
    cat_base = f"llm_category_{base_prefix}"
    cat_new = f"llm_category_{new_prefix}"
    verdict_base = f"llm_verdict_{base_prefix}"
    verdict_new = f"llm_verdict_{new_prefix}"
    dt_base = f"disagree_type_{base_prefix}"
    dt_new = f"disagree_type_{new_prefix}"

    print(f"\n{'='*70}")
    print(f"EVALUATOR COMPARISON: {base_prefix} vs {new_prefix}")
    print(f"{'='*70}")

    # Overall match rates
    if m4_base in merged.columns and m4_new in merged.columns:
        base_4pt = merged[m4_base].astype(bool).sum()
        new_4pt = merged[m4_new].astype(bool).sum()
        print(f"\n  4-POINT MATCH WITH HUMAN MAJORITY:")
        print(f"    {base_prefix}: {base_4pt}/{n} ({base_4pt/n:.1%})")
        print(f"    {new_prefix}:  {new_4pt}/{n} ({new_4pt/n:.1%})")
        print(f"    Change:  {new_4pt - base_4pt:+d}")

    if mb_base in merged.columns and mb_new in merged.columns:
        base_bin = merged[mb_base].astype(bool).sum()
        new_bin = merged[mb_new].astype(bool).sum()
        print(f"\n  BINARY MATCH WITH HUMAN MAJORITY:")
        print(f"    {base_prefix}: {base_bin}/{n} ({base_bin/n:.1%})")
        print(f"    {new_prefix}:  {new_bin}/{n} ({new_bin/n:.1%})")
        print(f"    Change:  {new_bin - base_bin:+d}")

    # Per-condition breakdown
    print(f"\n  PER-CONDITION 4PT MATCH:")
    for cond in COND_ORDER:
        cd = merged[merged["condition_generator"] == cond]
        if len(cd) == 0:
            continue
        b = cd[m4_base].astype(bool).sum() if m4_base in cd.columns else 0
        nw = cd[m4_new].astype(bool).sum() if m4_new in cd.columns else 0
        print(f"    {cond:20s}: {base_prefix}={b}/{len(cd)}  {new_prefix}={nw}/{len(cd)}  change={nw-b:+d}")

    # Improvements
    if "v4_improved_4pt" in merged.columns:
        improved = merged[merged["v4_improved_4pt"]]
        if len(improved) > 0:
            print(f"\n  IMPROVEMENTS ({new_prefix} now matches human, {base_prefix} didn't):")
            for _, r in improved.iterrows():
                print(f"    Item {r['item']:2d} {r['condition_generator']:20s} "
                      f"{base_prefix}={r.get(cat_base, '?'):20s} -> {new_prefix}={r.get(cat_new, '?'):20s} "
                      f"human={r.get('human_majority_category', '?')}")
        else:
            print(f"\n  IMPROVEMENTS: None")

    # Regressions
    if "v4_regressed_4pt" in merged.columns:
        regressed = merged[merged["v4_regressed_4pt"]]
        if len(regressed) > 0:
            print(f"\n  REGRESSIONS ({base_prefix} matched human, {new_prefix} doesn't):")
            for _, r in regressed.iterrows():
                print(f"    Item {r['item']:2d} {r['condition_generator']:20s} "
                      f"{base_prefix}={r.get(cat_base, '?'):20s} -> {new_prefix}={r.get(cat_new, '?'):20s} "
                      f"human={r.get('human_majority_category', '?')}")
        else:
            print(f"\n  REGRESSIONS: None")

    # Category changes (where LLM changed its mind regardless of human)
    if cat_base in merged.columns and cat_new in merged.columns:
        changed = merged[merged[cat_base] != merged[cat_new]]
        print(f"\n  LLM CATEGORY CHANGES: {len(changed)}/{n} sentences")
        if len(changed) > 0:
            print(f"    {'Item':>4s}  {'Condition':<20s}  {base_prefix:<20s}  {new_prefix:<20s}  {'Human':<20s}  {'Verdict'}")
            for _, r in changed.iterrows():
                # Did the change improve or regress?
                imp = "IMPROVED" if r.get("v4_improved_4pt", False) else (
                    "REGRESSED" if r.get("v4_regressed_4pt", False) else "LATERAL")
                print(f"    {r['item']:4.0f}  {r['condition_generator']:<20s}  "
                      f"{str(r.get(cat_base, '?')):<20s}  {str(r.get(cat_new, '?')):<20s}  "
                      f"{str(r.get('human_majority_category', '?')):<20s}  {imp}")

    # Verdict changes
    if verdict_base in merged.columns and verdict_new in merged.columns:
        v_changed = merged[merged[verdict_base] != merged[verdict_new]]
        print(f"\n  VERDICT CHANGES: {len(v_changed)}/{n}")
        if len(v_changed) > 0:
            for _, r in v_changed.iterrows():
                print(f"    Item {r['item']:2.0f} {r['condition_generator']:20s}: "
                      f"{r.get(verdict_base, '?')} -> {r.get(verdict_new, '?')}")

    # Scene gate (SKIPPED verdicts in new)
    if verdict_new in merged.columns:
        skipped = merged[merged[verdict_new] == "SKIPPED"]
        if len(skipped) > 0:
            print(f"\n  SCENE TYPICALITY GATE (SKIPPED in {new_prefix}): {len(skipped)} sentences")
            for _, r in skipped.iterrows():
                print(f"    Item {r['item']:2.0f} {r['condition_generator']:20s}: "
                      f"scene failed typicality check")
        else:
            print(f"\n  SCENE TYPICALITY GATE: No items triggered early-exit")

    # Cross-polarity changes
    if dt_base in merged.columns and dt_new in merged.columns:
        cross_base = (merged[dt_base] == "CROSS_POLARITY").sum()
        cross_new = (merged[dt_new] == "CROSS_POLARITY").sum()
        print(f"\n  CROSS-POLARITY ERRORS (most serious):")
        print(f"    {base_prefix}: {cross_base}")
        print(f"    {new_prefix}:  {cross_new}")
        print(f"    Change:  {cross_new - cross_base:+d}")


def compare_analysis_results(base_results, new_results, base_prefix, new_prefix):
    """Compare aggregate metrics between two analysis runs."""
    if not base_results or not new_results:
        print(f"\n  (Skipping aggregate metrics comparison — missing results JSON)")
        return

    print(f"\n{'='*70}")
    print(f"AGGREGATE METRICS COMPARISON")
    print(f"{'='*70}")

    metrics = [
        ("PRIMARY: 4pt unweighted kappa AAT delta", "strict_4point_aat", "unweighted_kappa", "delta"),
        ("PRIMARY: 4pt unweighted kappa AAT boot 5th", "strict_4point_aat", "unweighted_kappa", "bootstrap", "bootstrap_5th_pctile"),
        ("SUPPORTING: 4pt weighted kappa AAT delta", "strict_4point_aat", "weighted_kappa", "delta"),
        ("SUPPORTING: 4pt weighted kappa AAT boot 5th", "strict_4point_aat", "weighted_kappa", "bootstrap", "bootstrap_5th_pctile"),
        ("4pt exact match (LH vs majority)", "strict_comparison", "overall_exact"),
        ("4pt weighted kappa (LH vs majority)", "strict_comparison", "lh_kappa_weighted"),
        ("SUPPLEMENTARY: Binary delta", "binary_sentence_level", "delta"),
        ("SUPPLEMENTARY: Binary boot 5th", "binary_sentence_level", "bootstrap", "bootstrap_5th_pctile"),
    ]

    for label, *keys in metrics:
        def get_nested(d, ks):
            for k in ks:
                if isinstance(d, dict):
                    d = d.get(k)
                else:
                    return None
            return d

        v_base = get_nested(base_results, keys)
        v_new = get_nested(new_results, keys)
        if v_base is not None and v_new is not None:
            if isinstance(v_base, float):
                print(f"  {label:40s}: {base_prefix}={v_base:.4f}  {new_prefix}={v_new:.4f}  delta={v_new-v_base:+.4f}")
            else:
                print(f"  {label:40s}: {base_prefix}={v_base}  {new_prefix}={v_new}")


def main():
    p = argparse.ArgumentParser(description="Compare evaluator versions")
    p.add_argument("--prefixes", nargs="+", required=True,
                   help="Evaluator prefixes to compare (e.g., evalv3 evalv4)")
    args = p.parse_args()

    if len(args.prefixes) < 2:
        print("Need at least 2 prefixes to compare")
        sys.exit(1)

    base = args.prefixes[0]
    print(f"Base evaluator: {base}")

    df_base = load_sentence_examples(base)
    res_base = load_analysis_results(base)

    for new in args.prefixes[1:]:
        print(f"\nComparing: {base} vs {new}")
        df_new = load_sentence_examples(new)
        res_new = load_analysis_results(new)

        merged = merge_sentence_examples(df_base, df_new, base, new)

        # Save merged CSV
        out_path = sentence_comparison_path(base, new)
        merged.to_csv(out_path, index=False)
        print(f"  Saved merged comparison to {out_path}")

        # Print summary
        print_comparison_summary(merged, base, new)

        # Compare aggregate metrics
        compare_analysis_results(res_base, res_new, base, new)


if __name__ == "__main__":
    main()
