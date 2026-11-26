"""
Main orchestrator: loads data, runs all sections, saves results.

Usage (from project root):
    python -m analysis.main                         # evalv3 (default)
    python -m analysis.main --eval-prefix evalv4    # evalv4
    python -m analysis.main --eval-prefix textonly --plausibility-only
"""

import json
import argparse
from pathlib import Path

import numpy as np

from .config import (
    QUALTRICS_CSV, STIMULI_CSV, EVAL_CSV,
    EXCLUDE_ANNOTATORS, N_ITEMS, DELTAS,
    ensure_output_dirs, results_json_path,
)
from . import config as cfg
from .data_loading import (
    load_qualtrics, load_stimuli, load_llm_data,
    build_question_map, extract_responses,
)
from .sections import (
    survey_validation,
    hh_raw,
    strict_4point,
    strict_4point_aat,
    stratified_defect_aat,
    binary_aat,
    llm_confidence,
    visual_aat,
    additional,
    item_diagnostics,
    example_export,
    visual_examples,
)


def parse_args():
    p = argparse.ArgumentParser(description="Annotation agreement analysis")
    p.add_argument("--eval-prefix", default="evalv3",
                   help="Evaluator version prefix (e.g., evalv3, evalv4). "
                        "Controls which eval_run rows are loaded.")
    p.add_argument(
        "--eval-csv",
        default=None,
        help="Evaluation CSV to load (default: analysis.config.EVAL_CSV)",
    )
    p.add_argument(
        "--plausibility-only",
        action="store_true",
        help=(
            "Run the plausibility analyses, including the primary strict "
            "four-point AAT, but skip visual AAT and visual example exports. "
            "Use this for evaluators that intentionally omit visual fields."
        ),
    )
    p.add_argument(
        "--qualtrics-csv",
        default=None,
        help="Qualtrics export to score (default: analysis.config.QUALTRICS_CSV)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Write analysis JSON and exports here instead of Survey/outputs",
    )
    p.add_argument(
        "--finished-only",
        action="store_true",
        help="Keep responses whose Finished column is 1 or True",
    )
    return p.parse_args()


def _finished_response(value) -> bool:
    return str(value).strip().lower() in {"1", "true"}


def main():
    args = parse_args()
    eval_prefix = args.eval_prefix

    if args.output_dir:
        out = Path(args.output_dir)
        cfg.OUTPUT_DIR = out
        cfg.EXPORTS_DIR = out / "exports"
        cfg.FIGURES_DIR = out / "figures"
        cfg.TABLES_DIR = out / "tables"
    ensure_output_dirs()
    results_json = results_json_path(eval_prefix)
    cfg.EVAL_PREFIX = eval_prefix  # make available to sections

    np.random.seed(42)
    print("=" * 70)
    print(f"ANNOTATION AGREEMENT ANALYSIS v2 (Modular) — {eval_prefix}")
    print("=" * 70)

    # ---- Load data ----
    qualtrics_csv = Path(args.qualtrics_csv) if args.qualtrics_csv else QUALTRICS_CSV
    df, q_text = load_qualtrics(qualtrics_csv)
    if args.finished_only:
        if "Finished" not in df.columns:
            raise SystemExit(f"{qualtrics_csv} has no Finished column")
        keep = df["Finished"].map(_finished_response)
        print(f"Finished filter: kept {int(keep.sum())}/{len(df)} responses")
        df = df.loc[keep].reset_index(drop=True)
    items = load_stimuli(STIMULI_CSV)
    llm = load_llm_data(args.eval_csv or EVAL_CSV, items, eval_prefix=eval_prefix)
    qmap = build_question_map(items, q_text)
    plaus_df, diff_df, ident_df, pair_df, annotator_ids = extract_responses(
        df, qmap, items
    )

    if EXCLUDE_ANNOTATORS:
        print(f"\n  Excluding: {EXCLUDE_ANNOTATORS}")
        annotator_ids = [a for a in annotator_ids if a not in EXCLUDE_ANNOTATORS]
        plaus_df = plaus_df[~plaus_df["annotator"].isin(EXCLUDE_ANNOTATORS)]
        diff_df = diff_df[~diff_df["annotator"].isin(EXCLUDE_ANNOTATORS)]
        ident_df = ident_df[~ident_df["annotator"].isin(EXCLUDE_ANNOTATORS)]
        if len(pair_df) > 0:
            pair_df = pair_df[~pair_df["annotator"].isin(EXCLUDE_ANNOTATORS)]

    plaus_valid = plaus_df[plaus_df["rating"] != 5].copy()
    item_list = list(range(1, N_ITEMS + 1))
    results = {}

    # ---- A. Survey Validation ----
    res_a = survey_validation.run(
        plaus_df=plaus_df, plaus_valid=plaus_valid, diff_df=diff_df,
        qualtrics_df=df, n_ann=len(annotator_ids), n_items=N_ITEMS,
    )
    results.update(res_a)

    # ---- B. Raw H-H Agreement ----
    res_b = hh_raw.run(plaus_valid=plaus_valid, annotator_ids=annotator_ids)
    results.update({k: v for k, v in res_b.items() if not k.startswith("_")})
    pivot_raw = res_b["_pivot_raw"]

    # ---- C. Strict 4-Point Comparison ----
    res_c = strict_4point.run(
        plaus_valid=plaus_valid, diff_df=diff_df, llm=llm,
        annotator_ids=annotator_ids, item_list=item_list,
        pivot_raw=pivot_raw,
    )
    results.update({k: v for k, v in res_c.items() if not k.startswith("_")})
    spread_data = res_c["_spread_data"]
    diff_by_cond = res_c["_diff_by_cond"]

    # ---- C2. Strict 4-Point AAT (NEW) ----
    res_c2 = strict_4point_aat.run(
        plaus_valid=plaus_valid, llm=llm,
        annotator_ids=annotator_ids, item_list=item_list,
        pivot_raw=pivot_raw,
    )
    results.update(res_c2)

    # ---- C3. Stratified kappa AAT and generator-defect detection ----
    res_c3 = stratified_defect_aat.run(
        plaus_valid=plaus_valid, llm=llm,
        annotator_ids=annotator_ids, item_list=item_list,
        pivot_raw=pivot_raw,
    )
    results.update(res_c3)

    # ---- D. Binary AAT ----
    res_d = binary_aat.run(
        plaus_valid=plaus_valid, llm=llm,
        annotator_ids=annotator_ids, item_list=item_list,
    )
    results.update({k: v for k, v in res_d.items() if not k.startswith("_")})
    plaus_bin = res_d["_plaus_bin"]
    llm_binary_map = res_d["_llm_binary_map"]

    # ---- E. LLM Confidence ----
    res_e = llm_confidence.run(
        llm=llm, plaus_valid=plaus_valid, plaus_bin=plaus_bin,
        diff_df=diff_df, item_list=item_list,
    )
    results.update(res_e)

    # ---- F. Visual AAT ----
    # A text-only evaluator deliberately has no visual judgments. Keep the
    # historical full-analysis behavior as the default, and skip only when the
    # caller explicitly requests the plausibility-only protocol.
    ident_dist = {}
    if not args.plausibility_only:
        res_f = visual_aat.run(
            llm=llm, ident_df=ident_df, pair_df=pair_df,
            annotator_ids=annotator_ids, item_list=item_list,
            items=items,
        )
        results.update(res_f)
        ident_dist = res_f["visual_distinctiveness"].get("f1_identifiability", {}).get("distributions", {}).get("human", {})

    # ---- G. Additional Metrics ----
    res_g = additional.run(
        plaus_valid=plaus_valid, plaus_bin=plaus_bin,
        llm_binary_map=llm_binary_map, spread_data=spread_data,
        diff_by_cond=diff_by_cond, ident_dist=ident_dist,
        annotator_ids=annotator_ids,
    )
    results.update(res_g)

    # ---- H. Item Diagnostics (NEW) ----
    res_h = item_diagnostics.run(
        llm_confidence_results=results["llm_confidence"],
        items=items, llm=llm, plaus_valid=plaus_valid, diff_df=diff_df,
    )
    results.update(res_h)

    # ---- I. Per-Sentence Examples Export (NEW) ----
    res_i = example_export.run(
        plaus_valid=plaus_valid, diff_df=diff_df, llm=llm,
        items=items, annotator_ids=annotator_ids, item_list=item_list,
        eval_prefix=eval_prefix,
    )
    results.update(res_i)

    # ---- J. Visual Examples Export (NEW) ----
    if not args.plausibility_only:
        res_j = visual_examples.run(
            ident_df=ident_df, pair_df=pair_df, items=items, llm=llm,
            annotator_ids=annotator_ids, item_list=item_list,
            eval_prefix=eval_prefix,
        )
        results.update(res_j)

    # ---- Save ----
    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {results_json}")

    # ---- Summary ----
    _print_summary(results)


def _print_summary(results):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    sv = results["survey_validation"]
    print(f"\n  Annotators: {sv['n_annotators']}, Items: {sv['n_items']}, "
          f"Sentences: {sv['n_sentences']}")

    hh = results["hh_agreement_raw"]
    print(f"\n  Raw H-H (4-pt ordinal):")
    print(f"    Krippendorff's alpha: {hh['krippendorff_alpha']:.4f}")
    print(f"    Mean weighted kappa:  {hh['mean_kappa']:.4f}")

    sc = results["strict_comparison"]
    print(f"\n  STRICT 4-PT (direct LLM vs Human majority):")
    print(f"    Exact match:  {sc['overall_exact']:.1%}")
    print(f"    Weighted kw:  {sc['lh_kappa_weighted']:.4f}")
    print(f"    --> Hard conditions low, motivates binary")

    if "strict_4point_aat" in results:
        s4 = results["strict_4point_aat"]
        wk = s4["weighted_kappa"]
        uk = s4["unweighted_kappa"]
        wk_b = wk["bootstrap"]
        uk_b = uk["bootstrap"]
        print(f"\n  PRIMARY — 4-PT AAT (LLM vs. individual annotators):")
        print(f"    Unweighted kappa (exact, primary): HH={uk['hh_mean']:.4f}  LH={uk['lh_mean']:.4f}  "
              f"delta={uk['delta']:+.4f}  boot5th={uk_b['bootstrap_5th_pctile']:.4f}")
        print(f"    Weighted kappa (ordering, support): HH={wk['hh_mean']:.4f}  LH={wk['lh_mean']:.4f}  "
              f"delta={wk['delta']:+.4f}  boot5th={wk_b['bootstrap_5th_pctile']:.4f}")
        for d in DELTAS:
            wk_tag = "PASS" if wk_b["noninferiority"][f"delta_{d:.2f}"] else "FAIL"
            uk_tag = "PASS" if uk_b["noninferiority"][f"delta_{d:.2f}"] else "FAIL"
            print(f"    delta={d:.2f}: unweighted={uk_tag} (primary)  "
                  f"weighted={wk_tag} (supporting)")

    if "stratified_defect_aat" in results:
        strata = results["stratified_defect_aat"]
        print(f"\n  STRATIFIED AAT (easy must not mask hard):")
        for key in ("kappa_combined", "kappa_easy_PE_IE", "kappa_hard_PH_IH"):
            row = strata[key]
            tag = "PASS" if row["pass"] else "FAIL"
            print(f"    {key:22s} delta={row['delta']:+.4f}  p5={row['p5']:+.4f}  {tag}")
        for key, row in strata.items():
            if not key.startswith("po_"):
                continue
            tag = "PASS" if row["pass"] else "FAIL"
            print(f"    {key:22s} delta={row['delta']:+.4f}  p5={row['p5']:+.4f}  {tag}")
        defect = strata["defect_detection"]
        print(f"    defect recall={defect['overall_recall']:.4f}  "
              f"precision={defect['overall_precision']:.4f}  "
              f"f1={defect['overall_f1']:.4f}")
        print(f"    PH defect recall={defect['PH_defect_recall']:.4f}  "
              f"IH defect recall={defect['IH_defect_recall']:.4f}")

    bs = results["binary_sentence_level"]
    bp = bs["bootstrap"]
    print(f"\n  SUPPLEMENTARY — Binary plausibility (n={sv['n_sentences']} sentences):")
    print(f"    (Binary split at PH/IH boundary — highest human disagreement zone)")
    print(f"    H-H kappa:   {bs['hh_mean']:.4f}")
    print(f"    L-H kappa:   {bs['lh_mean']:.4f}")
    print(f"    Delta:       {bs['delta']:+.4f}")
    print(f"    Boot 5th:    {bp['bootstrap_5th_pctile']:.4f}")
    for d in DELTAS:
        tag = "PASS" if bp["noninferiority"][f"delta_{d:.2f}"] else "FAIL"
        print(f"    delta={d:.2f}:   {tag}")

    if "per_condition_aat" in bs:
        print(f"\n  PER-CONDITION AAT (AC1):")
        for cond, data in bs["per_condition_aat"].items():
            ac = data["ac1"]
            prev = data["prevalence_plausible"]["human_mean"]
            boot_info = ac.get("bootstrap", {})
            p5 = boot_info.get("bootstrap_5th_pctile", "?")
            print(f"    {cond:20s}: HH={ac['hh_mean']:.4f} LH={ac['lh_mean']:.4f} "
                  f"delta={ac['delta']:+.4f} boot5th={p5} (prev={prev:.0%})")

    lc = results["llm_confidence"]
    fa = lc["fourpoint_agreement"]
    vb = lc["verdict_type_breakdown"]
    adj = lc["adjacency_analysis"]
    print(f"\n  LLM CONFIDENCE (4-point primary):")
    print(f"    4-pt exact match: {fa['agree']}/{lc['total_confident']} ({fa['accuracy']:.0%})")
    print(f"    Binary match:     {lc['binary_agreement_supplementary']['agree']}/{lc['total_confident']} ({lc['binary_agreement_supplementary']['accuracy']:.0%})")
    print(f"    CORRECT+AGREE={vb['correct_4pt_agree']}, INCORRECT+AGREE={vb['incorrect_4pt_agree']}")
    print(f"    CORRECT+DISAGREE={vb['correct_4pt_disagree']}, INCORRECT+DISAGREE={vb['incorrect_4pt_disagree']}")
    if adj["n_total_disagree"] > 0:
        print(f"    Disagree adjacency: {adj['n_adjacent']}/{adj['n_total_disagree']} are 1-step")
    if "difficulty_correlation" in lc and "mannwhitney_p" in lc["difficulty_correlation"]:
        dc = lc["difficulty_correlation"]
        print(f"    Difficulty: agree={dc['mean_difficulty_agree']:.3f} vs "
              f"disagree={dc['mean_difficulty_disagree']:.3f} (p={dc['mannwhitney_p']:.4f})")

    if "visual_distinctiveness" in results:
        vd = results["visual_distinctiveness"]
        print(f"\n  VISUAL DISTINCTIVENESS (decomposed):")
        f1 = vd["f1_identifiability"]["threepoint"]
        f1b = vd["f1_identifiability"]["binary"]
        f2k = vd["f2_pairwise"]["kappa"]
        f2a = vd["f2_pairwise"]["ac1"]
        f3 = vd["f3_combined"]
        f1_boot = f1.get("bootstrap", {})
        f2k_boot = f2k.get("bootstrap", {})
        f2a_boot = f2a.get("bootstrap", {})
        print(f"    F1 Ident (3pt): HH={f1['hh_mean']:.4f} LH={f1['lh_mean']:.4f} "
              f"delta={f1['delta']:+.4f} boot5th={f1_boot.get('bootstrap_5th_pctile', '?')}")
        print(f"    F1 Ident (bin): HH={f1b['hh_mean']:.4f} LH={f1b['lh_mean']:.4f} "
              f"delta={f1b['delta']:+.4f}")
        print(f"    F2 Pair (kappa): HH={f2k['hh_mean']:.4f} LH={f2k['lh_mean']:.4f} "
              f"delta={f2k['delta']:+.4f} boot5th={f2k_boot.get('bootstrap_5th_pctile', '?')}")
        print(f"    F2 Pair (AC1):  HH={f2a['hh_mean']:.4f} LH={f2a['lh_mean']:.4f} "
              f"delta={f2a['delta']:+.4f} boot5th={f2a_boot.get('bootstrap_5th_pctile', '?')}")
        print(f"    F3 Combined:    HH={f3['hh_mean']:.4f} LH={f3['lh_mean']:.4f} "
              f"delta={f3['delta']:+.4f}")

    idd = results.get("item_diagnostics", {})
    if idd and "fourpoint_disagreements" in idd:
        fpd = idd["fourpoint_disagreements"]
        print(f"\n  ITEM DIAGNOSTICS (4-point):")
        print(f"    4-pt disagreements: {fpd['n_total']}")
        print(f"    By condition: {fpd['by_condition']}")
        print(f"    By distance: {fpd['by_distance']}")
        print(f"    Also binary disagree: {fpd['n_also_binary_disagree']}, 4pt-only: {fpd['n_only_4pt_disagree']}")
        print(f"    Across {fpd['n_unique_items']} unique items")


if __name__ == "__main__":
    main()
