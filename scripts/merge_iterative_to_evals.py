#!/usr/bin/env python3
"""
Comprehensive merge: reads evaluation data from all source directories,
assigns quality tiers, flags missing evaluations.
"""
import sys, os, json, glob, csv, argparse
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import load_verbs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERB_FILE = "data/verbs60.txt"
EVAL_DIR = "results/evaluations"
ITER_DIR = "results/iterative"
DEFAULT_VISUAL_OUTPUT_CSV = "results/aggregate/all_evaluations_merged_preview.csv"
DEFAULT_TEXT_OUTPUT_CSV = "results/aggregate/all_evaluations_merged_text_preview.csv"

parser = argparse.ArgumentParser(description="Merge evaluator JSON files into a CSV")
parser.add_argument("--verb-file", default=VERB_FILE, help="Expected verb list")
parser.add_argument("--eval-dir", default=EVAL_DIR, help="Single-shot evaluation root")
parser.add_argument("--iter-dir", default=ITER_DIR, help="Iterative run root")
parser.add_argument(
    "--eval-prefix",
    default=None,
    help="Only merge evaluation directories beginning with this prefix",
)
parser.add_argument(
    "--output-csv",
    default=None,
    help=("Destination CSV path. Defaults to separate visual/text aggregate files "
          "according to --criterion."),
)
parser.add_argument(
    "--criterion",
    choices=["visual", "text"],
    default="visual",
    help="Tier criterion; visual is the backward-compatible default",
)
parser.add_argument(
    "--iter-prefix",
    default=None,
    help="Only merge iterative run directories beginning with this prefix",
)
parser.add_argument(
    "--iter-only",
    action="store_true",
    help="Exclude single-shot evaluation directories",
)
parser.add_argument(
    "--iterative-stage",
    choices=["final", "rb", "r2r", "both"],
    default="final",
    help=("Iterative evaluation to merge: historical latest-successful ('final'), "
          "iteration 0 ('rb'), latest evaluated/stopped iteration ('r2r'), or both."),
)
args = parser.parse_args()
VERB_FILE = args.verb_file
EVAL_DIR = args.eval_dir
ITER_DIR = args.iter_dir
EVAL_PREFIX_FILTER = args.eval_prefix
ITER_PREFIX_FILTER = args.iter_prefix
CRITERION = args.criterion
OUTPUT_CSV = args.output_csv or (
    DEFAULT_VISUAL_OUTPUT_CSV if CRITERION == "visual" else DEFAULT_TEXT_OUTPUT_CSV
)

JUDGE_MODEL = "gpt-5.1-2025-11-13"
JUDGE_PROVIDER = "openai"
REASONING_EFFORT = "high"

verbs = load_verbs(VERB_FILE)
print(f"Expected verbs ({len(verbs)}): {verbs}")

# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------
def assign_tier(row, criterion="visual"):
    """
    Assign quality tier based on three dimensions:
      - Thematic Fit:  score_strict (0-4)
      - Scene Validity: scene_verdict (GOOD / AGENT_FIXABLE / NEEDS_REDESIGN)
      - Visual:         visual_verdict (ALL_DISTINCT / MOSTLY_DISTINCT / PROBLEMATIC)

    Tiers:
      GOLD:   4/4 + GOOD + ALL_DISTINCT
      SILVER: 4/4 + GOOD + MOSTLY_DISTINCT
      BRONZE: 4/4 + scene has issues (AGENT_FIXABLE or NEEDS_REDESIGN)
      FAIL:   score < 4, or evaluation unsuccessful
    """
    if not row.get("eval_success") or row.get("eval_success") == "False":
        return "FAIL"

    try:
        score = int(row.get("score_strict", 0))
    except (ValueError, TypeError):
        return "FAIL"

    if score < 4:
        return "FAIL"

    if criterion not in {"visual", "text"}:
        raise ValueError(f"Unknown criterion: {criterion}")

    scene = row.get("scene_verdict", "")
    visual = row.get("visual_verdict", "")

    # Text GOLD is the historical GOLD definition with only the visual
    # requirement removed. SILVER is visual-specific and is not assigned.
    if criterion == "text":
        if scene == "GOOD":
            return "GOLD"
        return "BRONZE"

    if scene == "GOOD" and visual == "ALL_DISTINCT":
        return "GOLD"
    elif scene == "GOOD" and visual == "MOSTLY_DISTINCT":
        return "SILVER"
    elif score == 4:
        return "BRONZE"
    else:
        return "FAIL"

def get_eval_run(run_name):
    if not run_name.startswith("evalv31agent_qwen30b-think_") and (run_name.endswith("_v7_agent_iterative_60") or run_name.endswith("_v7_agent_verbalized_iterative_60")):
        return "evalv3.1_agent_" + run_name
    else:
        return run_name

# ---------------------------------------------------------------------------
# Helper: extract evaluation row from an evaluation JSON dict
# ---------------------------------------------------------------------------
def extract_eval_row(eval_data, eval_run, source_run, verb, extra_summary="",
                     criterion="visual", method=""):
    ev = eval_data.get("evaluation", {}) or {}
    scores = eval_data.get("scores", {}) or {}
    scene = ev.get("scene_assessment", {}) or {}
    gradient = ev.get("gradient_ordering", {}) or {}
    visual = ev.get("visual_evaluation", {}) or {}
    conditions = ev.get("condition_evaluations", {}) or {}

    def get_cond(cond_name, field, default=""):
        c = conditions.get(cond_name, {})
        return c.get(field, default) if isinstance(c, dict) else default

    # --- v3: extract overlap audit for implausible_hard ---
    ih_cond = conditions.get("implausible_hard", {})
    ih_overlap = ih_cond.get("overlap_audit", {}) if isinstance(ih_cond, dict) else {}
    overlap_count = ih_overlap.get("overlap_count", "") if isinstance(ih_overlap, dict) else ""
    overlap_sufficient = ih_overlap.get("sufficient", "") if isinstance(ih_overlap, dict) else ""

    # --- v3: extract identifiability ratings ---
    ident = visual.get("identifiability", {}) or {}
    def get_ident(cond_name):
        ic = ident.get(cond_name, {})
        return ic.get("rating", "") if isinstance(ic, dict) else ""

    # --- v3: extract borderline_count from overall ---
    overall = ev.get("overall", {}) or {}
    borderline_count = overall.get("borderline_count", scores.get("borderline_count", ""))

    row = {
        "eval_run": eval_run,
        "source_run": source_run,
        "judge_model": JUDGE_MODEL,
        "judge_provider": JUDGE_PROVIDER,
        "reasoning_effort": REASONING_EFFORT,
        "criterion": criterion,
        "method": method,
        "verb": verb,
        "eval_success": eval_data.get("success", False),
        "elapsed_s": round(eval_data.get("elapsed", 0), 2),
        "input_tokens": eval_data.get("token_info", {}).get("input_tokens", ""),
        "output_tokens": eval_data.get("token_info", {}).get("output_tokens", ""),
        "score_strict": scores.get("score_strict", ""),
        "score_lenient": scores.get("score_lenient", ""),
        "has_scene_error": scores.get("has_scene_error", False),
        "verdict_plausible_easy": get_cond("plausible_easy", "verdict"),
        "actual_plausible_easy": get_cond("plausible_easy", "actual_condition"),
        "verdict_plausible_hard": get_cond("plausible_hard", "verdict"),
        "actual_plausible_hard": get_cond("plausible_hard", "actual_condition"),
        "verdict_implausible_easy": get_cond("implausible_easy", "verdict"),
        "actual_implausible_easy": get_cond("implausible_easy", "actual_condition"),
        "verdict_implausible_hard": get_cond("implausible_hard", "verdict"),
        "actual_implausible_hard": get_cond("implausible_hard", "actual_condition"),
        "scene_verdict": scene.get("scene_verdict", ""),
        "gradient_supportable": scene.get("gradient_supportable", ""),
        "scene_notes": scene.get("scene_notes", ""),
        "gradient_correct": gradient.get("correctly_ordered", ""),
        "visual_verdict": visual.get("visual_verdict", ""),
        # --- v3 fields (empty string when absent for backward compat) ---
        "overlap_count": overlap_count,
        "overlap_sufficient": overlap_sufficient,
        "borderline_count": borderline_count,
        "ident_plausible_easy": get_ident("plausible_easy"),
        "ident_plausible_hard": get_ident("plausible_hard"),
        "ident_implausible_hard": get_ident("implausible_hard"),
        "ident_implausible_easy": get_ident("implausible_easy"),
        "unidentifiable_agents": visual.get("unidentifiable_agents", ""),
        "summary": extra_summary,
    }

    # Assign tier
    row["tier"] = assign_tier(row, criterion=criterion)

    return row

# ---------------------------------------------------------------------------
# Phase 1: Read single-shot evaluations (v4, v5) from results/evaluations/
# ---------------------------------------------------------------------------
print(f"\n--- Phase 1: Single-shot evaluations from {EVAL_DIR} ---")
all_rows = []
eval_run_dirs = [] if args.iter_only else sorted(glob.glob(os.path.join(EVAL_DIR, "*")))
for eval_run_dir in eval_run_dirs:
    if not os.path.isdir(eval_run_dir):
        continue
    eval_run = os.path.basename(eval_run_dir)
    if EVAL_PREFIX_FILTER and not eval_run.startswith(EVAL_PREFIX_FILTER):
        continue
    source_run = eval_run[5:] if eval_run.startswith("eval_") else eval_run

    for json_file in sorted(glob.glob(os.path.join(eval_run_dir, "*.json"))):
        verb = os.path.basename(json_file).replace(".json", "")
        with open(json_file) as f:
            eval_data = json.load(f)

        if not eval_data.get("success", False):
            print(f"  WARN {eval_run}/{verb}: evaluation not successful")

        row = extract_eval_row(
            eval_data, eval_run, source_run, verb, extra_summary="Single-shot",
            criterion=CRITERION,
        )
        all_rows.append(row)

print(f"  Found {len(all_rows)} single-shot evaluation rows")

# ---------------------------------------------------------------------------
# Phase 2: Read iterative evaluations from results/iterative/
# ---------------------------------------------------------------------------
print(f"\n--- Phase 2: Iterative evaluations from {ITER_DIR} ---")
iter_count = 0
iter_run_dirs = [] if EVAL_PREFIX_FILTER else sorted(glob.glob(os.path.join(ITER_DIR, "*")))
for run_dir in iter_run_dirs:
    if not os.path.isdir(run_dir):
        continue
    run_name = os.path.basename(run_dir)
    if ITER_PREFIX_FILTER and not run_name.startswith(ITER_PREFIX_FILTER):
        continue

    for verb_dir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if not os.path.isdir(verb_dir):
            continue
        verb = os.path.basename(verb_dir)
        if verb not in verbs:
            continue

        eval_files = sorted(glob.glob(os.path.join(verb_dir, "iter_*_evaluation.json")))
        if not eval_files:
            print(f"  WARN {run_name}/{verb}: no evaluation files")
            continue

        history_path = os.path.join(verb_dir, "history.json")
        history = {}
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
    

        def iteration_number(path):
            match = re.search(r"iter_(\d+)_evaluation\.json$", path)
            return int(match.group(1)) if match else -1

        eval_files = sorted(eval_files, key=iteration_number)
        selections = []
        if args.iterative_stage == "final":
            # Preserve historical behavior: use the latest successful evaluation.
            selected = None
            for ef in reversed(eval_files):
                with open(ef) as f:
                    candidate = json.load(f)
                if candidate.get("success"):
                    selected = (ef, candidate)
                    break
            selections = [("", selected)]
        else:
            if args.iterative_stage in {"rb", "both"}:
                rb_path = next((ef for ef in eval_files if iteration_number(ef) == 0), None)
                rb_data = None
                if rb_path:
                    with open(rb_path) as f:
                        rb_data = json.load(f)
                selections.append(("RB", (rb_path, rb_data) if rb_path else None))
            if args.iterative_stage in {"r2r", "both"}:
                r2r_path = eval_files[-1] if eval_files else None
                r2r_data = None
                if r2r_path:
                    with open(r2r_path) as f:
                        r2r_data = json.load(f)
                selections.append(("R2R", (r2r_path, r2r_data) if r2r_path else None))

        for method, selected in selections:
            method_suffix = f"_{method}" if method else ""
            eval_run_name = get_eval_run(run_name) + method_suffix
            source_run_name = run_name + method_suffix
            eval_data = selected[1] if selected else None
            iteration_idx = iteration_number(selected[0]) if selected else "?"

            if not eval_data:
                print(f"  WARN {run_name}/{verb} [{method or 'final'}]: no evaluation found")
                row = {
                    "eval_run": eval_run_name,
                    "source_run": source_run_name,
                    "judge_model": JUDGE_MODEL, "judge_provider": JUDGE_PROVIDER,
                    "reasoning_effort": REASONING_EFFORT,
                    "criterion": CRITERION, "method": method,
                    "verb": verb, "eval_success": False,
                    "elapsed_s": "", "input_tokens": "", "output_tokens": "",
                    "score_strict": "", "score_lenient": "", "has_scene_error": "",
                    "verdict_plausible_easy": "", "actual_plausible_easy": "",
                    "verdict_plausible_hard": "", "actual_plausible_hard": "",
                    "verdict_implausible_easy": "", "actual_implausible_easy": "",
                    "verdict_implausible_hard": "", "actual_implausible_hard": "",
                    "scene_verdict": "", "gradient_supportable": "", "scene_notes": "",
                    "gradient_correct": "", "visual_verdict": "",
                    "summary": f"Iterative: {history.get('total_iterations', '?')} iterations, EVAL FAILED",
                    "tier": "FAIL",
                }
            else:
                summary = (
                    f"Iterative {method or 'final'} at iteration {iteration_idx}: "
                    f"{history.get('total_iterations', '?')} iterations, "
                    f"pass={history.get('final_pass', '?')}"
                )
                row = extract_eval_row(
                    eval_data=eval_data,
                    eval_run=eval_run_name,
                    source_run=source_run_name,
                    verb=verb,
                    extra_summary=summary,
                    criterion=CRITERION,
                    method=method,
                )
            all_rows.append(row)
            iter_count += 1

print(f"  Found {iter_count} iterative evaluation rows")

# ---------------------------------------------------------------------------
# Phase 3: Gap analysis
# ---------------------------------------------------------------------------
print(f"\n--- Phase 3: Gap analysis ---")
all_eval_runs = sorted(set(r["eval_run"] for r in all_rows))
print(f"All eval_runs found ({len(all_eval_runs)}):")
for er in all_eval_runs:
    count = sum(1 for r in all_rows if r["eval_run"] == er)
    success = sum(1 for r in all_rows if r["eval_run"] == er and r["eval_success"])
    print(f"  {er}: {count} verbs ({success} successful)")

print(f"\nMissing verbs per eval_run:")
missing_count = 0
for er in all_eval_runs:
    found_verbs = set(r["verb"] for r in all_rows if r["eval_run"] == er)
    missing_verbs = set(verbs) - found_verbs
    if missing_verbs:
        print(f"  MISSING {er}: {sorted(missing_verbs)}")
        missing_count += len(missing_verbs)
if missing_count == 0:
    print("  None! All eval_runs have all verbs.")
else:
    print(f"\n  Total missing: {missing_count} verb-evaluation pairs")

failed = [(r["eval_run"], r["verb"]) for r in all_rows if not r["eval_success"]]
if failed:
    print(f"\nFailed evaluations ({len(failed)}):")
    for er, v in failed:
        print(f"  {er}/{v}")
else:
    print(f"\nNo failed evaluations.")

# ---------------------------------------------------------------------------
# Phase 4: Tier summary
# ---------------------------------------------------------------------------
print(f"\n--- Phase 4: Tier distribution ---")

# Overall tier distribution
tier_counts = {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "FAIL": 0}
for r in all_rows:
    tier_counts[r.get("tier", "FAIL")] += 1
total = sum(tier_counts.values())
print(f"\nOverall tier distribution ({total} total):")
for tier in ["GOLD", "SILVER", "BRONZE", "FAIL"]:
    pct = tier_counts[tier] / total * 100 if total else 0
    print(f"  {tier:<8} {tier_counts[tier]:>4}  ({pct:.1f}%)")

# Per eval_run tier distribution
print(f"\nPer eval_run tier distribution:")
print(f"{'eval_run':<55} {'GOLD':>6} {'SILVER':>6} {'BRONZE':>6} {'FAIL':>6} {'Total':>6}")
print(f"{'-'*55} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for er in all_eval_runs:
    er_rows = [r for r in all_rows if r["eval_run"] == er]
    tc = {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "FAIL": 0}
    for r in er_rows:
        tc[r.get("tier", "FAIL")] += 1
    print(f"{er:<55} {tc['GOLD']:>6} {tc['SILVER']:>6} {tc['BRONZE']:>6} {tc['FAIL']:>6} {len(er_rows):>6}")

# Per verb tier distribution across all eval_runs
print(f"\nPer verb tier distribution (across all eval_runs):")
print(f"{'verb':<12} {'GOLD':>6} {'SILVER':>6} {'BRONZE':>6} {'FAIL':>6}")
print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for v in verbs:
    v_rows = [r for r in all_rows if r["verb"] == v]
    tc = {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "FAIL": 0}
    for r in v_rows:
        tc[r.get("tier", "FAIL")] += 1
    print(f"{v:<12} {tc['GOLD']:>6} {tc['SILVER']:>6} {tc['BRONZE']:>6} {tc['FAIL']:>6}")

# ---------------------------------------------------------------------------
# Phase 5: Write output
# ---------------------------------------------------------------------------
print(f"\n--- Phase 5: Writing output ---")

# Ensure tier is right after summary in column order
all_keys = []
seen = set()
for row in all_rows:
    for k in row:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}")
