#!/usr/bin/env python3
"""
Evaluate Events generation outputs using a judge model.

Takes intermediate results from a generation run and evaluates each verb.

Examples:
  # Evaluate a specific run using GPT-5 thinking
  python scripts/run_evaluator.py \
    --input-dir results/intermediate/gpt-5.1-2025-11-13_t0.0_20260224_235128 \
    --provider openai --model gpt-5.1-2025-11-13 \
    --reasoning-effort high

  # Evaluate patient-variant run using Claude
  python scripts/run_evaluator.py \
    --input-dir results/intermediate/gpt-5.1-2025-11-13_v7_patient_60 \
    --provider claude --model claude-sonnet-4-6 \
    --eval-prompt prompts/evaluator_v3.1_patient.json

  # Evaluate specific verbs from a run
  python scripts/run_evaluator.py \
    --input-dir results/intermediate/Qwen3-4B_t0.0_20260224 \
    --provider openai --model gpt-5.1-2025-11-13 \
    --verbs catch play break

  # Evaluate a single JSON file
  python scripts/run_evaluator.py \
    --input-file results/intermediate/some_run/catch.json \
    --provider openai --model gpt-5.1-2025-11-13
"""

import os
import sys
import json
import glob
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.evaluator import evaluate_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Events generation outputs")

    # Input (one of these required)
    p.add_argument("--input-dir", default=None,
                   help="Directory with intermediate JSON files from a generation run")
    p.add_argument("--input-file", default=None,
                   help="Single intermediate JSON file to evaluate")

    # Optional verb filter
    p.add_argument("--verbs", nargs="+", default=None,
                   help="Only evaluate these verbs (default: all in input-dir)")

    # Judge model
    p.add_argument("--provider", required=True, choices=["openai", "vllm", "claude"],
                   help="Provider for judge model")
    p.add_argument("--model", required=True, help="Judge model name")
    p.add_argument("--reasoning-effort", default="high",
                   choices=["", "low", "medium", "high", "xhigh", "max"],
                   help="Reasoning effort for judges that support it "
                        "(default: high; xhigh for Qwen3.8; max for GLM-5.3)")
    p.add_argument("--think-budget", type=int, default=8000,
                   help="Thinking budget tokens for supported vLLM and Claude judges")
    p.add_argument("--thinking-effort", default="high",
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="Effort level for Claude adaptive thinking (default: high)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None,
                   help="Optional nucleus-sampling probability")
    p.add_argument("--top-k", type=int, default=None,
                   help="Optional top-k sampling cutoff (local vLLM only)")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--port", type=int, default=8000, help="vLLM port (if using local judge)")

    # Evaluator prompt
    p.add_argument("--eval-prompt", default="prompts/evaluator_v3.1_agent.json",
                   help="Path to evaluator prompt JSON")
    p.add_argument(
        "--criterion",
        choices=["visual", "text"],
        default="visual",
        help=("Evaluation criterion. 'text' permits evaluator output without "
              "visual_evaluation; default preserves historical visual validation."),
    )

    # Output
    p.add_argument("--run-name", default=None, help="Evaluation run name (auto-generated if omitted)")
    p.add_argument(
        "--aggregate-csv",
        default=None,
        help=("Aggregate CSV destination. Defaults to all_evaluations.csv for visual "
              "runs and all_evaluations_text.csv for text-only runs."),
    )
    p.add_argument(
        "--no-aggregate",
        action="store_true",
        help=(
            "Write per-verb evaluation JSON files but do not update "
            "results/aggregate/all_evaluations.csv. Useful for concurrent "
            "Slurm replicates; rebuild the merged CSV afterward with "
            "scripts/merge_iterative_to_evals.py."
        ),
    )

    return p.parse_args()


def load_generation_results(args) -> list:
    """Load generation result JSON files. Returns list of (verb, parsed_output, filepath) tuples."""
    results = []

    if args.input_file:
        with open(args.input_file) as f:
            data = json.load(f)
        if data.get("success") and data.get("parsed"):
            results.append((data["verb"], data["parsed"], args.input_file))
        else:
            logger.warning(f"Skipping {args.input_file}: generation was not successful")
        return results

    if args.input_dir:
        pattern = os.path.join(args.input_dir, "*.json")
        for fp in sorted(glob.glob(pattern)):
            with open(fp) as f:
                data = json.load(f)
            verb = data.get("verb", os.path.basename(fp).replace(".json", ""))

            if args.verbs and verb not in [v.lower() for v in args.verbs]:
                continue

            if data.get("success") and data.get("parsed"):
                results.append((verb, data["parsed"], fp))
            else:
                logger.warning(f"Skipping {fp}: generation was not successful")

        return results

    logger.error("Provide --input-dir or --input-file")
    sys.exit(1)


def main():
    args = parse_args()
    gen_results = load_generation_results(args)

    if not gen_results:
        logger.error("No valid generation results found to evaluate")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.input_dir:
        source_run = os.path.basename(args.input_dir.rstrip("/"))
    elif args.input_file:
        source_run = os.path.basename(os.path.dirname(args.input_file))
    else:
        source_run = "unknown"

    criterion_tag = "" if args.criterion == "visual" else "_text"
    run_name = args.run_name or f"eval_{source_run}_by_{args.model}{criterion_tag}_{ts}"
    safe_run = run_name.replace("/", "_").replace(" ", "_")

    eval_dir = os.path.join("results", "evaluations", safe_run)
    agg_dir = os.path.join("results", "aggregate")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(agg_dir, exist_ok=True)

    eval_csv = args.aggregate_csv or os.path.join(
        agg_dir,
        "all_evaluations.csv" if args.criterion == "visual" else "all_evaluations_text.csv",
    )

    logger.info(f"Evaluation run: {safe_run}")
    logger.info(f"Judge: {args.provider} | {args.model} | reasoning={args.reasoning_effort}")
    logger.info(f"Criterion: {args.criterion}")
    logger.info(f"Evaluating {len(gen_results)} verb(s): {[v for v, _, _ in gen_results]}")

    all_rows = []

    for i, (verb, parsed_output, source_path) in enumerate(gen_results, 1):
        logger.info(f"[{i}/{len(gen_results)}] Evaluating verb: '{verb}'")

        eval_result = evaluate_events(
            verb=verb,
            generated_output=parsed_output,
            evaluator_prompt_path=args.eval_prompt,
            provider=args.provider,
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            port=args.port,
            reasoning_effort=args.reasoning_effort,
            think_budget=args.think_budget,
            thinking_effort=args.thinking_effort,
            criterion=args.criterion,
        )

        # Save detailed evaluation JSON
        out_path = os.path.join(eval_dir, f"{verb}.json")
        with open(out_path, "w") as f:
            json.dump(eval_result, f, indent=2, default=str)
        logger.info(f"  Saved: {out_path} | Success: {eval_result['success']}")

        # Build CSV row
        row = {
            "eval_run": safe_run,
            "source_run": source_run,
            "judge_model": args.model,
            "judge_provider": args.provider,
            "reasoning_effort": args.reasoning_effort,
            "criterion": args.criterion,
            "verb": verb,
            "eval_success": eval_result["success"],
            "elapsed_s": round(eval_result["elapsed"], 2),
            "input_tokens": eval_result["token_info"].get("input_tokens", 0),
            "output_tokens": eval_result["token_info"].get("output_tokens", 0),
        }

        if eval_result["success"] and eval_result["scores"]:
            scores = eval_result["scores"]
            row["score_strict"] = scores["score_strict"]
            row["score_lenient"] = scores["score_lenient"]
            row["has_scene_error"] = scores["has_scene_error"]

            for cond in ["plausible_easy", "plausible_hard", "implausible_easy", "implausible_hard"]:
                cond_data = scores["conditions"].get(cond, {})
                row[f"verdict_{cond}"] = cond_data.get("verdict", "")
                row[f"actual_{cond}"] = cond_data.get("actual_condition", "")

            ev = eval_result["evaluation"]
            sa = ev.get("scene_assessment", {}) or {}
            ve = ev.get("visual_evaluation", {}) or {}

            row["scene_verdict"] = sa.get("scene_verdict", "")
            row["gradient_supportable"] = sa.get("gradient_supportable", "")
            row["scene_notes"] = sa.get("scene_notes", "")
            row["gradient_correct"] = ev.get("gradient_ordering", {}).get("correctly_ordered", "")
            row["visual_verdict"] = ve.get("visual_verdict", "")
            row["summary"] = ev.get("overall", {}).get("summary", "")

            # New columns for non-agent variants (patient / instrument / location).
            # Empty string on agent-variant rows since the fields are not emitted.
            row["agent_action_prototypical"] = sa.get("agent_action_prototypical", "")
            fixed_ident = ve.get("fixed_agent_identifiability", {}) or {}
            row["fixed_agent_ident"] = (
                fixed_ident.get("rating", "") if isinstance(fixed_ident, dict) else ""
            )
        else:
            row["score_strict"] = ""
            row["score_lenient"] = ""
            row["has_scene_error"] = ""
            for cond in ["plausible_easy", "plausible_hard", "implausible_easy", "implausible_hard"]:
                row[f"verdict_{cond}"] = ""
                row[f"actual_{cond}"] = ""
            row["scene_notes"] = ""
            row["gradient_correct"] = ""
            row["summary"] = "; ".join(eval_result["validation_errors"])
            row["agent_action_prototypical"] = ""
            row["fixed_agent_ident"] = ""

        all_rows.append(row)

    # Write evaluation CSV (replacing any existing rows for same eval_run+verb)
    if all_rows and not args.no_aggregate:
        from utils.common import append_or_replace_csv
        append_or_replace_csv(eval_csv, all_rows, key_columns=["eval_run", "verb"])
    elif all_rows:
        logger.info("Skipping aggregate CSV update (--no-aggregate)")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY: {safe_run}")
    print(f"{'='*70}")
    print(f"{'Verb':<12} {'Strict':>8} {'Lenient':>8} {'PE':>10} {'PH':>10} {'IH':>10} {'IE':>10} {'Visual':>16} {'Scene':>16}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*16} {'-'*16}")

    for row in all_rows:
        if row["eval_success"]:
            scene_flag = row.get("scene_verdict", "?")
            visual_flag = row.get("visual_verdict") or ("N/A" if args.criterion == "text" else "?")
            print(f"{row['verb']:<12} {str(row['score_strict'])+'/4':>8} "
                  f"{str(row['score_lenient'])+'/4':>8} "
                  f"{row.get('verdict_plausible_easy','?'):>10} "
                  f"{row.get('verdict_plausible_hard','?'):>10} "
                  f"{row.get('verdict_implausible_hard','?'):>10} "
                  f"{row.get('verdict_implausible_easy','?'):>10} "
                  f"{visual_flag:>16} "
                  f"{scene_flag:>16}")
        else:
            print(f"{row['verb']:<12} {'FAIL':>8}")

    total_strict = sum(r.get("score_strict", 0) for r in all_rows if r["eval_success"])
    total_verbs = sum(1 for r in all_rows if r["eval_success"])
    if total_verbs:
        print(f"\nOverall: {total_strict}/{total_verbs*4} conditions correct "
              f"({total_strict/(total_verbs*4)*100:.1f}%) across {total_verbs} verbs")


if __name__ == "__main__":
    main()
