# scripts/run_iterative.py
"""
Iterative refinement orchestrator for Events task.

Pipeline: Generate -> Evaluate -> Refine -> Evaluate -> ... until pass or max iterations.

Examples:
  # Single verb with GPT-5.2
  python scripts/run_iterative.py \
    --provider openai --model gpt-5.2 \
    --eval-provider openai --eval-model gpt-5.2 --eval-reasoning xhigh \
    --verbs catch

  # Multiple verbs, local generator + API evaluator
  python scripts/run_iterative.py \
    --provider vllm --model Qwen3-30B-A3B --port 8000 \
    --eval-provider openai --eval-model gpt-5.2 --eval-reasoning high \
    --verb-file data/verbs.txt --max-iterations 3
"""

import os
import sys
import json
import csv
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.generator import generate_events
from inference.evaluator import evaluate_events
from inference.refiner import refine_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fix A: Safe nested dict accessor
# ---------------------------------------------------------------------------
def safe_get(d, *keys, default=None):
    """Safely traverse nested dicts. Returns default if any key is missing or value is None."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return default
        if d is None:
            return default
    return d


# ---------------------------------------------------------------------------
# Fix C: Evaluation with one retry on parse failure
# ---------------------------------------------------------------------------
def evaluate_with_retry(verb, generated_output, args, max_eval_retries=1):
    """Run evaluation, retrying once if JSON parsing fails (flaky thinking model output)."""
    for attempt in range(max_eval_retries + 1):
        eval_result = evaluate_events(
            verb=verb,
            generated_output=generated_output,
            evaluator_prompt_path=args.eval_prompt,
            provider=args.eval_provider,
            model_name=args.eval_model,
            temperature=args.eval_temperature,
            max_tokens=args.eval_max_tokens,
            port=args.eval_port,
            reasoning_effort=args.eval_reasoning,
            think_budget=args.eval_think_budget,
            criterion=args.criterion,
        )
        if eval_result["success"]:
            return eval_result

        # Check if it was a JSON parse failure (worth retrying) vs structural error (not worth)
        errors = eval_result.get("validation_errors", [])
        is_parse_failure = any("JSON parse" in e or "Generation failed" in e or "call failed" in e
                               for e in errors)

        if is_parse_failure and attempt < max_eval_retries:
            logger.warning(f"  Evaluation parse failed for '{verb}', retrying ({attempt+1}/{max_eval_retries})...")
            continue
        else:
            break

    return eval_result


# ---------------------------------------------------------------------------
# Fix C: Generation with one retry on parse failure
# ---------------------------------------------------------------------------
def generate_with_retry(verb, args, max_gen_retries=1):
    """Run generation, retrying once if JSON parsing fails."""
    for attempt in range(max_gen_retries + 1):
        gen_result = generate_events(
            verb=verb,
            prompt_path=args.gen_prompt,
            provider=args.provider,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            port=args.port,
            reasoning_effort=args.reasoning_effort,
            think_budget=args.think_budget,
        )
        if gen_result["success"]:
            return gen_result

        errors = gen_result.get("validation_errors", [])
        is_parse_failure = any("JSON parse" in e or "Generation failed" in e for e in errors)

        if is_parse_failure and attempt < max_gen_retries:
            logger.warning(f"  Generation parse failed for '{verb}', retrying ({attempt+1}/{max_gen_retries})...")
            continue
        else:
            break

    return gen_result


# ---------------------------------------------------------------------------
# Fix C: Refinement with one retry on parse failure
# ---------------------------------------------------------------------------
def refine_with_retry(verb, current_output, feedback, args, max_refine_retries=1):
    """Run refinement, retrying once if JSON parsing fails."""
    for attempt in range(max_refine_retries + 1):
        refine_result = refine_events(
            verb=verb,
            previous_output=current_output,
            evaluator_feedback=feedback,
            refine_prompt_path=args.refine_prompt,
            provider=args.provider,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            port=args.port,
            reasoning_effort=args.reasoning_effort,
            think_budget=args.think_budget,
        )
        if refine_result["success"]:
            return refine_result

        errors = refine_result.get("validation_errors", [])
        is_parse_failure = any("JSON parse" in e or "Refinement call failed" in e or "may not contain" in e for e in errors)

        if is_parse_failure and attempt < max_refine_retries:
            logger.warning(f"  Refinement parse failed for '{verb}', retrying ({attempt+1}/{max_refine_retries})...")
            continue
        else:
            break

    return refine_result

# ---------------------------------------------------------------------------
# Standardized YAML-format feedback for the refinement prompt
# ---------------------------------------------------------------------------

CONDITIONS_ORDERED = ["plausible_easy", "plausible_hard", "implausible_hard", "implausible_easy"]

OVERLAP_DIMS = ["occupational_domain", "typical_setting", "tools_instruments",
                "patient_object_type", "physical_actions"]


def _get_agents_from_output(output):
    """Extract agent names per condition from a generation/refinement output."""
    agents = {}
    if not output or not isinstance(output, dict):
        return agents
    for cond in CONDITIONS_ORDERED:
        agent_obj = safe_get(output, "agents", cond, default={})
        if isinstance(agent_obj, dict):
            agents[cond] = agent_obj.get("agent", "?")
        else:
            agents[cond] = "?"
    return agents

def _format_single_iteration_feedback(iteration_idx, output, eval_result, criterion="visual"):
    """Format one iteration's feedback in the standardized YAML schema.
    
    Args:
        iteration_idx: 0-based iteration number
        output: the generated/refined output dict for this iteration
        eval_result: the evaluation result dict for this iteration (may be None or failed)
    
    Returns:
        string in standardized YAML feedback format
    """
    lines = [f"=== ITERATION {iteration_idx} FEEDBACK ===", ""]

    agents = _get_agents_from_output(output)

    # Handle failed or missing evaluation
    if not eval_result or not eval_result.get("success") or not eval_result.get("evaluation"):
        lines.append("EVALUATION: FAILED")
        errors = []
        if eval_result:
            errors = eval_result.get("validation_errors", [])
        lines.append(f"  errors: {'; '.join(errors) if errors else 'Unknown evaluation failure'}")
        lines.append("")
        # Still list the agents that were tried
        lines.append("AGENTS ATTEMPTED:")
        for cond in CONDITIONS_ORDERED:
            lines.append(f"  {cond}: \"{agents.get(cond, '?')}\"")
        return "\n".join(lines)

    ev = eval_result["evaluation"]

    # --- SCENE ---
    scene = ev.get("scene_assessment", {}) or {}
    scene_verdict = scene.get("scene_verdict", "UNKNOWN")
    scene_notes = scene.get("scene_notes", "")
    instrument_ok = scene.get("instrument_correct", True)
    prototype_ok = scene.get("has_clear_prototype", True)
    gradient_supportable = scene.get("gradient_supportable", True)

    lines.append("SCENE:")
    lines.append(f"  verdict: {scene_verdict}")

    scene_issues = []
    if not instrument_ok:
        scene_issues.append("Instrument is incorrect for the verb.")
    if not prototype_ok:
        scene_issues.append("No clear prototypical agent exists for this scene.")
    if not gradient_supportable:
        reasoning = scene.get("gradient_support_reasoning", "")
        scene_issues.append(f"Scene cannot support full plausibility gradient. {reasoning}")
    if scene_verdict == "NEEDS_REDESIGN":
        scene_issues.append(scene_notes)

    lines.append(f"  issues: {' '.join(scene_issues).strip() if scene_issues else ''}")
    lines.append("")

    # --- AGENTS ---
    conditions = ev.get("condition_evaluations", {}) or {}
    lines.append("AGENTS:")
    for cond in CONDITIONS_ORDERED:
        ce = conditions.get(cond, {})
        if not isinstance(ce, dict):
            ce = {}

        agent = ce.get("agent", agents.get(cond, "?"))
        verdict = ce.get("verdict", "UNKNOWN")
        actual = ce.get("actual_condition", cond)
        reasoning = ce.get("reasoning", "")

        lines.append(f"  {cond}:")
        lines.append(f"    agent: \"{agent}\"")
        lines.append(f"    verdict: {verdict}")
        lines.append(f"    actual_condition: {actual}")
        lines.append(f"    issues: {reasoning if verdict != 'CORRECT' else ''}")

        # Counter-arguments (v3)
        counter_args = ce.get("counter_arguments", {})
        if isinstance(counter_args, dict) and counter_args:
            arg_higher = counter_args.get("argument_for_higher", "")
            arg_lower = counter_args.get("argument_for_lower", "")
            if verdict != "CORRECT":
                if arg_higher and arg_higher != "N/A":
                    lines.append(f"    argument_for_higher: {arg_higher}")
                if arg_lower and arg_lower != "N/A":
                    lines.append(f"    argument_for_lower: {arg_lower}")

        # Overlap audit (v3, implausible_hard only)
        if cond == "implausible_hard":
            oa = ce.get("overlap_audit", {})
            if isinstance(oa, dict) and oa:
                overlap_count = oa.get("overlap_count", "")
                sufficient = oa.get("sufficient", "")
                lines.append(f"    overlap_count: {overlap_count}")
                lines.append(f"    overlap_sufficient: {'YES' if sufficient is True else 'NO' if sufficient is False else ''}")

                details_parts = []
                for dim in OVERLAP_DIMS:
                    dim_data = oa.get(dim, {})
                    if isinstance(dim_data, dict):
                        overlaps = dim_data.get("overlaps", False)
                        dim_reason = dim_data.get("reasoning", "")
                        status = "YES" if overlaps else "NO"
                        details_parts.append(f"{dim}={status} ({dim_reason})")
                lines.append(f"    overlap_details: {'; '.join(details_parts) if details_parts else ''}")

                neg_test = oa.get("negative_test", "")
                if neg_test:
                    lines.append(f"    negative_test: {neg_test}")
            else:
                lines.append(f"    overlap_count: ")
                lines.append(f"    overlap_sufficient: ")
                lines.append(f"    overlap_details: ")

    lines.append("")

    # --- GRADIENT ---
    gradient = ev.get("gradient_ordering", {}) or {}
    gradient_ok = gradient.get("correctly_ordered", True)
    swapped = gradient.get("swapped_pairs", "None")
    collapsed = gradient.get("collapsed_pairs", "None")

    lines.append("GRADIENT:")
    lines.append(f"  correctly_ordered: {'YES' if gradient_ok else 'NO'}")

    gradient_issues = []
    if not gradient_ok:
        if swapped and swapped != "None":
            gradient_issues.append(f"Swapped pairs: {swapped}")
        if collapsed and collapsed != "None":
            gradient_issues.append(f"Collapsed pairs: {collapsed}")
    lines.append(f"  issues: {' '.join(gradient_issues) if gradient_issues else ''}")
    lines.append("")

    if criterion == "text":
        return "\n".join(lines).rstrip()

    # --- IDENTIFIABILITY (v3) ---
    visual = ev.get("visual_evaluation", {}) or {}
    ident = visual.get("identifiability", {})

    lines.append("IDENTIFIABILITY:")
    ident_issues = []
    if isinstance(ident, dict) and ident:
        for cond in CONDITIONS_ORDERED:
            ic = ident.get(cond, {})
            if isinstance(ic, dict):
                rating = ic.get("rating", "")
                reason = ic.get("reasoning", "")
                lines.append(f"  {cond}: {rating}")
                if rating in ("UNIDENTIFIABLE", "AMBIGUOUS"):
                    ident_issues.append(f"{cond} (\"{agents.get(cond, '?')}\"): {rating} - {reason}")
            else:
                lines.append(f"  {cond}: ")
    else:
        for cond in CONDITIONS_ORDERED:
            lines.append(f"  {cond}: ")
    lines.append(f"  issues: {'; '.join(ident_issues) if ident_issues else ''}")
    lines.append("")

    # --- VISUAL DISTINCTIVENESS ---
    visual_verdict = visual.get("visual_verdict", "UNKNOWN")
    indistinguishable = visual.get("indistinguishable_pairs", "None")

    lines.append("VISUAL DISTINCTIVENESS:")
    lines.append(f"  verdict: {visual_verdict}")

    visual_issues = []
    if visual_verdict != "ALL_DISTINCT":
        if indistinguishable and indistinguishable != "None":
            visual_issues.append(f"Indistinguishable pairs: {indistinguishable}")
        pairwise = visual.get("pairwise_checks", {})
        if isinstance(pairwise, dict):
            for pair_key, pair_val in pairwise.items():
                if isinstance(pair_val, dict) and not pair_val.get("distinguishable", True):
                    visual_issues.append(f"{pair_key}: {pair_val.get('note', 'too similar')}")
    lines.append(f"  issues: {'; '.join(visual_issues) if visual_issues else ''}")

    return "\n".join(lines)


def format_feedback_for_refiner(iteration_history, criterion="visual"):
    """Format the full iteration history into standardized YAML feedback.
    
    Args:
        iteration_history: list of dicts, each with:
            - "iteration": int (0-based)
            - "output": the generated/refined output dict
            - "eval_result": the evaluation result dict (may be None)
    
    Returns:
        string containing all iterations' feedback in standardized format
    """
    if not iteration_history:
        return "No previous iteration data available. Please regenerate from scratch."

    blocks = []
    for entry in iteration_history:
        idx = entry.get("iteration", 0)
        output = entry.get("output")
        eval_result = entry.get("eval_result")
        blocks.append(_format_single_iteration_feedback(
            idx, output, eval_result, criterion=criterion
        ))

    return "\n\n".join(blocks)



def parse_args():
    p = argparse.ArgumentParser(description="Iterative refinement for Events generation")

    # Generator model
    p.add_argument("--provider", required=True, choices=["vllm", "openai", "claude"])
    p.add_argument("--model", required=True, help="Generator model name")
    p.add_argument("--port", type=int, default=8000, help="vLLM port")
    p.add_argument("--temperature", type=float, default=0.3, help="Generator temperature")
    p.add_argument("--max-tokens", type=int, default=2048, help="Generator max tokens")
    p.add_argument("--reasoning-effort", default="", choices=["", "low", "medium", "high", "xhigh"])
    p.add_argument("--think-budget", type=int, default=0)

    # Evaluator model
    p.add_argument("--eval-provider", required=True, choices=["openai", "vllm", "claude"])
    p.add_argument("--eval-model", required=True, help="Evaluator model name")
    p.add_argument("--eval-port", type=int, default=8000)
    p.add_argument("--eval-reasoning", default="high", choices=["", "low", "medium", "high", "xhigh"])
    p.add_argument("--eval-think-budget", type=int, default=8000)
    p.add_argument("--eval-max-tokens", type=int, default=24576),
    p.add_argument("--eval-temperature", type=float, default=0.0)

    # Prompts
    p.add_argument("--gen-prompt", default="prompts/v7_agent.json", help="Generator prompt")
    p.add_argument("--eval-prompt", default="prompts/evaluator_v3.1_agent.json", help="Evaluator prompt")
    p.add_argument("--refine-prompt", default="prompts/v7_agent_iterative.json", help="Refinement prompt")
    p.add_argument(
        "--criterion",
        choices=["visual", "text"],
        default="visual",
        help=("Stopping/evaluation criterion. 'text' drops only the visual "
              "requirement; default preserves historical behavior."),
    )

    # Verb input
    p.add_argument("--verbs", nargs="+", default=None)
    p.add_argument("--verb-file", default=None)

    # Iteration control
    p.add_argument("--max-iterations", type=int, default=3,
                   help="Maximum refinement iterations per verb (0 = generate + evaluate only)")
    p.add_argument("--stop-on-pass", action="store_true", default=True,
                   help="Stop iterating when all requirements for --criterion pass")

    # Output
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--aggregate-csv",
        default=None,
        help=("Aggregate CSV destination. Defaults to all_iterative_results.csv for "
              "visual runs and all_iterative_results_text.csv for text-only runs."),
    )

    return p.parse_args()


def load_verbs(args):
    if args.verbs:
        return [v.strip().lower() for v in args.verbs]
    if args.verb_file:
        with open(args.verb_file) as f:
            return [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]
    logger.error("Provide --verbs or --verb-file")
    sys.exit(1)


def check_pass(eval_result, criterion="visual"):
    """Check if evaluation passes all criteria."""
    if criterion not in {"visual", "text"}:
        raise ValueError(f"Unknown criterion: {criterion}")
    if not eval_result.get("success"):
        return False

    score = safe_get(eval_result, "scores", "score_strict", default=0)
    if score < 4:
        return False

    if criterion == "visual":
        visual_verdict = safe_get(
            eval_result, "evaluation", "visual_evaluation", "visual_verdict", default=""
        )
        if visual_verdict != "ALL_DISTINCT":
            return False

    scene_verdict = safe_get(eval_result, "evaluation", "scene_assessment", "scene_verdict", default="")
    if criterion == "text":
        # Text stopping and text GOLD deliberately use the same scene bar.
        if scene_verdict != "GOOD":
            return False
    elif scene_verdict == "NEEDS_REDESIGN":
        # Preserve the historical visual-mode behavior, where a fixable scene
        # may still pass when all other requirements are satisfied.
        return False

    instrument_ok = safe_get(eval_result, "evaluation", "scene_assessment", "instrument_correct", default=True)
    prototype_ok = safe_get(eval_result, "evaluation", "scene_assessment", "has_clear_prototype", default=True)
    gradient_ok = safe_get(eval_result, "evaluation", "scene_assessment", "gradient_supportable", default=True)

    if not instrument_ok or not prototype_ok or not gradient_ok:
        return False

    return True


# ---------------------------------------------------------------------------
# Updated process_verb — accumulates iteration history
# ---------------------------------------------------------------------------
def process_verb(verb, args, run_dir):
    """Process a single verb through the generate -> evaluate -> refine loop."""

    verb_dir = os.path.join(run_dir, verb)
    os.makedirs(verb_dir, exist_ok=True)

    history = {
        "verb": verb,
        "criterion": args.criterion,
        "iterations": [],
        "final_pass": False,
        "total_iterations": 0,
    }

    # Accumulate iteration data for the refiner
    iteration_history = []

    # --- Iteration 0: Initial generation ---
    logger.info(f"  [iter 0] Generating initial output...")
    gen_result = generate_with_retry(verb, args)

    if not gen_result["success"]:
        logger.warning(f"  [iter 0] Generation failed: {gen_result['validation_errors']}")
        history["iterations"].append({"iteration": 0, "stage": "generation", "success": False,
                                       "errors": gen_result["validation_errors"]})
        save_json(os.path.join(verb_dir, "history.json"), history)
        return history

    current_output = gen_result["parsed"]
    save_json(os.path.join(verb_dir, "iter_0_generation.json"), gen_result)

    # Explicit method export: preserve iteration 0 as the RB baseline.
    rb_dir = os.path.join(run_dir, "_rb")
    os.makedirs(rb_dir, exist_ok=True)
    save_json(os.path.join(rb_dir, f"{verb}.json"), {
        "verb": verb, "success": True, "parsed": current_output,
        "method": "RB", "criterion": args.criterion,
    })

    # --- Evaluate ---
    logger.info(f"  [iter 0] Evaluating...")
    eval_result = evaluate_with_retry(verb, current_output, args)
    save_json(os.path.join(verb_dir, "iter_0_evaluation.json"), eval_result)

    # Track this iteration
    iteration_history.append({
        "iteration": 0,
        "output": current_output,
        "eval_result": eval_result,
    })

    # Handle evaluation failure
    if not eval_result["success"]:
        logger.warning(f"  [iter 0] Evaluation failed: {eval_result['validation_errors']}")
        history["iterations"].append({
            "iteration": 0, "stage": "evaluation", "success": False,
            "errors": eval_result["validation_errors"],
            "gen_elapsed": gen_result["elapsed"],
            "eval_elapsed": eval_result["elapsed"],
        })
        history["final_pass"] = False
        history["total_iterations"] = 1
        history["final_iteration"] = 0
        save_json(os.path.join(verb_dir, "final_output.json"), current_output)
        save_json(os.path.join(verb_dir, "history.json"), history)
        r2r_dir = os.path.join(run_dir, "_r2r")
        os.makedirs(r2r_dir, exist_ok=True)
        save_json(os.path.join(r2r_dir, f"{verb}.json"), {
            "verb": verb, "success": True, "parsed": current_output,
            "method": "R2R", "criterion": args.criterion,
        })
        return history

    passed = check_pass(eval_result, criterion=args.criterion)
    score = safe_get(eval_result, "scores", "score_strict", default="?")
    score_lenient = safe_get(eval_result, "scores", "score_lenient", default="?")
    visual = safe_get(eval_result, "evaluation", "visual_evaluation", "visual_verdict", default="?")
    scene_v = safe_get(eval_result, "evaluation", "scene_assessment", "scene_verdict", default="?")
    v_pe = safe_get(eval_result, "evaluation", "condition_evaluations", "plausible_easy", "verdict", default="?")
    v_ph = safe_get(eval_result, "evaluation", "condition_evaluations", "plausible_hard", "verdict", default="?")
    v_ih = safe_get(eval_result, "evaluation", "condition_evaluations", "implausible_hard", "verdict", default="?")
    v_ie = safe_get(eval_result, "evaluation", "condition_evaluations", "implausible_easy", "verdict", default="?")
    status_bits = [f"Score: {score}/4", f"Scene: {scene_v}"]
    if args.criterion == "visual":
        status_bits.insert(1, f"Visual: {visual}")
    logger.info(f"  [iter 0] {' | '.join(status_bits)} | Pass: {passed}")

    history["iterations"].append({
        "iteration": 0,
        "score_strict": score,
        "score_lenient": score_lenient,
        "visual_verdict": visual,
        "scene_verdict": scene_v,
        "verdict_plausible_easy": v_pe,
        "verdict_plausible_hard": v_ph,
        "verdict_implausible_hard": v_ih,
        "verdict_implausible_easy": v_ie,
        "passed": passed,
        "gen_elapsed": gen_result["elapsed"],
        "eval_elapsed": eval_result["elapsed"],
    })

    # --- Refinement loop ---
    for iteration in range(1, args.max_iterations + 1):
        if passed and args.stop_on_pass:
            logger.info(f"  Passed at iteration {iteration - 1}. Stopping.")
            break

        logger.info(f"  [iter {iteration}] Refining...")

        # Format feedback from full history
        feedback = format_feedback_for_refiner(iteration_history, criterion=args.criterion)

        logger.info(f"  [iter {iteration}] Feedback:\n{feedback}")

        # Refine
        refine_result = refine_with_retry(verb, current_output, feedback, args)
        save_json(os.path.join(verb_dir, f"iter_{iteration}_refinement.json"), refine_result)

        if not refine_result["success"]:
            logger.warning(f"  [iter {iteration}] Refinement failed: {refine_result['validation_errors']}")
            history["iterations"].append({"iteration": iteration, "stage": "refinement",
                                           "success": False, "errors": refine_result["validation_errors"]})
            break

        current_output = refine_result["parsed"]

        # Surface the chosen strategy and high-level changes for at-a-glance run review
        strategy = current_output.get("revision_strategy", "UNKNOWN")
        changes = current_output.get("changes_made", "")
        logger.info(f"  [iter {iteration}] Strategy={strategy} | Changes: {changes[:200]}")

        # Re-evaluate
        logger.info(f"  [iter {iteration}] Re-evaluating...")
        eval_result = evaluate_with_retry(verb, current_output, args)
        save_json(os.path.join(verb_dir, f"iter_{iteration}_evaluation.json"), eval_result)

        # Track this iteration
        iteration_history.append({
            "iteration": iteration,
            "output": current_output,
            "eval_result": eval_result,
        })

        # Handle evaluation failure
        if not eval_result["success"]:
            logger.warning(f"  [iter {iteration}] Evaluation failed: {eval_result['validation_errors']}")
            history["iterations"].append({
                "iteration": iteration, "stage": "evaluation", "success": False,
                "errors": eval_result["validation_errors"],
                "refine_elapsed": refine_result["elapsed"],
                "eval_elapsed": eval_result["elapsed"],
            })
            # Continue to next iteration — feedback will show this eval as FAILED
            logger.info(f"  [iter {iteration}] Continuing with eval failure recorded in history")
            continue

        passed = check_pass(eval_result, criterion=args.criterion)
        score = safe_get(eval_result, "scores", "score_strict", default="?")
        visual = safe_get(eval_result, "evaluation", "visual_evaluation", "visual_verdict", default="?")
        scene_v = safe_get(eval_result, "evaluation", "scene_assessment", "scene_verdict", default="?")
        status_bits = [f"Score: {score}/4", f"Scene: {scene_v}"]
        if args.criterion == "visual":
            status_bits.insert(1, f"Visual: {visual}")
        logger.info(f"  [iter {iteration}] {' | '.join(status_bits)} | Pass: {passed}")

        history["iterations"].append({
            "iteration": iteration,
            "score_strict": score,
            "visual_verdict": visual,
            "scene_verdict": scene_v,
            "passed": passed,
            "refine_elapsed": refine_result["elapsed"],
            "eval_elapsed": eval_result["elapsed"],
            "changes_made": changes,
            "revision_strategy": strategy,
        })

    history["final_pass"] = passed
    history["total_iterations"] = len(history["iterations"])
    history["final_iteration"] = iteration_history[-1]["iteration"]

    # Save final output
    save_json(os.path.join(verb_dir, "final_output.json"), current_output)
    save_json(os.path.join(verb_dir, "history.json"), history)

    # Save wrapped final output for run_evaluator.py compatibility
    wrapped_dir = os.path.join(run_dir, "_finals")
    os.makedirs(wrapped_dir, exist_ok=True)
    wrapped = {"verb": verb, "success": True, "parsed": current_output}
    save_json(os.path.join(wrapped_dir, f"{verb}.json"), wrapped)

    # Explicit R2R export is kept separate from the historical _finals export.
    r2r_dir = os.path.join(run_dir, "_r2r")
    os.makedirs(r2r_dir, exist_ok=True)
    r2r_wrapped = {
        "verb": verb, "success": True, "parsed": current_output,
        "method": "R2R", "criterion": args.criterion,
    }
    save_json(os.path.join(r2r_dir, f"{verb}.json"), r2r_wrapped)

    return history

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    args = parse_args()
    verbs = load_verbs(args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    criterion_tag = "" if args.criterion == "visual" else "_text"
    run_name = args.run_name or (
        f"iterative_{args.model}_t{args.temperature}{criterion_tag}_{ts}"
    )
    safe_run = run_name.replace("/", "_").replace(" ", "_")

    run_dir = os.path.join("results", "iterative", safe_run)
    agg_dir = os.path.join("results", "aggregate")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(agg_dir, exist_ok=True)

    logger.info(f"Run: {safe_run}")
    logger.info(f"Generator: {args.provider} | {args.model} | temp={args.temperature}")
    logger.info(f"Evaluator: {args.eval_provider} | {args.eval_model} | reasoning={args.eval_reasoning}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Criterion: {args.criterion}")
    logger.info(f"Verbs: {verbs}")

    # Save run config
    save_json(os.path.join(run_dir, "config.json"), vars(args))

    all_rows = []

    for i, verb in enumerate(verbs, 1):
        logger.info(f"\n[{i}/{len(verbs)}] Processing verb: '{verb}'")

        history = process_verb(verb, args, run_dir)

        # Build CSV row
        last_iter = history["iterations"][-1] if history["iterations"] else {}
        row = {
            "run_name": safe_run,
            "generator_model": args.model,
            "evaluator_model": args.eval_model,
            "criterion": args.criterion,
            "verb": verb,
            "final_pass": history["final_pass"],
            "total_iterations": history["total_iterations"],
            "final_score": last_iter.get("score_strict", ""),
            "final_score_lenient": last_iter.get("score_lenient", ""),
            "final_visual": last_iter.get("visual_verdict", ""),
            "final_scene": last_iter.get("scene_verdict", ""),
            "verdict_plausible_easy": last_iter.get("verdict_plausible_easy", ""),
            "verdict_plausible_hard": last_iter.get("verdict_plausible_hard", ""),
            "verdict_implausible_hard": last_iter.get("verdict_implausible_hard", ""),
            "verdict_implausible_easy": last_iter.get("verdict_implausible_easy", ""),
        }
        # Add per-iteration scores
        for it in history["iterations"]:
            idx = it.get("iteration", "?")
            row[f"iter_{idx}_score"] = it.get("score_strict", "")
            row[f"iter_{idx}_visual"] = it.get("visual_verdict", "")
            row[f"iter_{idx}_strategy"] = it.get("revision_strategy", "")

        all_rows.append(row)

    # Write aggregate CSV (replacing any existing rows for same run+verb)
    iter_csv = args.aggregate_csv or os.path.join(
        agg_dir,
        "all_iterative_results.csv"
        if args.criterion == "visual"
        else "all_iterative_results_text.csv",
    )
    if all_rows:
        from utils.common import append_or_replace_csv
        # Collect all possible column names across rows
        all_keys = []
        seen = set()
        for row in all_rows:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        append_or_replace_csv(iter_csv, all_rows, key_columns=["run_name", "verb"])

    # Print summary
    print(f"\n{'='*70}")
    print(f"ITERATIVE REFINEMENT SUMMARY: {safe_run}")
    print(f"{'='*70}")
    print(f"{'Verb':<12} {'Pass':>6} {'Iters':>6} {'Strict':>8} {'Lenient':>8} {'PE':>10} {'PH':>10} {'IH':>10} {'IE':>10} {'Visual':>16} {'Scene':>16}")
    print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*16} {'-'*16}")

    for row in all_rows:
        status = "YES" if row["final_pass"] else "NO"
        print(f"{row['verb']:<12} {status:>6} {row['total_iterations']:>6} "
              f"{str(row['final_score'])+'/4':>8} "
              f"{str(row['final_score_lenient'])+'/4':>8} "
              f"{str(row.get('verdict_plausible_easy', '?')):>10} "
              f"{str(row.get('verdict_plausible_hard', '?')):>10} "
              f"{str(row.get('verdict_implausible_hard', '?')):>10} "
              f"{str(row.get('verdict_implausible_easy', '?')):>10} "
              f"{str(row['final_visual']):>16} "
              f"{str(row.get('final_scene', '')):>16}")

    passed = sum(1 for r in all_rows if r["final_pass"])
    print(f"\nPassed: {passed}/{len(all_rows)} verbs")


if __name__ == "__main__":
    main()
