#!/usr/bin/env python3
"""
Run an Events generation experiment.

Examples:
  # Single verb, local vLLM (Qwen3-30B on port 8000)
  python scripts/run_experiment.py \
    --provider vllm --model Qwen3-30B-A3B \
    --port 8000 --verbs catch

  # All verbs from file, OpenAI GPT-5
  python scripts/run_experiment.py \
    --provider openai --model gpt-5.1 \
    --verb-file data/verbs.txt --temperature 0.0

  # Patient variant
  python scripts/run_experiment.py \
    --provider openai --model gpt-5.1-2025-11-13 \
    --prompt prompts/v7_patient.json \
    --verbs eat write
"""

import os
import sys
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.generator import (
    generate_events,
    detect_varying_slot,
    ALL_SLOTS,
    CONDITIONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run Events generation experiment")

    # Model / provider
    p.add_argument("--provider", required=True, choices=["vllm", "openai", "claude"],
                   help="Model provider")
    p.add_argument("--model", required=True, help="Model name (e.g., Qwen3-30B-A3B, gpt-4o)")
    p.add_argument("--port", type=int, default=8000, help="vLLM server port (only for vllm provider)")

    # Prompt
    p.add_argument("--prompt", default="prompts/v4.json", help="Path to prompt JSON file")

    # Verb input (one of these)
    p.add_argument("--verbs", nargs="+", default=None, help="One or more verbs (space-separated)")
    p.add_argument("--verb-file", default=None, help="Path to file with one verb per line")

    # Generation params
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=None, dest="top_p",
                   help="Nucleus sampling cutoff. Omit to leave the server default.")
    p.add_argument("--top-k", type=int, default=None, dest="top_k",
                   help="Top-k cutoff for local vLLM.")
    p.add_argument("--seed", type=int, default=None,
                   help="Per-request seed. Omit to leave sampling unseeded.")
    p.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
    p.add_argument("--think-budget", type=int, default=0,
                   help="Thinking budget tokens for supported vLLM and Claude models")
    p.add_argument("--reasoning-effort", default="", choices=["", "low", "medium", "high", "xhigh"],
                   help="Reasoning effort for OpenAI GPT-5 models (leave empty to disable; xhigh for GPT-5.2)")
    p.add_argument("--thinking-effort", default="high",
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="Thinking effort for Claude models (default: high)")

    # Experiment control
    p.add_argument("--run-name", default=None,
                   help="Experiment name (default: auto-generated from model+timestamp)")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent requests to the model server (1 keeps the original sequential loop)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the verb list into this many disjoint shards")
    p.add_argument("--shard", type=int, default=0,
                   help="Which shard to run (0 .. num-shards-1). Verb i goes to shard i %% num-shards")

    return p.parse_args()


# -------------------------------------------------------------------------
# Load verbs
# -------------------------------------------------------------------------
def load_verbs(args) -> list:
    if args.verbs:
        verbs = [v.strip().lower() for v in args.verbs]
    elif args.verb_file:
        with open(args.verb_file) as f:
            verbs = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]
    else:
        print("ERROR: Provide --verbs or --verb-file")
        sys.exit(1)

    n = args.num_shards
    if n < 1:
        print("ERROR: --num-shards must be >= 1")
        sys.exit(1)
    if not (0 <= args.shard < n):
        print(f"ERROR: --shard must be in 0..{n - 1}")
        sys.exit(1)
    if n == 1:
        return verbs
    return [verb for i, verb in enumerate(verbs) if i % n == args.shard]


# -------------------------------------------------------------------------
# Varying-slot inference from prompt path (fallback for failed runs)
# -------------------------------------------------------------------------
def infer_varying_slot_from_prompt(prompt_path: str) -> str:
    """
    Best-effort inference of the varying slot from the prompt filename.
    Used when generation fails and we have no parsed output to inspect.
    Returns 'unknown' if no slot name is recognizable in the filename.
    """
    name = os.path.basename(prompt_path).lower()
    # Prefer slot names that appear as a token in the filename
    for slot in ALL_SLOTS:
        if f"_{slot}" in name or name.startswith(f"v7_{slot}") or f"{slot}_" in name:
            return slot
    return "unknown"


# -------------------------------------------------------------------------
# Build CSV row (slot-aware)
# -------------------------------------------------------------------------
def build_row(args, safe_run: str, verb: str, result: dict) -> dict:
    """
    Construct a CSV row from a generation result.
    Detects which slot varied (from parsed output if success, from prompt path if fail)
    and emits the per-condition value column under the matching name.
    """
    row = {
        "run_name": safe_run,
        "model": args.model,
        "provider": args.provider,
        "prompt_version": os.path.basename(args.prompt).replace(".json", ""),
        # Reasoning models reject temperature, so the CSV records it only when
        # the request actually sends it.
        "temperature": "" if args.reasoning_effort else args.temperature,
        "top_p": "" if args.top_p is None else args.top_p,
        "seed": "" if args.seed is None else args.seed,
        "reasoning_effort": args.reasoning_effort,
        "verb": verb,
        "success": result["success"],
        "elapsed_s": round(result["elapsed"], 2),
        "input_tokens": result["token_info"].get("input_tokens", 0),
        "output_tokens": result["token_info"].get("output_tokens", 0),
        "validation_errors": "; ".join(result["validation_errors"]) if result["validation_errors"] else "",
    }

    parsed = result["parsed"] if result["success"] and result["parsed"] else None
    row["schema"] = "v7"

    # Determine the varying slot: prefer parsed output, fall back to prompt path
    varying_block = None
    varying_key = None
    if parsed:
        try:
            varying_block, varying_key = detect_varying_slot(parsed)
        except ValueError as e:
            logger.warning(f"Could not detect varying slot for '{verb}' from parsed output: {e}")

    if varying_key is None:
        varying_key = infer_varying_slot_from_prompt(args.prompt)

    row["varying_slot"] = varying_key

    # Always emit verb_form and the four ALL_SLOTS columns (varying slot left empty)
    row["verb_form"] = ""
    for slot in ALL_SLOTS:
        row[slot] = ""

    for cond in CONDITIONS:
        row[f"sentence_{cond}"] = ""
        row[f"rationale_{cond}"] = ""
        if varying_key in ALL_SLOTS:
            row[f"{varying_key}_{cond}"] = ""

    if parsed:
        p = parsed
        scene = p.get("scene", {})
        row["verb_form"] = scene.get("verb_form", "")

        # Fixed slots come from scene; varying slot is left blank at scene level
        for slot in ALL_SLOTS:
            if slot != varying_key:
                row[slot] = scene.get(slot, "")

        # Per-condition: sentence, varying value, rationale
        sentences = p.get("sentences", {})
        if varying_block and varying_block in p:
            varying_data = p[varying_block]
            for cond in CONDITIONS:
                row[f"sentence_{cond}"] = sentences.get(cond, "")
                entry = varying_data.get(cond, {}) or {}
                if varying_key in ALL_SLOTS:
                    row[f"{varying_key}_{cond}"] = entry.get(varying_key, "")
                row[f"rationale_{cond}"] = entry.get("rationale", "")

    return row


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    verbs = load_verbs(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model}_t{args.temperature}_{ts}"
    safe_run = run_name.replace("/", "_").replace(" ", "_")

    # Directories
    inter_dir = os.path.join("results", "intermediate", safe_run)
    agg_dir = os.path.join("results", "aggregate")
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(agg_dir, exist_ok=True)

    agg_csv = os.path.join(agg_dir, "all_results.csv")

    logger.info(f"Run: {safe_run}")
    logger.info(
        f"Provider: {args.provider} | Model: {args.model} | "
        f"Temp: {args.temperature} | top_p: {args.top_p} | Seed: {args.seed} | "
        f"Reasoning: {args.reasoning_effort or 'off'}"
    )
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Verbs ({len(verbs)}): {verbs}")
    if args.num_shards > 1:
        logger.info(f"Shard: {args.shard}/{args.num_shards}")
    logger.info(f"Workers: {args.workers} | Max tokens: {args.max_tokens}")

    if not verbs:
        logger.info("No verbs in this shard. Nothing to do.")
        return

    def generate_one(i, verb):
        logger.info(f"[{i}/{len(verbs)}] Generating for verb: '{verb}'")
        result = generate_events(
            verb=verb,
            prompt_path=args.prompt,
            provider=args.provider,
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_tokens=args.max_tokens,
            port=args.port,
            think_budget=args.think_budget,
            reasoning_effort=args.reasoning_effort,
            thinking_effort=args.thinking_effort,
            top_k=args.top_k,
        )
        out_path = os.path.join(inter_dir, f"{verb}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"  Saved: {out_path} | Success: {result['success']} | {result['elapsed']:.2f}s")
        return i, build_row(args, safe_run, verb, result)

    # Concurrent requests share one vLLM server. workers=1 keeps one request
    # in flight, which is what every earlier experiment actually ran.
    results_by_index = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(generate_one, i, verb) for i, verb in enumerate(verbs, 1)]
        for future in as_completed(futures):
            i, row = future.result()
            results_by_index[i] = row
    all_results = [results_by_index[i] for i in range(1, len(verbs) + 1)]

    # Write / append to aggregate CSV (replacing any existing rows for same run+verb).
    # Note: append_or_replace_csv in utils/common.py is column-safe — it builds the
    # union of existing and new columns and pads each row to the union schema, so
    # writing patient-variant rows into a CSV with existing agent-variant rows
    # preserves all prior data and adds the new patient_<cond> columns at the end.
    if all_results:
        from utils.common import append_or_replace_csv
        append_or_replace_csv(agg_csv, all_results, key_columns=["run_name", "verb"])

    # Summary
    successes = sum(1 for r in all_results if r["success"])
    logger.info(f"\nDone. {successes}/{len(all_results)} verbs generated successfully.")


if __name__ == "__main__":
    main()