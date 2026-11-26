"""
Surprisal-based Evaluator (evalv5) — Sentence Log-Probability

Reads generation result JSONs (same input format as run_evaluator.py) and computes
sentence log-probabilities via vLLM's completions API. Ranks agents by log P
to produce verdicts compatible with the analysis pipeline.

Ranking logic:
  Highest log P → predicted PE, next → PH, next → IH, lowest → IE
  Verdict = CORRECT if predicted condition matches generator assignment.

Output format matches evaluator JSON so existing merge/analysis scripts work:
  python -m analysis.main --eval-prefix evalv5_qwen30b_nothink

Usage:
    python scripts/compute_surprisal.py \
        --input-dir results/intermediate/gpt-5.1-2025-11-13_v6 \
        --model Qwen3-30B-A3B-Instruct-2507 \
        --port 8002 \
        --run-name evalv5_qwen30b_nothink_gpt-5.1-2025-11-13_v6 \
        --verbs eat erase wash
"""

import argparse
import json
import logging
import os
import time
import numpy as np
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COND_ORDER = ["plausible_easy", "plausible_hard", "implausible_hard", "implausible_easy"]
COND_RANK = {c: i for i, c in enumerate(COND_ORDER)}
RANK_TO_COND = {i: c for c, i in COND_RANK.items()}

OUTPUT_BASE = "results/evaluations"


def compute_sentence_logprobs(client, model, sentence):
    """
    Compute token-level log-probabilities for a sentence using completions API.
    Returns: (tokens, token_logprobs, token_info) or (None, None, None) on error.
    """
    try:
        response = client.completions.create(
            model=model,
            prompt=sentence,
            max_tokens=1,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
    except Exception as e:
        logger.error(f"API error: {e}")
        return None, None, None

    choice = response.choices[0]
    tokens = list(choice.logprobs.tokens)
    token_logprobs = list(choice.logprobs.token_logprobs)

    # Remove generated token (last one) — we only want input logprobs
    if len(tokens) > 1:
        input_text = "".join(tokens[:-1])
        if len(input_text.strip()) >= len(sentence.strip()) - 5:
            tokens = tokens[:-1]
            token_logprobs = token_logprobs[:-1]

    usage = response.usage
    token_info = {
        "input_tokens": getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
    }

    return tokens, token_logprobs, token_info


def find_agent_boundary(tokens, agent_text):
    """Find token index where agent phrase ends."""
    reconstructed = ""
    for i, tok in enumerate(tokens):
        reconstructed += tok
        if len(reconstructed.strip()) >= len(agent_text.strip()):
            return i + 1
    return len(tokens)


def load_generation(input_dir, verb):
    """Load a generation JSON file and extract the 4 sentences + agents."""
    path = os.path.join(input_dir, f"{verb}.json")
    if not os.path.exists(path):
        logger.error(f"Generation file not found: {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    gen = data.get("parsed", data)
    sentences = gen.get("sentences", {})
    agents = gen.get("agents", {})

    result = {}
    for cond in COND_ORDER:
        if cond not in sentences:
            logger.warning(f"Missing condition {cond} in {path}")
            continue
        agent_info = agents.get(cond, {})
        agent_name = agent_info.get("agent", "") if isinstance(agent_info, dict) else str(agent_info)
        result[cond] = {
            "sentence": sentences[cond],
            "agent": agent_name,
        }

    if len(result) != 4:
        logger.error(f"Expected 4 conditions, got {len(result)} for {verb}")
        return None

    return result


def rank_by_logp(condition_scores, rank_metric):
    """
    Rank conditions by total_logp (descending).
    Returns dict: generator_condition → predicted_condition.
    """
    sorted_conds = sorted(condition_scores.keys(),
                          key=lambda c: condition_scores[c][rank_metric],
                          reverse=True)
    return {gen_cond: RANK_TO_COND[rank] for rank, gen_cond in enumerate(sorted_conds)}


def build_evaluation_json(verb, condition_scores, predictions):
    """Build evaluator-compatible JSON output from surprisal scores."""
    eval_output = {
        "constraint_checks": {
            "all_human_professionals": True,
            "consistency_maintained": True,
            "role_action_mismatch_valid": True,
            "constraint_notes": "Surprisal-based evaluation (no constraint checking)"
        },
        "condition_evaluations": {},
        "gradient_ordering": {
            "correctly_ordered": True,
            "swapped_pairs": "None",
            "collapsed_pairs": "None"
        },
        "visual_evaluation": {
            "identifiability": {
                cond: {"rating": "SKIPPED", "reasoning": "N/A for surprisal eval"}
                for cond in COND_ORDER
            },
            "agent_appearances": {cond: "N/A" for cond in COND_ORDER},
            "pairwise_checks": {
                f"{a}_vs_{b}": {"distinguishable": True, "note": "N/A"}
                for a, b in [("pe","ph"),("pe","ih"),("pe","ie"),("ph","ih"),("ph","ie"),("ih","ie")]
            },
            "visual_verdict": "MOSTLY_DISTINCT",
            "indistinguishable_pairs": "None",
            "unidentifiable_agents": "None"
        },
        "scene_assessment": {
            "instrument_correct": True,
            "has_clear_prototype": True,
            "imageable": True,
            "gradient_supportable": True,
            "gradient_support_reasoning": "Surprisal-based evaluation",
            "scene_verdict": "GOOD",
            "scene_notes": "Surprisal-based — scene not assessed"
        },
        "overall": {}
    }

    correct_count = 0
    for cond in COND_ORDER:
        scores = condition_scores[cond]
        predicted = predictions[cond]
        is_correct = (predicted == cond)
        if is_correct:
            correct_count += 1

        eval_output["condition_evaluations"][cond] = {
            "agent": scores["agent"],
            "verdict": "CORRECT" if is_correct else "INCORRECT",
            "actual_condition": predicted,
            "reasoning": (
                f"Log P ranking: total={scores['total_logp']:.2f}, "
                f"frame={scores['frame_logp']:.2f}. "
                f"Rank predicts {predicted}."
            ),
            "counter_arguments": {
                "argument_for_higher": "N/A",
                "argument_for_lower": "N/A"
            }
        }

    eval_output["overall"] = {
        "score": f"{correct_count}/4",
        "borderline_count": 0,
        "scene_verdict": "GOOD",
        "visual_verdict": "MOSTLY_DISTINCT",
        "summary": (
            f"Surprisal-based evaluation: {correct_count}/4 conditions match "
            f"log P ranking via sentence log-probability."
        )
    }

    # Check gradient ordering
    logps = [condition_scores[c]["total_logp"] for c in COND_ORDER]
    is_monotonic = all(logps[i] >= logps[i+1] for i in range(3))
    eval_output["gradient_ordering"]["correctly_ordered"] = is_monotonic
    if not is_monotonic:
        swaps = []
        for i in range(3):
            if logps[i] < logps[i+1]:
                swaps.append(f"{COND_ORDER[i]} < {COND_ORDER[i+1]}")
        eval_output["gradient_ordering"]["swapped_pairs"] = "; ".join(swaps)

    return eval_output


def extract_scores(evaluation):
    """Extract scores dict compatible with the merge/analysis pipeline."""
    scores = {
        "correct_count": 0,
        "borderline_count": 0,
        "incorrect_count": 0,
        "has_scene_error": False,
        "scene_verdict": "GOOD",
        "gradient_supportable": True,
        "conditions": {},
        "overlap_count": None,
        "overlap_sufficient": None,
        "overlap_negative_test": None,
        "identifiability": None,
        "unidentifiable_agents": None,
    }

    for cond in COND_ORDER:
        ce = evaluation["condition_evaluations"].get(cond, {})
        verdict = ce.get("verdict", "UNKNOWN")
        scores["conditions"][cond] = {
            "verdict": verdict,
            "actual_condition": ce.get("actual_condition", cond),
        }
        if verdict == "CORRECT":
            scores["correct_count"] += 1
        elif verdict == "INCORRECT":
            scores["incorrect_count"] += 1

    scores["score_strict"] = scores["correct_count"]
    scores["score_lenient"] = float(scores["correct_count"])
    return scores


def process_verb(client, model, input_dir, verb, rank_metric):
    """Process one verb: load generation, compute log P, produce evaluation."""
    t0 = time.perf_counter()

    gen_data = load_generation(input_dir, verb)
    if gen_data is None:
        return None

    condition_scores = {}
    total_tokens = {"input_tokens": 0, "output_tokens": 0}

    for cond in COND_ORDER:
        sentence = gen_data[cond]["sentence"]
        agent = gen_data[cond]["agent"]

        tokens, token_logprobs, token_info = compute_sentence_logprobs(client, model, sentence)
        if tokens is None:
            logger.error(f"Failed to compute logprobs for {verb}/{cond}")
            return None

        total_tokens["input_tokens"] += token_info.get("input_tokens", 0)

        valid_lps = [lp for lp in token_logprobs if lp is not None]
        total_logp = sum(valid_lps)
        mean_logp = np.mean(valid_lps) if valid_lps else 0.0
        perplexity = float(np.exp(-mean_logp)) if valid_lps else float("inf")

        agent_end = find_agent_boundary(tokens, agent)
        agent_lps = [lp for lp in token_logprobs[1:agent_end] if lp is not None]
        frame_lps = [lp for lp in token_logprobs[agent_end:] if lp is not None]

        condition_scores[cond] = {
            "agent": agent,
            "sentence": sentence,
            "total_logp": total_logp,
            "mean_logp": mean_logp,
            "perplexity": perplexity,
            "agent_logp": sum(agent_lps) if agent_lps else 0.0,
            "frame_logp": sum(frame_lps) if frame_lps else 0.0,
            "n_tokens": len(tokens),
            "n_agent_tokens": agent_end,
        }

    predictions = rank_by_logp(condition_scores, rank_metric)
    evaluation = build_evaluation_json(verb, condition_scores, predictions)
    scores = extract_scores(evaluation)
    elapsed = time.perf_counter() - t0

    return {
        "verb": verb,
        "raw_response": "surprisal-based evaluation via sentence log-probability",
        "evaluation": evaluation,
        "validation_errors": [],
        "scores": scores,
        "elapsed": round(elapsed, 2),
        "token_info": total_tokens,
        "success": True,
        "surprisal_data": {
            cond: {
                "total_logp": round(condition_scores[cond]["total_logp"], 4),
                "mean_logp": round(condition_scores[cond]["mean_logp"], 4),
                "perplexity": round(condition_scores[cond]["perplexity"], 4),
                "agent_logp": round(condition_scores[cond]["agent_logp"], 4),
                "frame_logp": round(condition_scores[cond]["frame_logp"], 4),
                "n_tokens": condition_scores[cond]["n_tokens"],
                "n_agent_tokens": condition_scores[cond]["n_agent_tokens"],
            }
            for cond in COND_ORDER
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Surprisal-based evaluator (evalv5)")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--provider", default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--verbs", nargs="+", required=True)
    # Ignored — kept for SLURM script compatibility
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--think-budget", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--eval-prompt", default="")
    parser.add_argument("--rank-metric", default="total_logp",
                    choices=["total_logp", "frame_logp", "agent_logp"],
                    help="Metric to rank conditions by")
    args = parser.parse_args()

    output_dir = os.path.join(OUTPUT_BASE, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="dummy")
    try:
        models = client.models.list()
        logger.info(f"Connected to vLLM. Models: {[m.id for m in models.data]}")
    except Exception as e:
        logger.error(f"Cannot connect to vLLM at port {args.port}: {e}")
        return

    logger.info(f"Evaluation run: {args.run_name}")
    logger.info(f"Surprisal model: {args.model}")
    logger.info(f"Evaluating {len(args.verbs)} verb(s): {args.verbs}")

    results = []
    for i, verb in enumerate(args.verbs):
        logger.info(f"[{i+1}/{len(args.verbs)}] Computing surprisal for: '{verb}'")

        result = process_verb(client, args.model, args.input_dir, verb, args.rank_metric)
        if result is None:
            logger.error(f"  Failed for '{verb}'")
            continue

        out_path = os.path.join(output_dir, f"{verb}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        strict = result["scores"]["score_strict"]
        sd = result["surprisal_data"]
        ranked = sorted(COND_ORDER, key=lambda c: sd[c]["total_logp"], reverse=True)
        rank_str = " > ".join(f"{c[:2].upper()}({sd[c]['total_logp']:.1f})" for c in ranked)
        logger.info(f"  Saved: {out_path} | {strict}/4 strict | {rank_str}")

        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"SURPRISAL EVALUATION SUMMARY: {args.run_name}")
    print(f"{'='*70}")
    print(f"{'Verb':<15s} {'Strict':>8s}  {'LogP Ranking'}")
    print(f"{'-'*15} {'-'*8}  {'-'*50}")
    for r in results:
        sd = r["surprisal_data"]
        ranked = sorted(COND_ORDER, key=lambda c: sd[c]["total_logp"], reverse=True)
        rank_str = " > ".join(c[:2].upper() for c in ranked)
        print(f"{r['verb']:<15s} {r['scores']['score_strict']:>4d}/4    {rank_str}")


if __name__ == "__main__":
    main()