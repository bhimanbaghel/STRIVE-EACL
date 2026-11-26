"""
Surprisal Evaluator v5.2 — Frame-First Conditional Surprisal + Entropy

Approach B: Restructures sentence so the frame comes FIRST, then the agent.
This way agent surprisal directly measures thematic fit (not prior frequency),
and entropy at the agent position captures model "confusion" (IH signature).

Prompt construction:
  "Someone {verb_form} {patient} {instrument} {location}. This person is a"
  → then measure P(agent_noun | frame) and H(distribution | frame)

Metrics:
  - conditional_surprisal: -log P(agent_noun | frame) — lower = better fit
  - positional_entropy: H at agent position — higher = more confusion (IH signal)
  - conditional_logp: log P(agent_noun | frame) — higher = better fit (used for ranking)

Ranking: by conditional_logp descending (highest = PE, lowest = IE)

Usage:
    python scripts/compute_surprisal_v52.py \
        --input-dir results/intermediate/gpt-5.1-2025-11-13_v6 \
        --model Qwen3-30B-A3B-Instruct-2507 \
        --port 8002 \
        --run-name evalv52_qwen30b_nothink_gpt-5.1-2025-11-13_v6 \
        --verbs eat erase wash
"""

import argparse
import json
import logging
import math
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
TOP_K_LOGPROBS = 20  # for entropy approximation


def strip_article(agent_text):
    """Strip leading article from agent phrase: 'A goalie' → 'goalie'."""
    agent = agent_text.strip()
    for prefix in ["An ", "an ", "A ", "a "]:
        if agent.startswith(prefix):
            return agent[len(prefix):]
    return agent


def build_frame_prefix(scene):
    """
    Construct frame-first prefix from scene components.
    Returns: "Someone {verb_form} {patient} {instrument} {location}. This person is a"
    """
    verb_form = scene.get("verb_form", "")
    patient = scene.get("patient", "")
    instrument = scene.get("instrument", "")
    location = scene.get("location", "")

    return f"Someone {verb_form} {patient} {instrument} {location}. This person is a"


def compute_conditional_surprisal(client, model, prefix, agent_noun):
    """
    Compute P(agent_noun | prefix) using completions API.
    Feed prefix + " " + agent_noun, get logprobs of agent tokens.
    Returns: (conditional_logp, n_agent_tokens, token_info)
    """
    full_prompt = prefix + " " + agent_noun

    try:
        response = client.completions.create(
            model=model,
            prompt=full_prompt,
            max_tokens=1,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
    except Exception as e:
        logger.error(f"API error in surprisal: {e}")
        return None, 0, {}

    choice = response.choices[0]
    tokens = list(choice.logprobs.tokens)
    token_logprobs = list(choice.logprobs.token_logprobs)

    # Remove generated token (last one)
    if len(tokens) > 1:
        input_text = "".join(tokens[:-1])
        if len(input_text.strip()) >= len(full_prompt.strip()) - 5:
            tokens = tokens[:-1]
            token_logprobs = token_logprobs[:-1]

    # Find where agent tokens start: after the prefix
    # Reconstruct prefix from tokens to find the boundary
    prefix_len = len(prefix)
    reconstructed = ""
    agent_start_idx = len(tokens)  # default: no agent found
    for i, tok in enumerate(tokens):
        reconstructed += tok
        # Allow small variance for tokenizer whitespace
        if len(reconstructed.strip()) >= prefix_len + 1 and agent_start_idx == len(tokens):
            agent_start_idx = i
            break

    # Agent token logprobs
    agent_lps = [lp for lp in token_logprobs[agent_start_idx:] if lp is not None]
    conditional_logp = sum(agent_lps) if agent_lps else 0.0
    n_agent_tokens = len(agent_lps)

    usage = response.usage
    token_info = {
        "input_tokens": getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
    }

    return conditional_logp, n_agent_tokens, token_info


def compute_entropy(client, model, prefix):
    """
    Compute entropy at the next token position after prefix.
    Uses top-k logprobs as approximation.
    Returns: (entropy, top_tokens_with_probs)
    """
    try:
        response = client.completions.create(
            model=model,
            prompt=prefix,
            max_tokens=1,
            echo=False,
            logprobs=TOP_K_LOGPROBS,
            temperature=0.0,
        )
    except Exception as e:
        logger.error(f"API error in entropy: {e}")
        return None, []

    choice = response.choices[0]

    if not choice.logprobs or not choice.logprobs.top_logprobs:
        logger.warning("No top_logprobs returned")
        return None, []

    top_logprobs = choice.logprobs.top_logprobs[0]  # dict: token → logprob

    # Compute entropy from top-k
    probs = [math.exp(lp) for lp in top_logprobs.values()]
    prob_sum = sum(probs)

    # Add residual mass
    residual = max(0.0, 1.0 - prob_sum)

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    # Add residual entropy (assume uniform over remaining vocab)
    # This is an upper-bound correction
    if residual > 1e-10:
        entropy -= residual * math.log2(residual)

    # Top tokens for inspection
    top_tokens = sorted(top_logprobs.items(), key=lambda x: x[1], reverse=True)[:5]

    return entropy, top_tokens


def load_generation(input_dir, verb):
    """Load generation JSON, extract scene, agents, and sentences."""
    path = os.path.join(input_dir, f"{verb}.json")
    if not os.path.exists(path):
        logger.error(f"Not found: {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    gen = data.get("parsed", data)
    scene = gen.get("scene", {})
    sentences = gen.get("sentences", {})
    agents = gen.get("agents", {})

    if not scene or not scene.get("verb_form"):
        logger.error(f"No scene data in {path}")
        return None

    result = {"scene": scene}
    for cond in COND_ORDER:
        if cond not in sentences:
            logger.warning(f"Missing {cond} in {path}")
            continue
        agent_info = agents.get(cond, {})
        agent_name = agent_info.get("agent", "") if isinstance(agent_info, dict) else str(agent_info)
        result[cond] = {
            "sentence": sentences[cond],
            "agent": agent_name,
            "agent_noun": strip_article(agent_name),
        }

    if len([c for c in COND_ORDER if c in result]) != 4:
        logger.error(f"Missing conditions for {verb}")
        return None

    return result


def build_evaluation_json(verb, condition_data, predictions, frame_prefix):
    """Build evaluator-compatible JSON."""
    eval_output = {
        "constraint_checks": {
            "all_human_professionals": True,
            "consistency_maintained": True,
            "role_action_mismatch_valid": True,
            "constraint_notes": "Surprisal v5.2: frame-first conditional"
        },
        "condition_evaluations": {},
        "gradient_ordering": {"correctly_ordered": True, "swapped_pairs": "None", "collapsed_pairs": "None"},
        "visual_evaluation": {
            "identifiability": {c: {"rating": "SKIPPED", "reasoning": "N/A"} for c in COND_ORDER},
            "agent_appearances": {c: "N/A" for c in COND_ORDER},
            "pairwise_checks": {
                f"{a}_vs_{b}": {"distinguishable": True, "note": "N/A"}
                for a, b in [("pe","ph"),("pe","ih"),("pe","ie"),("ph","ih"),("ph","ie"),("ih","ie")]
            },
            "visual_verdict": "MOSTLY_DISTINCT",
            "indistinguishable_pairs": "None",
            "unidentifiable_agents": "None"
        },
        "scene_assessment": {
            "instrument_correct": True, "has_clear_prototype": True,
            "imageable": True, "gradient_supportable": True,
            "gradient_support_reasoning": "Surprisal v5.2",
            "scene_verdict": "GOOD",
            "scene_notes": f"Frame prefix: '{frame_prefix}'"
        },
        "overall": {}
    }

    correct_count = 0
    for cond in COND_ORDER:
        d = condition_data[cond]
        predicted = predictions[cond]
        is_correct = (predicted == cond)
        if is_correct:
            correct_count += 1

        eval_output["condition_evaluations"][cond] = {
            "agent": d["agent"],
            "verdict": "CORRECT" if is_correct else "INCORRECT",
            "actual_condition": predicted,
            "reasoning": (
                f"Conditional logP={d['conditional_logp']:.2f}, "
                f"surprisal={d['conditional_surprisal']:.2f}, "
                f"entropy={d['entropy']:.2f}. "
                f"Rank predicts {predicted}."
            ),
            "counter_arguments": {"argument_for_higher": "N/A", "argument_for_lower": "N/A"}
        }

    eval_output["overall"] = {
        "score": f"{correct_count}/4",
        "borderline_count": 0,
        "scene_verdict": "GOOD",
        "visual_verdict": "MOSTLY_DISTINCT",
        "summary": f"Frame-first surprisal: {correct_count}/4 match conditional logP ranking."
    }

    logps = [condition_data[c]["conditional_logp"] for c in COND_ORDER]
    mono = all(logps[i] >= logps[i+1] for i in range(3))
    eval_output["gradient_ordering"]["correctly_ordered"] = mono
    if not mono:
        swaps = [f"{COND_ORDER[i]} < {COND_ORDER[i+1]}" for i in range(3) if logps[i] < logps[i+1]]
        eval_output["gradient_ordering"]["swapped_pairs"] = "; ".join(swaps)

    return eval_output


def extract_scores(evaluation):
    """Extract pipeline-compatible scores."""
    scores = {
        "correct_count": 0, "borderline_count": 0, "incorrect_count": 0,
        "has_scene_error": False, "scene_verdict": "GOOD",
        "gradient_supportable": True, "conditions": {},
        "overlap_count": None, "overlap_sufficient": None,
        "overlap_negative_test": None, "identifiability": None,
        "unidentifiable_agents": None,
    }
    for cond in COND_ORDER:
        ce = evaluation["condition_evaluations"].get(cond, {})
        verdict = ce.get("verdict", "UNKNOWN")
        scores["conditions"][cond] = {"verdict": verdict, "actual_condition": ce.get("actual_condition", cond)}
        if verdict == "CORRECT":
            scores["correct_count"] += 1
        elif verdict == "INCORRECT":
            scores["incorrect_count"] += 1
    scores["score_strict"] = scores["correct_count"]
    scores["score_lenient"] = float(scores["correct_count"])
    return scores


def process_verb(client, model, input_dir, verb):
    """Process one verb with frame-first approach."""
    t0 = time.perf_counter()

    gen = load_generation(input_dir, verb)
    if gen is None:
        return None

    scene = gen["scene"]
    frame_prefix = build_frame_prefix(scene)
    logger.info(f"    Frame: '{frame_prefix}'")

    # Compute entropy once (same prefix for all 4 agents)
    entropy_val, top_tokens = compute_entropy(client, model, frame_prefix)
    if entropy_val is not None:
        top_str = ", ".join(f"'{t}':{lp:.2f}" for t, lp in top_tokens[:5])
        logger.info(f"    Entropy={entropy_val:.3f}, top tokens: [{top_str}]")

    condition_data = {}
    total_tokens = {"input_tokens": 0, "output_tokens": 0}

    for cond in COND_ORDER:
        d = gen[cond]
        agent_noun = d["agent_noun"]

        cond_logp, n_agent_tok, tok_info = compute_conditional_surprisal(
            client, model, frame_prefix, agent_noun
        )

        if cond_logp is None:
            logger.error(f"Failed for {verb}/{cond}")
            return None

        total_tokens["input_tokens"] += tok_info.get("input_tokens", 0)

        condition_data[cond] = {
            "agent": d["agent"],
            "agent_noun": agent_noun,
            "sentence": d["sentence"],
            "conditional_logp": cond_logp,
            "conditional_surprisal": -cond_logp,
            "entropy": entropy_val if entropy_val is not None else 0.0,
            "n_agent_tokens": n_agent_tok,
        }

    # Rank by conditional_logp descending (highest = most expected = PE)
    sorted_conds = sorted(COND_ORDER, key=lambda c: condition_data[c]["conditional_logp"], reverse=True)
    predictions = {gen_cond: RANK_TO_COND[rank] for rank, gen_cond in enumerate(sorted_conds)}

    evaluation = build_evaluation_json(verb, condition_data, predictions, frame_prefix)
    scores = extract_scores(evaluation)
    elapsed = time.perf_counter() - t0

    return {
        "verb": verb,
        "raw_response": "surprisal v5.2: frame-first conditional surprisal + entropy",
        "evaluation": evaluation,
        "validation_errors": [],
        "scores": scores,
        "elapsed": round(elapsed, 2),
        "token_info": total_tokens,
        "success": True,
        "surprisal_data": {
            cond: {
                "conditional_logp": round(condition_data[cond]["conditional_logp"], 4),
                "conditional_surprisal": round(condition_data[cond]["conditional_surprisal"], 4),
                "entropy": round(condition_data[cond]["entropy"], 4),
                "n_agent_tokens": condition_data[cond]["n_agent_tokens"],
                "agent_noun": condition_data[cond]["agent_noun"],
            }
            for cond in COND_ORDER
        },
        "frame_prefix": frame_prefix,
        "entropy_info": {
            "positional_entropy": round(entropy_val, 4) if entropy_val is not None else None,
            "top_tokens": [{"token": t, "logprob": round(lp, 4)} for t, lp in (top_tokens or [])],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Surprisal evaluator v5.2 (frame-first)")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--provider", default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--verbs", nargs="+", required=True)
    # Ignored for compatibility
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--think-budget", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--eval-prompt", default="")
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
    logger.info(f"Method: frame-first conditional surprisal + entropy")
    logger.info(f"Evaluating {len(args.verbs)} verb(s): {args.verbs}")

    results = []
    for i, verb in enumerate(args.verbs):
        logger.info(f"[{i+1}/{len(args.verbs)}] Processing: '{verb}'")

        result = process_verb(client, args.model, args.input_dir, verb)
        if result is None:
            logger.error(f"  Failed for '{verb}'")
            continue

        out_path = os.path.join(output_dir, f"{verb}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        strict = result["scores"]["score_strict"]
        sd = result["surprisal_data"]
        ranked = sorted(COND_ORDER, key=lambda c: sd[c]["conditional_logp"], reverse=True)
        rank_str = " > ".join(f"{c[:2].upper()}({sd[c]['conditional_surprisal']:.1f})" for c in ranked)
        logger.info(f"  {verb}: {strict}/4 | entropy={result['entropy_info']['positional_entropy']:.2f} | {rank_str}")

        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"SURPRISAL v5.2 SUMMARY: {args.run_name}")
    print(f"{'='*70}")
    print(f"{'Verb':<12s} {'Score':>6s}  {'Entropy':>8s}  {'Ranking (by surprisal, ascending)'}")
    print(f"{'-'*12} {'-'*6}  {'-'*8}  {'-'*50}")
    for r in results:
        sd = r["surprisal_data"]
        ranked = sorted(COND_ORDER, key=lambda c: sd[c]["conditional_surprisal"])
        rank_str = " < ".join(f"{c[:2].upper()}({sd[c]['conditional_surprisal']:.1f})" for c in ranked)
        ent = r["entropy_info"]["positional_entropy"]
        print(f"{r['verb']:<12s} {r['scores']['score_strict']:>4d}/4  {ent:>8.2f}  {rank_str}")

    total_correct = sum(r["scores"]["score_strict"] for r in results)
    total_items = len(results) * 4
    print(f"\nOverall: {total_correct}/{total_items} ({total_correct/total_items:.0%})")


if __name__ == "__main__":
    main()