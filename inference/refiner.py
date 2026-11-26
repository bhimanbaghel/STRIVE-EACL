# inference/refiner.py
"""
Refiner for Events task.
Takes a previous generation output and evaluator feedback,
calls the model to produce a revised output.

The evaluator provides a scene_verdict of GOOD, AGENT_FIXABLE, or NEEDS_REDESIGN.
The orchestrator formats this into actionable feedback for the refiner.
"""

import json
import logging
from typing import Dict, Any
from inference.providers import generate, extract_json
from inference.generator import normalize_keys

logger = logging.getLogger(__name__)


def load_refine_prompt(prompt_path: str) -> dict:
    with open(prompt_path, "r") as f:
        return json.load(f)


def format_refine_prompt(prompt_cfg: dict, verb: str,
                         previous_output: dict, evaluator_feedback: str) -> tuple:
    """
    Format the refinement prompt with verb, previous output, and feedback.
    Returns: (system_prompt, user_prompt)
    """
    system = "\n".join(prompt_cfg.get("system", []))
    user = "\n".join(prompt_cfg.get("userTemplate", []))
    user = user.replace("{verb}", verb)
    user = user.replace("{previous_output}", json.dumps(previous_output, indent=2))
    user = user.replace("{evaluator_feedback}", evaluator_feedback)
    return system, user


# Reuse validation from generator
CONDITIONS = ["plausible_easy", "plausible_hard", "implausible_easy", "implausible_hard"]
VALID_STRATEGIES = {"SCENE_REDESIGN", "AGENT_REPLACEMENT", "AGENT_SWAP", "MIXED"}
REQUIRED_KEYS = {"scene", "agents", "sentences", "revision_strategy", "changes_made"}



def validate_refined_output(data: dict) -> list:
    """Validate the refined output JSON structure."""
    errors = []

    for k in REQUIRED_KEYS:
        if k not in data:
            errors.append(f"Missing top-level key: {k}")
    if errors:
        return errors

    # Strategy must be one of the four enum values
    strategy = data.get("revision_strategy", "")
    if strategy not in VALID_STRATEGIES:
        errors.append(f"Invalid revision_strategy: {strategy!r} "
                      f"(expected one of {sorted(VALID_STRATEGIES)})")

    # changes_made must be a non-empty string
    cm = data.get("changes_made", "")
    if not isinstance(cm, str) or not cm.strip():
        errors.append("changes_made must be a non-empty string")

    # Scene keys
    for k in ["verb_form", "patient", "instrument", "location"]:
        if k not in data.get("scene", {}):
            errors.append(f"Missing scene key: {k}")

    # Agents and sentences
    for cond in CONDITIONS:
        if cond not in data.get("agents", {}):
            errors.append(f"Missing agent for: {cond}")
        else:
            agent_obj = data["agents"][cond]
            if "agent" not in agent_obj:
                errors.append(f"Missing 'agent' in agents.{cond}")
            if "rationale" not in agent_obj:
                errors.append(f"Missing 'rationale' in agents.{cond}")
        if cond not in data.get("sentences", {}):
            errors.append(f"Missing sentence for: {cond}")

    if errors:
        return errors

    # Consistency: scene slots in sentences
    scene = data["scene"]
    for cond in CONDITIONS:
        sent = data["sentences"][cond]
        for slot in ["patient", "instrument", "location"]:
            val = scene.get(slot, "")
            if val and val.lower() not in sent.lower():
                errors.append(f"Sentence '{cond}' may not contain scene {slot}: '{val}'")
        vf = scene.get("verb_form", "")
        if vf and vf.lower() not in sent.lower():
            errors.append(f"Sentence '{cond}' may not contain verb_form: '{vf}'")

    # Agent in sentence
    for cond in CONDITIONS:
        agent = data["agents"][cond]["agent"]
        sent = data["sentences"][cond]
        if agent.lower() not in sent.lower():
            errors.append(f"Agent '{agent}' not in sentence for '{cond}'")
    
    # Per-agent 'changed' field: coerce string→bool, warn on missing
    for cond in CONDITIONS:
        agent_obj = data.get("agents", {}).get(cond, {})
        if "changed" in agent_obj:
            val = agent_obj["changed"]
            if isinstance(val, str):
                # Coerce common string forms to bool
                if val.strip().lower() in ("true", "yes", "1"):
                    agent_obj["changed"] = True
                elif val.strip().lower() in ("false", "no", "0"):
                    agent_obj["changed"] = False
                else:
                    errors.append(f"agents.{cond}.changed has non-boolean string: {val!r}")
            elif not isinstance(val, bool):
                errors.append(f"agents.{cond}.changed must be a boolean (got {type(val).__name__})")
        # If 'changed' is missing entirely, leave as-is (not strictly enforced -- the field
        # is informational; the orchestrator doesn't gate on it).

    return errors

def _check_strategy_consistency(data: dict) -> list:
    """Soft check: is revision_strategy consistent with the changed flags?"""
    warnings = []
    strategy = data.get("revision_strategy", "")
    changed_flags = {
        cond: data.get("agents", {}).get(cond, {}).get("changed", None)
        for cond in CONDITIONS
    }
    n_changed = sum(1 for v in changed_flags.values() if v is True)

    if strategy == "AGENT_REPLACEMENT" and n_changed == 0:
        warnings.append("Strategy=AGENT_REPLACEMENT but no agent marked changed=true")
    elif strategy == "AGENT_SWAP" and n_changed != 2:
        warnings.append(f"Strategy=AGENT_SWAP but {n_changed} agents marked changed=true (expected 2)")
    elif strategy == "SCENE_REDESIGN" and n_changed != 4:
        warnings.append(f"Strategy=SCENE_REDESIGN but only {n_changed}/4 agents marked changed=true")

    return warnings

def refine_events(verb: str, previous_output: dict, evaluator_feedback: str,
                  refine_prompt_path: str, provider: str, model_name: str,
                  temperature: float = 0.3, max_tokens: int = 2048,
                  port: int = 8000, reasoning_effort: str = "",
                  think_budget: int = 0, **extra_kwargs) -> Dict[str, Any]:
    """
    Refine a previous generation based on evaluator feedback.

    Returns dict with keys:
        verb, raw_response, parsed, validation_errors,
        elapsed, token_info, success
    """
    prompt_cfg = load_refine_prompt(refine_prompt_path)
    system_prompt, user_prompt = format_refine_prompt(
        prompt_cfg, verb, previous_output, evaluator_feedback
    )

    result = {
        "verb": verb,
        "raw_response": None,
        "parsed": None,
        "validation_errors": [],
        "elapsed": 0.0,
        "token_info": {},
        "success": False,
    }

    try:
        raw, elapsed, token_info = generate(
            provider=provider,
            model_name=model_name,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            port=port,
            reasoning_effort=reasoning_effort,
            think_budget=think_budget,
            **extra_kwargs,
        )
        result["raw_response"] = raw
        result["elapsed"] = elapsed
        result["token_info"] = token_info

    except Exception as e:
        result["validation_errors"] = [f"Refinement call failed: {e}"]
        logger.error(f"Refinement failed for verb='{verb}': {e}")
        return result

    # Parse JSON
    try:
        parsed = extract_json(raw)
        parsed = normalize_keys(parsed)
        result["parsed"] = parsed
    except ValueError as e:
        result["validation_errors"] = [f"JSON parse failed: {e}"]
        logger.warning(f"Refiner JSON parse failed for verb='{verb}': {e}")
        return result

    # Hard validation -- gates success
    errs = validate_refined_output(parsed)
    result["validation_errors"] = errs
    result["success"] = len(errs) == 0

    # Soft consistency check -- log only, do not gate success
    strategy_warnings = _check_strategy_consistency(parsed)
    result["strategy_warnings"] = strategy_warnings
    for w in strategy_warnings:
        logger.warning(f"Refinement consistency warning for verb='{verb}': {w}")

    if errs:
        logger.warning(f"Refinement validation errors for verb='{verb}': {errs}")
    else:
        strategy = parsed.get("revision_strategy", "unknown")
        changes = parsed.get("changes_made", "")
        logger.info(f"Refinement for verb='{verb}': strategy={strategy}, changes={changes}")

    return result
