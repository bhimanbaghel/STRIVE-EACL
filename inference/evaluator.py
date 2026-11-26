"""
Evaluator for Events task outputs.
Sends generated output to a judge model (GPT-5 thinking mode recommended)
and parses structured evaluation results.

Updated for v7 prompt family: slot-aware. Detects which slot varies
(agent / patient / instrument / location) in the generator output and:
  - strips the right block for the evaluator
  - accepts the variant-specific scene_verdict (AGENT_FIXABLE / PATIENT_FIXABLE / ...)
  - reads the right identifiability block in visual_evaluation

Backward compatible with evaluator_v2/v3 agent-variant outputs.
"""

import json
import logging
from typing import Dict, Any

from inference.providers import generate, extract_json
from inference.generator import detect_varying_slot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loading and formatting
# ---------------------------------------------------------------------------
def load_evaluator_prompt(prompt_path: str) -> dict:
    with open(prompt_path, "r") as f:
        return json.load(f)


def format_evaluator_prompt(prompt_cfg: dict, verb: str, generated_output: dict) -> tuple:
    """
    Format evaluator prompt with the verb and the generated output to evaluate.
    Returns: (system_prompt, user_prefix, user_prompt)
      - user_prefix: fixed template portion (cacheable for Claude)
      - user_prompt: variable portion only (verb + generated_output)
    """
    system = "\n".join(prompt_cfg.get("system", []))
    template_lines = prompt_cfg.get("userTemplate", [])

    split_idx = next(
        (i for i, line in enumerate(template_lines) if "{verb}" in line),
        len(template_lines)
    )

    user_prefix = "\n".join(template_lines[:split_idx])
    user_suffix = "\n".join(template_lines[split_idx:])
    user_suffix = user_suffix.replace("{verb}", verb)
    user_suffix = user_suffix.replace("{generated_output}", json.dumps(generated_output, indent=2))

    return system, user_prefix, user_suffix


# ---------------------------------------------------------------------------
# Evaluation result validation
# ---------------------------------------------------------------------------
CONDITIONS = ["plausible_easy", "plausible_hard", "implausible_easy", "implausible_hard"]
VALID_VERDICTS = {"CORRECT", "BORDERLINE", "INCORRECT", "SKIPPED"}
VALID_IDENTIFIABILITY = {"IDENTIFIABLE", "AMBIGUOUS", "UNIDENTIFIABLE", "SKIPPED"}

# Slot-aware: each variant emits its own *_FIXABLE label, plus GOOD/NEEDS_REDESIGN
VALID_SCENE_VERDICTS = {
    "GOOD",
    "AGENT_FIXABLE", "PATIENT_FIXABLE", "INSTRUMENT_FIXABLE", "LOCATION_FIXABLE",
    "NEEDS_REDESIGN",
}

# Per-condition identifiability blocks vary by evaluator variant:
#   - agent eval (v3 / v3.1_agent):   visual_evaluation.identifiability
#   - patient eval (v3.1_patient):    visual_evaluation.patient_identifiability
#   - future variants:                visual_evaluation.{instrument,location}_identifiability
IDENT_BLOCK_CANDIDATES = [
    "identifiability",
    "patient_identifiability",
    "instrument_identifiability",
    "location_identifiability",
]


def validate_evaluation(data: dict, criterion: str = "visual") -> list:
    """Validate the evaluator's JSON output structure. Returns list of errors.

    Backward compatible: new v3 fields (counter_arguments, overlap_audit,
    identifiability, borderline_count) are validated if present but not required.
    Slot-aware: accepts AGENT_FIXABLE / PATIENT_FIXABLE / ... scene verdicts
    and looks for per-condition identifiability under any variant-specific key.
    """
    errors = []

    if criterion not in {"visual", "text"}:
        raise ValueError(f"Unknown evaluation criterion: {criterion}")

    required_keys = ["constraint_checks", "condition_evaluations",
                     "gradient_ordering", "scene_assessment", "overall"]
    if criterion == "visual":
        required_keys.append("visual_evaluation")

    for key in required_keys:
        if key not in data:
            errors.append(f"Missing top-level key: {key}")
    if errors:
        return errors

    # Check condition evaluations
    for cond in CONDITIONS:
        if cond not in data["condition_evaluations"]:
            errors.append(f"Missing condition evaluation: {cond}")
        else:
            ce = data["condition_evaluations"][cond]
            if "verdict" not in ce:
                errors.append(f"Missing verdict for {cond}")
            elif ce["verdict"] not in VALID_VERDICTS:
                errors.append(f"Invalid verdict for {cond}: {ce['verdict']}")
            if "reasoning" not in ce:
                errors.append(f"Missing reasoning for {cond}")

            if "counter_arguments" in ce:
                ca = ce["counter_arguments"]
                if not isinstance(ca, dict):
                    errors.append(f"counter_arguments for {cond} must be a dict")

            if "overlap_audit" in ce:
                oa = ce["overlap_audit"]
                if not isinstance(oa, dict):
                    errors.append(f"overlap_audit for {cond} must be a dict")
                elif cond == "implausible_hard":
                    if "overlap_count" in oa:
                        oc = oa["overlap_count"]
                        if not isinstance(oc, (int, float)):
                            errors.append(f"overlap_count must be numeric, got: {type(oc).__name__}")
                    if "sufficient" in oa and not isinstance(oa["sufficient"], bool):
                        errors.append(f"overlap_audit.sufficient must be bool")

    # Check visual evaluation (slot-aware identifiability block)
    visual = data.get("visual_evaluation", {}) or {}
    ident = None
    if isinstance(visual, dict):
        for candidate in IDENT_BLOCK_CANDIDATES:
            if candidate in visual and isinstance(visual[candidate], dict):
                ident = visual[candidate]
                break

    if isinstance(ident, dict):
        for cond in CONDITIONS:
            if cond in ident:
                ic = ident[cond]
                if isinstance(ic, dict):
                    rating = ic.get("rating", "")
                    if rating and rating not in VALID_IDENTIFIABILITY:
                        errors.append(f"Invalid identifiability rating for {cond}: {rating}")

    # Check scene assessment (slot-aware verdict)
    scene = data.get("scene_assessment", {}) or {}
    if "scene_verdict" not in scene:
        errors.append("Missing scene_verdict in scene_assessment")
    elif scene["scene_verdict"] not in VALID_SCENE_VERDICTS:
        errors.append(f"Invalid scene_verdict: {scene['scene_verdict']}")

    # Check overall score format
    if "overall" in data:
        score = data["overall"].get("score", "")
        if "/" not in str(score):
            errors.append(f"Score format should be 'N/4', got: {score}")

        if "borderline_count" in data["overall"]:
            bc = data["overall"]["borderline_count"]
            if not isinstance(bc, (int, float)):
                errors.append(f"overall.borderline_count must be numeric, got: {type(bc).__name__}")

    return errors


# ---------------------------------------------------------------------------
# Score extraction (slot-aware)
# ---------------------------------------------------------------------------
def extract_scores(evaluation: dict) -> dict:
    """Extract numeric scores from evaluation for easy aggregation.

    Slot-aware:
      - Detects per-condition identifiability under the variant-specific key
        (identifiability / patient_identifiability / ...)
      - Extracts fixed_agent_identifiability when present (non-agent variants)
      - Extracts agent_action_prototypical when present (non-agent variants)

    Backward compatible with agent-variant outputs (v3 / v3.1_agent).
    """
    scores = {
        "correct_count": 0,
        "borderline_count": 0,
        "incorrect_count": 0,
        "has_scene_error": False,
        "scene_verdict": evaluation.get("scene_assessment", {}).get("scene_verdict", "UNKNOWN"),
        "gradient_supportable": evaluation.get("scene_assessment", {}).get("gradient_supportable", True),
        "conditions": {},
    }

    scene = evaluation.get("scene_assessment", {}) or {}
    # has_scene_error: any of the structural failures
    scores["has_scene_error"] = (
        scene.get("instrument_correct", True) is False
        or scene.get("has_clear_prototype", True) is False
        or scene.get("scene_verdict") == "NEEDS_REDESIGN"
        # For non-agent variants: agent_action_prototypical must hold or the scene fails
        or scene.get("agent_action_prototypical", True) is False
    )

    for cond in CONDITIONS:
        ce = evaluation.get("condition_evaluations", {}).get(cond, {}) or {}
        verdict = ce.get("verdict", "UNKNOWN")
        scores["conditions"][cond] = {
            "verdict": verdict,
            "actual_condition": ce.get("actual_condition", cond),
        }
        if verdict == "CORRECT":
            scores["correct_count"] += 1
        elif verdict == "BORDERLINE":
            scores["borderline_count"] += 1
        elif verdict == "INCORRECT":
            scores["incorrect_count"] += 1

    scores["score_strict"] = scores["correct_count"]
    scores["score_lenient"] = scores["correct_count"] + 0.5 * scores["borderline_count"]

    # Overlap audit for implausible_hard
    ih_ce = evaluation.get("condition_evaluations", {}).get("implausible_hard", {}) or {}
    oa = ih_ce.get("overlap_audit", {}) if isinstance(ih_ce, dict) else {}
    if isinstance(oa, dict) and oa:
        scores["overlap_count"] = oa.get("overlap_count", None)
        scores["overlap_sufficient"] = oa.get("sufficient", None)
        scores["overlap_negative_test"] = oa.get("negative_test", None)
    else:
        scores["overlap_count"] = None
        scores["overlap_sufficient"] = None
        scores["overlap_negative_test"] = None

    # Per-condition identifiability of the varying entity — slot-aware lookup
    visual = evaluation.get("visual_evaluation", {}) or {}
    ident_block = None
    if isinstance(visual, dict):
        for candidate in IDENT_BLOCK_CANDIDATES:
            if candidate in visual and isinstance(visual[candidate], dict):
                ident_block = visual[candidate]
                break

    if ident_block:
        scores["identifiability"] = {}
        unidentifiable = []
        for cond in CONDITIONS:
            ic = ident_block.get(cond, {})
            rating = ic.get("rating", "UNKNOWN") if isinstance(ic, dict) else "UNKNOWN"
            scores["identifiability"][cond] = rating
            if rating == "UNIDENTIFIABLE":
                unidentifiable.append(cond)
        # Keep historical key name for downstream compatibility; semantically it
        # is "unidentifiable conditions of the varying entity"
        scores["unidentifiable_agents"] = unidentifiable if unidentifiable else None
    else:
        scores["identifiability"] = None
        scores["unidentifiable_agents"] = None

    # Fixed-agent identifiability (present only in non-agent variants where the agent is fixed)
    fixed_ident = visual.get("fixed_agent_identifiability", {}) if isinstance(visual, dict) else {}
    if isinstance(fixed_ident, dict) and fixed_ident:
        scores["fixed_agent_identifiability"] = fixed_ident.get("rating", "UNKNOWN")
    else:
        scores["fixed_agent_identifiability"] = None

    # Agent-action-prototypical check (present only in non-agent variants)
    scores["agent_action_prototypical"] = scene.get("agent_action_prototypical", None)

    return scores


# ---------------------------------------------------------------------------
# Slot-aware stripping for evaluator input
# ---------------------------------------------------------------------------
def strip_for_evaluator(generated_output, criterion: str = "visual"):
    """Strip generator reasoning so the evaluator judges independently.

    Slot-aware: works for v7_agent / v7_patient / v7_instrument / v7_location
    variants. Detects which top-level varying block is present
    (agents / patients / instruments / locations) and strips accordingly.

    Keeps in scene: all fixed slots.
    Keeps per condition: <varying_key>, visual_markers, overlap_dimensions (IH only).
    Keeps: sentences.
    Strips: reasoning, candidates, selection_reasoning (top-level);
            thematic_fit, thematic_fit_prob, rationale (per condition).
    """
    if not isinstance(generated_output, dict):
        return generated_output

    try:
        varying_block, varying_key = detect_varying_slot(generated_output)
    except ValueError:
        # Fallback: assume agent variant for backward compatibility with
        # outputs from earlier prompt versions
        varying_block, varying_key = "agents", "agent"

    stripped = {
        "scene": generated_output.get("scene", {}),
        varying_block: {},
        "sentences": generated_output.get("sentences", {}),
    }

    source_block = generated_output.get(varying_block, {}) or {}
    for condition, slot_data in source_block.items():
        if not isinstance(slot_data, dict):
            stripped[varying_block][condition] = slot_data
            continue
        entry = {varying_key: slot_data.get(varying_key, "")}
        # Text-only judging must not be influenced by visual annotations that
        # may be present in historical generator outputs used for AAT.
        if criterion == "visual" and "visual_markers" in slot_data:
            entry["visual_markers"] = slot_data["visual_markers"]
        if condition == "implausible_hard" and "overlap_dimensions" in slot_data:
            entry["overlap_dimensions"] = slot_data["overlap_dimensions"]
        stripped[varying_block][condition] = entry

    return stripped


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def evaluate_events(verb: str, generated_output: dict, evaluator_prompt_path: str,
                    provider: str, model_name: str, temperature: float = 0.0,
                    max_tokens: int = 16384, port: int = 8000,
                    reasoning_effort: str = "high",
                    think_budget: int = 8000, max_retries: int = 1,
                    thinking_effort: str = "high",
                    criterion: str = "visual",
                    **extra_kwargs) -> Dict[str, Any]:
    """
    Evaluate a generated events output using a judge model.
    Retries up to max_retries times on JSON parse failures.

    Returns dict with keys:
        verb, raw_response, evaluation, validation_errors,
        scores, elapsed, token_info, success
    """
    prompt_cfg = load_evaluator_prompt(evaluator_prompt_path)
    if criterion not in {"visual", "text"}:
        raise ValueError(f"Unknown evaluation criterion: {criterion}")

    eval_input = strip_for_evaluator(generated_output, criterion=criterion)
    system_prompt, user_prefix, user_prompt = format_evaluator_prompt(prompt_cfg, verb, eval_input)

    for attempt in range(max_retries + 1):
        result = {
            "verb": verb,
            "raw_response": None,
            "evaluation": None,
            "validation_errors": [],
            "scores": None,
            "elapsed": 0.0,
            "token_info": {},
            "success": False,
            "criterion": criterion,
        }

        try:
            raw, elapsed, token_info = generate(
                provider=provider,
                model_name=model_name,
                prompt=user_prompt,
                system_prompt=system_prompt,
                user_prefix=user_prefix,
                max_tokens=max_tokens,
                temperature=temperature,
                port=port,
                reasoning_effort=reasoning_effort,
                think_budget=think_budget,
                thinking_effort=thinking_effort,
                **extra_kwargs,
            )
            result["raw_response"] = raw
            result["elapsed"] = elapsed
            result["token_info"] = token_info

        except Exception as e:
            result["validation_errors"] = [f"Evaluation call failed: {e}"]
            logger.error(f"Evaluation failed for verb='{verb}': {e}")
            if attempt < max_retries:
                logger.warning(f"Retrying evaluation for '{verb}' ({attempt+1}/{max_retries})...")
                continue
            return result

        try:
            parsed = extract_json(raw)
            result["evaluation"] = parsed
        except ValueError as e:
            result["validation_errors"] = [f"JSON parse failed: {e}"]
            logger.warning(f"Evaluator JSON parse failed for verb='{verb}': {e}")
            if attempt < max_retries:
                logger.warning(f"Retrying evaluation for '{verb}' ({attempt+1}/{max_retries})...")
                continue
            return result

        errs = validate_evaluation(parsed, criterion=criterion)
        result["validation_errors"] = errs

        if errs:
            logger.warning(f"Evaluator validation errors for verb='{verb}': {errs}")
            if attempt < max_retries:
                logger.warning(f"Retrying evaluation for '{verb}' ({attempt+1}/{max_retries})...")
                continue
            result["success"] = False
            return result

        result["scores"] = extract_scores(parsed)
        result["success"] = True

        ti = result["token_info"]
        cache_str = ""
        if ti.get("cache_write_tokens") or ti.get("cached_tokens"):
            cache_str = (f" | cache_write={ti.get('cache_write_tokens', 0)} "
                         f"cache_hit={ti.get('cached_tokens', 0)}")
        logger.info(f"Evaluation for verb='{verb}': {result['scores']['score_strict']}/4 strict, "
                    f"{result['scores']['score_lenient']}/4 lenient, {elapsed:.2f}s"
                    f" | in={ti.get('input_tokens',0)} out={ti.get('output_tokens',0)}{cache_str}")

        return result

    return result
