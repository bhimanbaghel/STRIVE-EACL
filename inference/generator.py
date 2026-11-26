"""
Core generation logic for the Events task.
Loads prompt, formats it with a verb, calls the model, parses and validates output.

Slot-aware: detects which slot varies (agent / patient / instrument / location)
from the parsed output structure and validates the v7 prompt family accordingly.
"""

import json
import logging
from typing import Dict, Any, Tuple
from inference.providers import generate, extract_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONDITIONS = ["plausible_easy", "plausible_hard", "implausible_easy", "implausible_hard"]
EXPECTED_CONDITION_KEYS = set(CONDITIONS)

FIT_LEVELS = {"plausible_easy": "HIGH", "plausible_hard": "MODERATE",
              "implausible_easy": "VERY LOW", "implausible_hard": "LOW"}

# All four slot names that could be the varying or fixed slots
ALL_SLOTS = ["agent", "patient", "instrument", "location"]

# Map varying-block name (at the top level) -> per-entry key name
# e.g., for v7_patient: data["patients"][cond]["patient"]
VARYING_BLOCK_TO_KEY = {
    "agents": "agent",
    "patients": "patient",
    "instruments": "instrument",
    "locations": "location",
}

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
def load_prompt(prompt_path: str) -> dict:
    """Load a prompt JSON file."""
    with open(prompt_path, "r") as f:
        return json.load(f)


def format_prompt(prompt_cfg: dict, verb: str) -> tuple:
    """
    Format the prompt template with the given verb.
    Returns: (system_prompt, user_prefix, user_prompt)
      - user_prefix: fixed template portion (cacheable for Claude)
      - user_prompt: variable portion (verb-specific)
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

    return system, user_prefix, user_suffix


# ---------------------------------------------------------------------------
# Slot detection
# ---------------------------------------------------------------------------
def detect_varying_slot(data: dict) -> Tuple[str, str]:
    """
    Detect which slot varies in a v7 parsed output by inspecting top-level keys.
    Returns (block_name, per_entry_key), e.g., ('patients', 'patient').
    Raises ValueError if zero or multiple varying blocks are present.
    """
    found = [block for block in VARYING_BLOCK_TO_KEY if block in data]
    if len(found) == 0:
        raise ValueError(
            f"No varying-slot block found. Expected one of "
            f"{list(VARYING_BLOCK_TO_KEY.keys())}."
        )
    if len(found) > 1:
        raise ValueError(
            f"Multiple varying-slot blocks found: {found}. Expected exactly one."
        )
    block = found[0]
    return block, VARYING_BLOCK_TO_KEY[block]


def output_schema(data: dict) -> str:
    """Identify whether the parsed output follows the v7 schema."""
    if not isinstance(data, dict):
        return "unknown"
    if "scene" in data or any(block in data for block in VARYING_BLOCK_TO_KEY):
        return "v7"
    return "unknown"


# ---------------------------------------------------------------------------
# Key normalization via fuzzy matching
# ---------------------------------------------------------------------------
def _best_match(key: str, candidates: set, threshold: float = 0.8) -> str:
    """Find the closest matching key using SequenceMatcher. Returns original if no good match."""
    from difflib import SequenceMatcher
    if key in candidates:
        return key
    best_key, best_score = key, 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, key.lower(), candidate.lower()).ratio()
        if score > best_score:
            best_key, best_score = candidate, score
    return best_key if best_score >= threshold else key


def _fix_condition_keys(d: dict, expected: set) -> dict:
    """Fuzzy match condition keys against the expected keys."""
    if not isinstance(d, dict):
        return d
    fixed = {}
    for k, v in d.items():
        matched = _best_match(k, expected)
        if matched != k:
            logger.info(f"Key correction: '{k}' -> '{matched}'")
        if matched in fixed and matched != k:
            logger.info(f"Key correction skipped: '{k}' -> '{matched}' (already present)")
            matched = k
        fixed[matched] = v
    return fixed


def normalize_keys(data: dict) -> dict:
    """Fix misspelled condition keys in the varying slot block and sentences."""
    if not isinstance(data, dict):
        return data

    for section in list(VARYING_BLOCK_TO_KEY.keys()) + ["sentences"]:
        if section in data and isinstance(data[section], dict):
            data[section] = _fix_condition_keys(data[section], EXPECTED_CONDITION_KEYS)

    return data


# ---------------------------------------------------------------------------
# Output validation (slot-aware)
# ---------------------------------------------------------------------------
def validate_output(data: dict) -> list:
    """Return errors for parsed JSON that violates the v7 task constraints."""

    errors = []

    # Top-level keys always required
    if "scene" not in data:
        errors.append("Missing top-level key: scene")
    if "sentences" not in data:
        errors.append("Missing top-level key: sentences")

    # Exactly one varying-slot block must be present
    try:
        varying_block, varying_key = detect_varying_slot(data)
    except ValueError as e:
        errors.append(f"Varying-slot detection failed: {e}")
        return errors

    if errors:
        return errors

    # Scene contains verb_form + the three FIXED slots (everything except varying)
    fixed_slots = [s for s in ALL_SLOTS if s != varying_key]
    for slot in ["verb_form"] + fixed_slots:
        if slot not in data["scene"]:
            errors.append(f"Missing scene slot: {slot}")

    # Varying block: each condition needs the varying key + rationale.
    # Sentences block: each condition needs a sentence.
    for cond in CONDITIONS:
        if cond not in data[varying_block]:
            errors.append(f"Missing {varying_block}.{cond}")
        else:
            entry = data[varying_block][cond]
            if not isinstance(entry, dict):
                errors.append(f"{varying_block}.{cond} is not a dict")
                continue
            if varying_key not in entry:
                errors.append(f"Missing '{varying_key}' field in {varying_block}.{cond}")
            if "rationale" not in entry:
                errors.append(f"Missing 'rationale' field in {varying_block}.{cond}")
        if cond not in data["sentences"]:
            errors.append(f"Missing sentence for condition: {cond}")

    if errors:
        return errors

    # Consistency: verb_form and the three fixed slots must appear in every sentence
    scene = data["scene"]
    for cond in CONDITIONS:
        sent = data["sentences"][cond]
        for slot in ["verb_form"] + fixed_slots:
            val = scene.get(slot, "")
            if val and val.lower() not in sent.lower():
                errors.append(f"Sentence '{cond}' may not contain scene {slot}: '{val}'")

    # The varying slot's value for each condition must appear in that sentence
    for cond in CONDITIONS:
        slot_value = data[varying_block][cond].get(varying_key, "")
        sent = data["sentences"][cond]
        if slot_value and slot_value.lower() not in sent.lower():
            errors.append(
                f"{varying_key.capitalize()} '{slot_value}' not found in sentence for '{cond}'"
            )

    # Optional: validate verbalized-sampling 'candidates' if present
    if "candidates" in data:
        if not isinstance(data["candidates"], list):
            errors.append("'candidates' must be a list")
        elif len(data["candidates"]) < 10:
            errors.append(f"Expected ~20 candidates, got {len(data['candidates'])}")
        else:
            for i, c in enumerate(data["candidates"]):
                if not isinstance(c, dict):
                    errors.append(f"Candidate {i} is not a dict")
                elif varying_key not in c or "thematic_fit_prob" not in c:
                    errors.append(
                        f"Candidate {i} missing '{varying_key}' or 'thematic_fit_prob'"
                    )

    return errors


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_events(verb: str, prompt_path: str, provider: str,
                    model_name: str, temperature: float = 0.0,
                    max_tokens: int = 2048, port: int = 8000,
                    think_budget: int = 0, reasoning_effort: str = "",
                    thinking_effort: str = "high",
                    max_retries: int = 1, seed: int = None,
                    top_p: float = None, **extra_kwargs) -> Dict[str, Any]:
    """
    Generate event sentences for a given verb.
    Works with any v7 prompt variant (agent / patient / instrument / location).
    Retries up to max_retries times on JSON parse or validation failures.

    Returns dict with keys:
        verb, raw_response, parsed, validation_errors,
        elapsed, token_info, success, schema, seed, top_p
    """
    if seed is not None:
        extra_kwargs["seed"] = int(seed)
    if top_p is not None:
        extra_kwargs["top_p"] = float(top_p)
    prompt_cfg = load_prompt(prompt_path)
    system_prompt, user_prefix, user_prompt = format_prompt(prompt_cfg, verb)

    for attempt in range(max_retries + 1):
        result = {
            "verb": verb,
            "raw_response": None,
            "parsed": None,
            "validation_errors": [],
            "elapsed": 0.0,
            "token_info": {},
            "success": False,
            "schema": "",
            "seed": seed,
            "top_p": top_p,
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
                think_budget=think_budget,
                reasoning_effort=reasoning_effort,
                thinking_effort=thinking_effort,
                **extra_kwargs,
            )
            result["raw_response"] = raw
            result["elapsed"] = elapsed
            result["token_info"] = token_info

        except Exception as e:
            result["validation_errors"] = [f"Generation failed: {e}"]
            logger.error(f"Generation failed for verb='{verb}': {e}")
            if attempt < max_retries:
                logger.warning(f"Retrying generation for '{verb}' ({attempt+1}/{max_retries})...")
                continue
            return result

        # Parse JSON
        try:
            parsed = extract_json(raw)
            parsed = normalize_keys(parsed)
            result["parsed"] = parsed
            result["schema"] = output_schema(parsed)
        except ValueError as e:
            result["validation_errors"] = [f"JSON parse failed: {e}"]
            logger.warning(f"JSON parse failed for verb='{verb}': {e}")
            if attempt < max_retries:
                logger.warning(f"Retrying generation for '{verb}' ({attempt+1}/{max_retries})...")
                continue
            return result

        # Validate (slot-aware)
        errs = validate_output(parsed)
        result["validation_errors"] = errs
        result["success"] = len(errs) == 0

        if errs:
            logger.warning(f"Validation errors for verb='{verb}': {errs}")
            if attempt < max_retries:
                logger.warning(f"Retrying generation for '{verb}' ({attempt+1}/{max_retries})...")
                continue
        else:
            ti = result["token_info"]
            cache_str = ""
            if ti.get("cache_write_tokens") or ti.get("cached_tokens"):
                cache_str = (f" | cache_write={ti.get('cache_write_tokens', 0)} "
                             f"cache_hit={ti.get('cached_tokens', 0)}")
            logger.info(f"Success for verb='{verb}' in {elapsed:.2f}s"
                        f" | in={ti.get('input_tokens',0)} out={ti.get('output_tokens',0)}{cache_str}")

        return result

    return result