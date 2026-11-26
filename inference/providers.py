"""
Model providers for Events task.
Supports: vLLM (local OpenAI-compatible), OpenAI API, and Anthropic (Claude) API.
"""

import os
import re
import json
import time
import logging
from typing import Optional, Tuple, Dict, Any
from functools import wraps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load .env file from project root
try:
    from dotenv import load_dotenv
    # Walk up from inference/ to find .env in project root
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed, rely on shell environment variables

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class APIError(Exception):
    pass

class RateLimitError(APIError):
    pass

class ModelNotAvailableError(APIError):
    pass

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------
def retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    t0 = time.perf_counter()
                    result, token_info = func(*args, **kwargs)
                    elapsed = time.perf_counter() - t0
                    return result, elapsed, token_info
                except RateLimitError as e:
                    last_exc = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Rate limit, retry in {delay:.1f}s ({attempt+1}/{max_retries+1})")
                    time.sleep(delay)
                except (APIError, Exception) as e:
                    last_exc = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"API error, retry in {delay:.1f}s ({attempt+1}/{max_retries+1}): {e}")
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator

# ---------------------------------------------------------------------------
# JSON extraction (robust bracket-balanced scanner)
# ---------------------------------------------------------------------------
def _escape_control_chars_in_strings(json_text: str) -> str:
    """Walk the JSON text and replace literal control characters (\\n, \\r,
    \\t, other <0x20 bytes) inside string values with their proper JSON escape
    sequences. Leaves whitespace outside strings untouched. Idempotent.

    This repairs the most common JSON-spec violation produced by smaller
    instruction-tuned models when they emit a long multi-line string as a
    JSON value (e.g., a 'reasoning' scratchpad field): raw line breaks inside
    the string instead of escaped \\n sequences."""
    out = []
    in_string = False
    escape = False
    for c in json_text:
        if escape:
            out.append(c)
            escape = False
            continue
        if c == '\\':
            out.append(c)
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            out.append(c)
            continue
        if in_string:
            if c == '\n':
                out.append('\\n')
            elif c == '\r':
                out.append('\\r')
            elif c == '\t':
                out.append('\\t')
            elif ord(c) < 0x20:
                out.append(f'\\u{ord(c):04x}')
            else:
                out.append(c)
        else:
            out.append(c)
    return ''.join(out)


def _try_parse(json_text: str):
    """Two-stage JSON parse. Stage 1: strict json.loads. Stage 2 (only if
    stage 1 raises): escape literal control characters inside string values
    and retry. Raises json.JSONDecodeError if both stages fail, exactly like
    json.loads would on the original input.

    Logs at INFO level when stage 2 succeeded, so per-model format-compliance
    rates can be derived from logs."""
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as first_err:
        repaired = _escape_control_chars_in_strings(json_text)
        if repaired == json_text:
            raise  # Nothing to repair; re-raise the original error
        try:
            result = json.loads(repaired)
            logger.info(
                "extract_json: repaired control characters inside string values "
                "(stage-2 parse). Likely cause: model emitted literal newlines "
                "inside a multi-line string field."
            )
            return result
        except json.JSONDecodeError:
            raise first_err  # Original error is more informative


def extract_json(text: str) -> dict:
    """Extract JSON from text that may contain markdown fences or preamble.

    Permissive against three common malformations:
      1. Markdown code fences (```json ... ``` or ``` ... ```)
      2. Preamble text (Response:/Output:/Answer:)
      3. Literal control characters inside JSON string values
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string")

    # Try fenced blocks first
    for pat in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        m = re.search(pat, text)
        if m:
            try:
                return _try_parse(m.group(1).strip())
            except json.JSONDecodeError:
                pass

    # Bracket-balanced scan
    cleaned = re.sub(r"^\s*(Response:|Output:|Answer:)\s*", "", text.strip(), flags=re.IGNORECASE)
    start = None
    for i, c in enumerate(cleaned):
        if c in "{[":
            start = i
            break
    if start is None:
        raise ValueError("No JSON found in response")

    stack, in_str, esc = [], False, False
    for j in range(start, len(cleaned)):
        c = cleaned[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                continue
            stack.pop()
            if not stack:
                try:
                    return _try_parse(cleaned[start:j+1])
                except json.JSONDecodeError:
                    break
    raise ValueError("No valid JSON found in response")

# ---------------------------------------------------------------------------
# Provider: vLLM (local OpenAI-compatible server)
# ---------------------------------------------------------------------------
@retry_with_backoff(max_retries=3)
def generate_vllm(model_name: str, prompt: str, system_prompt: str = "",
                  port: int = 8000, max_tokens: int = 2048,
                  temperature: float = 0.0, think_budget: int = 0,
                  user_prefix: str = "",
                  **kwargs) -> Tuple[str, dict]:
    """Call a local vLLM server via OpenAI-compatible API.

    When think_budget > 0, enables thinking mode for models that support it.
    - Dedicated thinking models (Qwen3-*-Thinking): always think, no extra params needed.
      Output contains thinking followed by </think> then the answer.
    - Regular Qwen3 models: toggled via chat_template_kwargs.

    Server must be started with --enable-reasoning for thinking models:
      Qwen3-*-Thinking: --enable-reasoning --reasoning-parser deepseek_r1
      Qwen3-* (regular): --enable-reasoning --reasoning-parser qwen3
    """
    from openai import OpenAI

    if user_prefix:
        prompt = user_prefix + "\n" + prompt

    # Max-reasoning local judges can take many minutes per call; the OpenAI
    # default read timeout (600s) is too short once a few long generations
    # share one vLLM server.
    # Optional host for cross-node vLLM (default localhost / VLLM_HOST).
    host = kwargs.pop("host", None) or os.environ.get("VLLM_HOST", "localhost")
    client = OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="dummy",
        timeout=3600.0,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Keep model-card sampling recipes reproducible. ``top_k`` is a vLLM
    # extension, while ``top_p`` / ``seed`` are part of the OpenAI-compatible request.
    top_p = kwargs.get("top_p")
    top_k = kwargs.get("top_k")
    seed = kwargs.get("seed")
    if top_p is not None:
        params["top_p"] = top_p
    if top_k is not None:
        params.setdefault("extra_body", {})["top_k"] = top_k
    if seed is not None:
        params["seed"] = int(seed)

    if think_budget > 0:
        # GLM controls thinking through its top-level reasoning_effort field;
        # do not pass the Qwen-specific enable_thinking template argument.
        # Regular Qwen3 models need the explicit toggle; dedicated Thinking
        # models always think (chat template auto-includes <think>).
        if "glm" not in model_name.lower() and "Thinking" not in model_name:
            params.setdefault("extra_body", {})["chat_template_kwargs"] = {
                "enable_thinking": True,
            }
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort == "low" and "nemotron" in model_name.lower():
            # Nemotron-3 exposes low-effort reasoning through its chat-template
            # flag rather than Qwen3.8's top-level reasoning_effort control.
            params.setdefault("extra_body", {}).setdefault(
                "chat_template_kwargs", {}
            )["low_effort"] = True
        elif reasoning_effort == "xhigh":
            # vLLM 0.19 rejects the OpenAI-level `xhigh` literal. Omit it so
            # Qwen3.8's chat template defaults to xhigh; the caller still
            # records the intended effort in run metadata.
            pass
        elif reasoning_effort:
            # Qwen3.8 / GLM-5.3 accept this OpenAI-compatible field. Qwen
            # defaults to xhigh when unset; GLM-5.3-Flash defaults to max.
            params["reasoning_effort"] = reasoning_effort
        # Thinking models need non-zero temperature
        if temperature == 0.0:
            params["temperature"] = 0.6

    resp = client.chat.completions.create(**params)
    usage = resp.usage
    token_info = {
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }

    message = resp.choices[0].message
    content = message.content
    finish_reason = getattr(resp.choices[0], "finish_reason", None)
    if finish_reason:
        token_info["finish_reason"] = finish_reason

    # If vLLM was started with --reasoning-parser, thinking is already
    # separated into reasoning_content; content has only the answer.
    reasoning = getattr(message, "reasoning_content", None)
    if not reasoning:
        extra = getattr(message, "model_extra", None) or {}
        if isinstance(extra, dict):
            reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    api_reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
    if api_reasoning_tokens is None and isinstance(details, dict):
        api_reasoning_tokens = details.get("reasoning_tokens")

    # Fallback: Qwen3-*-Thinking outputs thinking then </think> then answer.
    # The opening <think> is injected by the chat template, so the model
    # output has NO opening tag — only </think> as the delimiter.
    if not reasoning and content and "</think>" in content:
        parts = content.split("</think>", 1)
        reasoning = parts[0]
        content = parts[1].strip() if len(parts) > 1 else ""

    if reasoning:
        token_info["reasoning_text"] = reasoning
        token_info["reasoning_tokens"] = (
            int(api_reasoning_tokens) if api_reasoning_tokens else len(reasoning.split())
        )
    elif api_reasoning_tokens:
        token_info["reasoning_tokens"] = int(api_reasoning_tokens)

    if not content:
        raise APIError("Empty response from vLLM")
    return content, token_info
    
# ---------------------------------------------------------------------------
# Provider: OpenAI API (GPT-4o, GPT-5, etc.)
# ---------------------------------------------------------------------------
def _uses_responses_api(model_name: str) -> bool:
    """GPT-5 and GPT-6 reject temperature once reasoning effort is set."""
    name = model_name.lower()
    return "gpt-5" in name or "gpt-6" in name

@retry_with_backoff(max_retries=3)
def generate_openai(model_name: str, prompt: str, system_prompt: str = "",
                    max_tokens: int = 2048, temperature: float = 0.0,
                    reasoning_effort: str = "", user_prefix: str = "", **kwargs) -> Tuple[str, dict]:
    """Call OpenAI API. Uses Responses API for GPT-5, Chat Completions for others."""
    from openai import OpenAI

    if user_prefix:
        prompt = user_prefix + "\n" + prompt

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ModelNotAvailableError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    if _uses_responses_api(model_name):
        # GPT-5 uses the Responses API
        input_msgs = []
        if system_prompt:
            input_msgs.append({"role": "system", "content": system_prompt})
        input_msgs.append({"role": "user", "content": prompt})

        params = {
            "model": model_name,
            "input": input_msgs,
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "text"}},
        }
        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}
            # Reasoning models reject temperature
        else:
            params["temperature"] = temperature

        resp = client.responses.create(**params)
        usage = resp.usage
        token_info = {
            "input_tokens": usage.input_tokens if usage else 0,
            "output_tokens": usage.output_tokens if usage else 0,
            "cached_tokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0),
        }
        content = resp.output_text
        if not content:
            raise APIError("Empty response from OpenAI Responses API")
        return content, token_info

    else:
        # GPT-4o and older use Chat Completions API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = client.chat.completions.create(**params)
        usage = resp.usage
        token_info = {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "cached_tokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0),
        }
        content = resp.choices[0].message.content
        if not content:
            raise APIError("Empty response from OpenAI")
        return content, token_info

# ---------------------------------------------------------------------------
# Provider: Anthropic (Claude) API
# ---------------------------------------------------------------------------
# Models that support (and prefer) adaptive thinking
_ADAPTIVE_THINKING_MODELS = (
    "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-5-5",
)
# Models where manual budget_tokens is rejected outright
_ADAPTIVE_ONLY_MODELS = ("claude-opus-4-7", "claude-haiku-5-5")
# Claude 4.7 and later accept only the default temperature.
_OMIT_TEMPERATURE_MODELS = ("claude-haiku-5-5",)

@retry_with_backoff(max_retries=3)
def generate_claude(model_name: str, prompt: str, system_prompt: str = "",
                    max_tokens: int = 2048, temperature: float = 0.0,
                    think_budget: int = 0, thinking_effort: str = "high",
                    user_prefix: str = "", **kwargs) -> Tuple[str, dict]:
    """
    think_budget > 0 enables thinking. How it's applied depends on the model:
      - Opus 4.7:          adaptive only (budget_tokens rejected); uses thinking_effort
      - Sonnet 4.6 / Opus 4.6: adaptive mode with thinking_effort (manual deprecated)
      - Older models:      manual mode with budget_tokens (adaptive not supported)
    thinking_effort: effort level for adaptive mode — "low"|"medium"|"high"|"xhigh"|"max"
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ModelNotAvailableError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    if user_prefix:
        user_content = [
            {"type": "text", "text": user_prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = prompt

    params: Dict[str, Any] = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_content}],
    }

    if system_prompt:
        params["system"] = [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ]

    if think_budget > 0:
        is_adaptive_model = any(m in model_name for m in _ADAPTIVE_THINKING_MODELS)
        is_adaptive_only  = any(m in model_name for m in _ADAPTIVE_ONLY_MODELS)

        if is_adaptive_model:
            # Adaptive mode — recommended for Sonnet 4.6 / Opus 4.6, required for Opus 4.7
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": thinking_effort}
        else:
            # Older models (Sonnet 4.5, Haiku 4.5, etc.) — manual mode only
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": max(1024, think_budget),
            }

        if not any(m in model_name for m in _OMIT_TEMPERATURE_MODELS):
            params["temperature"] = 1.0  # required when thinking is active
    elif not any(m in model_name for m in _OMIT_TEMPERATURE_MODELS):
        params["temperature"] = temperature

    try:
        with client.messages.stream(**params) as stream:
            resp = stream.get_final_message()
    except anthropic.RateLimitError as e:
        raise RateLimitError(str(e)) from e
    except anthropic.APIStatusError as e:
        raise APIError(f"Anthropic API error {e.status_code}: {e.message}") from e

    usage = resp.usage
    token_info = {
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
        "cached_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }

    thinking_tokens = 0
    text_parts = []
    for block in resp.content:
        if block.type == "thinking":
            thinking_tokens += len(block.thinking.split())
        elif block.type == "text":
            text_parts.append(block.text)

    if thinking_tokens:
        token_info["reasoning_tokens"] = thinking_tokens

    content = "\n".join(text_parts).strip()
    if not content:
        raise APIError("Empty response from Anthropic")
    return content, token_info

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PROVIDER_REGISTRY = {
    "vllm": generate_vllm,
    "openai": generate_openai,
    "claude": generate_claude,
}

def generate(provider: str, **kwargs) -> Tuple[str, float, dict]:
    """
    Unified generation entry point.
    Returns: (response_text, elapsed_seconds, token_info_dict)
    """
    if provider not in PROVIDER_REGISTRY:
        raise ModelNotAvailableError(f"Unknown provider: {provider}. Options: {list(PROVIDER_REGISTRY.keys())}")
    return PROVIDER_REGISTRY[provider](**kwargs)
