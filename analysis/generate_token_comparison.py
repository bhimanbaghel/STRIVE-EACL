#!/usr/bin/env python3
"""
Compare generator output tokens: (w/oRe+ST) vs (wRes+ST) / (wRe+ST).

Run from project root:
    uv run python analysis/generate_token_comparison.py

Reads the released per-verb token counts and writes the comparison report.
Use --from-json to refresh the counts from the local generation and refinement
outputs. Generator totals exclude evaluator calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

MODEL_ORDER = ["gpt-5.1", "sonnet-4.6", "Qwen3-Next-80B", "Qwen3-30B", "Ministral-14B", "Qwen3-4B"]

# (model, strategy) -> (result_type, run_name) — v7 family only
GENERATION_PATHS_V7 = {
    ("gpt-5.1",        "v7_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_60"),
    ("sonnet-4.6",     "v7_60"): ("intermediate", "claude-sonnet-4-6_v7_60"),
    ("Qwen3-Next-80B", "v7_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_60"),
    ("Qwen3-30B",      "v7_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_60"),
    ("Ministral-14B",  "v7_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_60"),
    ("Qwen3-4B",       "v7_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_60"),
    ("gpt-5.1",        "v7_agent_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_agent_60"),
    ("sonnet-4.6",     "v7_agent_60"): ("intermediate", "claude-sonnet-4-6_v7_agent_60"),
    ("Qwen3-Next-80B", "v7_agent_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_agent_60"),
    ("Qwen3-30B",      "v7_agent_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_agent_60"),
    ("Ministral-14B",  "v7_agent_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_agent_60"),
    ("Qwen3-4B",       "v7_agent_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_agent_60"),
    ("gpt-5.1",        "v7_verbalized_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_verbalized_60"),
    ("sonnet-4.6",     "v7_verbalized_60"): ("intermediate", "claude-sonnet-4-6_v7_verbalized_60"),
    ("Qwen3-Next-80B", "v7_verbalized_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_verbalized_60"),
    ("Qwen3-30B",      "v7_verbalized_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_verbalized_60"),
    ("Ministral-14B",  "v7_verbalized_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_verbalized_60"),
    ("Qwen3-4B",       "v7_verbalized_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_verbalized_60"),
    ("gpt-5.1",        "v7_agent_verbalized_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_agent_verbalized_60"),
    ("sonnet-4.6",     "v7_agent_verbalized_60"): ("intermediate", "claude-sonnet-4-6_v7_agent_verbalized_60"),
    ("Qwen3-Next-80B", "v7_agent_verbalized_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_agent_verbalized_60"),
    ("Qwen3-30B",      "v7_agent_verbalized_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_agent_verbalized_60"),
    ("Ministral-14B",  "v7_agent_verbalized_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_agent_verbalized_60"),
    ("Qwen3-4B",       "v7_agent_verbalized_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_agent_verbalized_60"),
    ("gpt-5.1",        "v7_agent_iterative_60"): ("iterative", "gpt-5.1-2025-11-13_v7_agent_iterative_60"),
    ("sonnet-4.6",     "v7_agent_iterative_60"): ("iterative", "claude-sonnet-4-6_v7_agent_iterative_60"),
    ("Qwen3-Next-80B", "v7_agent_iterative_60"): ("iterative", "Qwen3-Next-80B-A3B-Instruct_v7_agent_iterative_60"),
    ("Qwen3-30B",      "v7_agent_iterative_60"): ("iterative", "Qwen3-30B-A3B-Instruct-2507_v7_agent_iterative_60"),
    ("Ministral-14B",  "v7_agent_iterative_60"): ("iterative", "Ministral-3-14B-Instruct-2512_v7_agent_iterative_60"),
    ("Qwen3-4B",       "v7_agent_iterative_60"): ("iterative", "Qwen3-4B-Instruct-2507_v7_agent_iterative_60"),
    ("gpt-5.1",        "v7_agent_verbalized_iterative_60"): ("iterative", "gpt-5.1-2025-11-13_v7_agent_verbalized_iterative_60"),
    ("sonnet-4.6",     "v7_agent_verbalized_iterative_60"): ("iterative", "claude-sonnet-4-6_v7_agent_verbalized_iterative_60"),
    ("Qwen3-Next-80B", "v7_agent_verbalized_iterative_60"): ("iterative", "Qwen3-Next-80B-A3B-Instruct_v7_agent_verbalized_iterative_60"),
    ("Qwen3-30B",      "v7_agent_verbalized_iterative_60"): ("iterative", "Qwen3-30B-A3B-Instruct-2507_v7_agent_verbalized_iterative_60"),
    ("Ministral-14B",  "v7_agent_verbalized_iterative_60"): ("iterative", "Ministral-3-14B-Instruct-2512_v7_agent_verbalized_iterative_60"),
    ("Qwen3-4B",       "v7_agent_verbalized_iterative_60"): ("iterative", "Qwen3-4B-Instruct-2507_v7_agent_verbalized_iterative_60"),
}

STRATEGY_LABEL = {
    "v7_60": "RB (w/oRe+ST)",
    "v7_agent_60": "RB (wRes+ST)",
    "v7_verbalized_60": "R2V (w/oRe+ST)",
    "v7_agent_verbalized_60": "R2V (wRes+ST)",
    "v7_agent_iterative_60": "R2R (wRe+ST)",
    "v7_agent_verbalized_iterative_60": "R2VR (wRe+ST)",
}
V7_STRATEGIES = set(STRATEGY_LABEL)
ITERATIVE_STRATEGIES = {"v7_agent_iterative_60", "v7_agent_verbalized_iterative_60"}

# baseline (w/oRe+ST) → with reasoning / iterative
REASONING_PAIRS = [
    ("RB", "v7_60", "v7_agent_60"),
    ("R2V", "v7_verbalized_60", "v7_agent_verbalized_60"),
    ("R2R", "v7_60", "v7_agent_iterative_60"),
    ("R2VR", "v7_verbalized_60", "v7_agent_verbalized_iterative_60"),
]

TOKEN_COUNTS_CSV = "results/aggregate/generation_token_counts.csv"
OUTPUT_MD = "results/aggregate/generation_token_comparison.md"
IDENTIFIER_COLUMNS = ("model", "strategy", "verb")
COUNT_COLUMNS = ("input_tokens", "output_tokens", "n_iterations", "output_tokens_pipeline")
TOKEN_COLUMNS = IDENTIFIER_COLUMNS + COUNT_COLUMNS


def _load_single_shot(run_name: str, verb: str) -> dict | None:
    path = os.path.join("results/intermediate", run_name, f"{verb}.json")
    if not os.path.exists(path):
        return None
    ti = json.load(open(path)).get("token_info") or {}
    out = ti.get("output_tokens")
    if out is None:
        return None
    return {
        "output_tokens": int(out),
        "input_tokens": int(ti.get("input_tokens") or 0),
        "reasoning_tokens": int(ti.get("reasoning_tokens") or 0),
        "n_api_calls": 1,
    }


def _load_iterative(run_name: str, verb: str, *, include_eval: bool) -> dict | None:
    verb_dir = os.path.join("results/iterative", run_name, verb)
    if not os.path.isdir(verb_dir):
        return None
    tot = {"output_tokens": 0, "input_tokens": 0, "reasoning_tokens": 0, "n_api_calls": 0}
    for fname in os.listdir(verb_dir):
        if not fname.startswith("iter_") or not fname.endswith(".json"):
            continue
        if not include_eval and fname.endswith("_evaluation.json"):
            continue
        ti = json.load(open(os.path.join(verb_dir, fname))).get("token_info") or {}
        tot["output_tokens"] += int(ti.get("output_tokens") or 0)
        tot["input_tokens"] += int(ti.get("input_tokens") or 0)
        tot["reasoning_tokens"] += int(ti.get("reasoning_tokens") or 0)
        tot["n_api_calls"] += 1
    return tot if tot["n_api_calls"] else None


def _iteration_count(run_name: str, verb: str) -> int:
    """Number of iterative rounds for one verb (from history.json or iter_* files)."""
    verb_dir = os.path.join("results/iterative", run_name, verb)
    hist_path = os.path.join(verb_dir, "history.json")
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        if h.get("total_iterations"):
            return max(1, int(h["total_iterations"]))
        if h.get("iterations"):
            return max(1, len(h["iterations"]))
    max_iter = -1
    if os.path.isdir(verb_dir):
        for fname in os.listdir(verb_dir):
            if not fname.startswith("iter_"):
                continue
            parts = fname.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                max_iter = max(max_iter, int(parts[1]))
    return max_iter + 1 if max_iter >= 0 else 1


def _effective_output(r: dict) -> float:
    """
    Tokens used for cross-strategy comparison.
    Single-shot: total output (one call). Iterative: total / n_iterations (per round).
    """
    if r["strategy"] in ITERATIVE_STRATEGIES:
        n = r.get("n_iterations") or 1
        return r["output_tokens"] / n
    return float(r["output_tokens"])


def load_verb_tokens(result_type: str, run_name: str, verb: str) -> dict | None:
    if result_type == "intermediate":
        row = _load_single_shot(run_name, verb)
        if row:
            row["n_iterations"] = 1
        return row
    gen = _load_iterative(run_name, verb, include_eval=False)
    full = _load_iterative(run_name, verb, include_eval=True)
    if not gen:
        return None
    n_iter = _iteration_count(run_name, verb)
    row = dict(gen)
    row["n_iterations"] = n_iter
    row["output_tokens_per_iter"] = gen["output_tokens"] / n_iter
    if full:
        row["output_tokens_pipeline"] = full["output_tokens"]
        row["output_tokens_pipeline_per_iter"] = full["output_tokens"] / n_iter
        row["n_api_calls_pipeline"] = full["n_api_calls"]
    return row


def list_verbs(result_type: str, run_name: str) -> list[str]:
    if result_type == "intermediate":
        base = os.path.join("results/intermediate", run_name)
        if not os.path.isdir(base):
            return []
        return sorted(f[:-5] for f in os.listdir(base) if f.endswith(".json"))
    base = os.path.join("results/iterative", run_name)
    if not os.path.isdir(base):
        return []
    return sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    )


def collect_records() -> list[dict]:
    records = []
    for (model, strategy), (result_type, run_name) in GENERATION_PATHS_V7.items():
        if strategy not in V7_STRATEGIES:
            continue
        for verb in list_verbs(result_type, run_name):
            ti = load_verb_tokens(result_type, run_name, verb)
            if not ti:
                continue
            records.append({
                "model": model,
                "strategy": strategy,
                "label": STRATEGY_LABEL[strategy],
                "verb": verb,
                **ti,
            })
    return records


def load_token_counts() -> list[dict]:
    records = []
    with open(TOKEN_COUNTS_CSV, newline="") as stream:
        for row in csv.DictReader(stream):
            record = {column: row[column] for column in IDENTIFIER_COLUMNS}
            for column in COUNT_COLUMNS:
                if row[column] != "":
                    record[column] = int(row[column])
            records.append(record)
    return records


def write_token_counts(records: list[dict]) -> None:
    with open(TOKEN_COUNTS_CSV, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TOKEN_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def _pct(part: float, whole: float) -> str:
    if whole == 0:
        return "—"
    return f"{100 * part / whole:.1f}%"


def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else float("nan")


def _median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else float("nan")


def build_tables(records: list[dict]) -> str:
    by_key = defaultdict(dict)
    for r in records:
        by_key[(r["model"], r["strategy"])][r["verb"]] = r

    lines = [
        "# Generation output token comparison",
        "",
        "Compares **(w/oRe+ST)** baselines to **(wRes+ST)** single-shot reasoning and "
        "**(wRe+ST)** iterative runs. Per-verb counts in "
        "`results/aggregate/generation_token_counts.csv` come from the saved calls' `token_info` fields.",
        "",
        "**Iterative runs:** generator totals sum `iter_*_generation.json` and "
        "`iter_*_refinement.json` only (GPT evaluator `iter_*_evaluation.json` excluded). "
        "For R2R/R2VR, cross-strategy comparisons use **output ÷ n_iterations** (mean per "
        "round from `history.json`), so multi-round runs are not inflated vs single-shot.",
        "",
        "---",
        "",
        "## 1. Output tokens by model × strategy (totals per verb)",
        "",
        "Per verb: **sum** of all generator calls in that experiment. "
        "Iterative rows also show mean rounds and per-iteration average.",
        "",
        "| Model | Strategy | Label | n | Mean total out | Median total | Mean iters | "
        "Mean out/iter | Mean input |",
        "|-------|----------|-------|---|----------------|--------------|------------|"
        "-------------|------------|",
    ]

    for model in MODEL_ORDER:
        for strategy in STRATEGY_LABEL:
            rows = [r for r in records if r["model"] == model and r["strategy"] == strategy]
            if not rows:
                continue
            outs = [r["output_tokens"] for r in rows]
            per_iter = [_effective_output(r) for r in rows]
            iters = [r.get("n_iterations", 1) for r in rows]
            ins = [r["input_tokens"] for r in rows]
            iter_col = f"{_mean(iters):.1f}" if strategy in ITERATIVE_STRATEGIES else "1.0"
            lines.append(
                f"| {model} | {strategy} | {STRATEGY_LABEL[strategy]} | {len(rows)} | "
                f"{_fmt(_mean(outs))} | {_fmt(_median(outs))} | {iter_col} | "
                f"{_fmt(_mean(per_iter))} | {_fmt(_mean(ins))} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 2. Extra output tokens vs (w/oRe+ST) baseline (fair comparison)",
        "",
        "Paired on the same **(model, verb)**.",
        "",
        "- **RB / R2V:** single-shot total output (one API call).",
        "- **R2R / R2VR:** **per-iteration** generator output (total ÷ `total_iterations`), "
        "compared to single-shot baseline.",
        "",
        "| Modality | Baseline | With reasoning / iterative | n pairs | Mean Δ | Median Δ "
        "| Mean ratio | % increase |",
        "|----------|----------|---------------------------|---------|--------|----------|"
        "|------------|------------|",
    ]

    pooled_norm = []
    for modality, base_s, with_s in REASONING_PAIRS:
        deltas = []
        for model in MODEL_ORDER:
            base_v = by_key.get((model, base_s), {})
            with_v = by_key.get((model, with_s), {})
            for verb in sorted(set(base_v) & set(with_v)):
                d = _effective_output(with_v[verb]) - _effective_output(base_v[verb])
                deltas.append(d)
                pooled_norm.append(d)
        if not deltas:
            lines.append(
                f"| {modality} | {STRATEGY_LABEL[base_s]} | {STRATEGY_LABEL[with_s]} | 0 | — | — | — | — |"
            )
            continue
        base_vals, with_vals = [], []
        for model in MODEL_ORDER:
            bv, wv = by_key.get((model, base_s), {}), by_key.get((model, with_s), {})
            for verb in sorted(set(bv) & set(wv)):
                base_vals.append(_effective_output(bv[verb]))
                with_vals.append(_effective_output(wv[verb]))
        base_mean, with_mean = _mean(base_vals), _mean(with_vals)
        lines.append(
            f"| {modality} | {STRATEGY_LABEL[base_s]} | {STRATEGY_LABEL[with_s]} | {len(deltas)} | "
            f"+{_fmt(_mean(deltas))} | +{_fmt(_median(deltas))} | "
            f"{with_mean / base_mean:.2f}× | +{_pct(with_mean - base_mean, base_mean)} |"
        )

    lines += [
        "",
        "### 2b. Mean Δ by model (normalized where applicable)",
        "",
        "| Modality | Model | n pairs | Mean Δ | Median Δ | Mean iters (with side) |",
        "|----------|-------|---------|--------|----------|------------------------|",
    ]
    for modality, base_s, with_s in REASONING_PAIRS:
        for model in MODEL_ORDER:
            base_v = by_key.get((model, base_s), {})
            with_v = by_key.get((model, with_s), {})
            verbs = sorted(set(base_v) & set(with_v))
            if not verbs:
                continue
            deltas = [_effective_output(with_v[v]) - _effective_output(base_v[v]) for v in verbs]
            mean_iters = _mean([with_v[v].get("n_iterations", 1) for v in verbs])
            lines.append(
                f"| {modality} | {model} | {len(deltas)} | +{_fmt(_mean(deltas))} | "
                f"+{_fmt(_median(deltas))} | {mean_iters:.1f} |"
            )

    lines += [
        "",
        "### 2c. Unnormalized iterative totals (reference only)",
        "",
        "Same pairs as §2 but R2R/R2VR use **summed** tokens across all rounds (inflated vs single-shot).",
        "",
        "| Modality | Mean Δ (total sum) | Median Δ |",
        "|----------|-------------------|----------|",
    ]
    for modality, base_s, with_s in REASONING_PAIRS:
        if with_s not in ITERATIVE_STRATEGIES:
            continue
        deltas = []
        for model in MODEL_ORDER:
            bv, wv = by_key.get((model, base_s), {}), by_key.get((model, with_s), {})
            for v in sorted(set(bv) & set(wv)):
                deltas.append(wv[v]["output_tokens"] - bv[v]["output_tokens"])
        if deltas:
            lines.append(
                f"| {modality} | +{_fmt(_mean(deltas))} | +{_fmt(_median(deltas))} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 3. Pooled summary (normalized comparison)",
        "",
    ]
    if pooled_norm:
        lines.append(
            f"- **Average additional output tokens (fair comparison):** "
            f"**+{_fmt(_mean(pooled_norm))}** per paired cell (median +{_fmt(_median(pooled_norm))}, "
            f"n={len(pooled_norm)}). R2R/R2VR use per-iteration generator output."
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Iterative pipeline totals (generator + GPT evaluator)",
        "",
        "For R2R / R2VR only: mean output tokens when **all** iterative API calls are included.",
        "",
        "| Model | Strategy | Mean gen-only out | Mean full pipeline out | Eval share |",
        "|-------|----------|-------------------|------------------------|------------|",
    ]
    for model in MODEL_ORDER:
        for strategy in ("v7_agent_iterative_60", "v7_agent_verbalized_iterative_60"):
            rows = [r for r in records if r["model"] == model and r["strategy"] == strategy]
            if not rows or "output_tokens_pipeline" not in rows[0]:
                continue
            gen = _mean([r["output_tokens"] for r in rows])
            full = _mean([r["output_tokens_pipeline"] for r in rows])
            lines.append(
                f"| {model} | {STRATEGY_LABEL[strategy]} | {_fmt(gen)} | {_fmt(full)} | "
                f"{_pct(full - gen, full)} of pipeline |"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Reproduce the generator token comparison.")
    parser.add_argument("--from-json", action="store_true", help="Refresh the token CSV from saved model calls.")
    args = parser.parse_args()

    refresh = args.from_json or not os.path.isfile(TOKEN_COUNTS_CSV)
    records = collect_records() if refresh else load_token_counts()
    if not records:
        raise SystemExit("No generator token records found.")

    body = build_tables(records)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    if refresh:
        write_token_counts(records)
    with open(OUTPUT_MD, "w") as f:
        f.write(body)
    print(f"Wrote {len(records)} verb-level records → {OUTPUT_MD}")

    # Console summary
    by_key = defaultdict(dict)
    for r in records:
        by_key[(r["model"], r["strategy"])][r["verb"]] = r
    print("\nMean additional output tokens vs (w/oRe+ST) [normalized for R2R/R2VR]:")
    for modality, base_s, with_s in REASONING_PAIRS:
        deltas = []
        for model in MODEL_ORDER:
            bv, wv = by_key.get((model, base_s), {}), by_key.get((model, with_s), {})
            for v in set(bv) & set(wv):
                deltas.append(_effective_output(wv[v]) - _effective_output(bv[v]))
        if deltas:
            print(f"  {modality}: +{statistics.mean(deltas):,.0f} mean  (+{statistics.median(deltas):,.0f} median, n={len(deltas)})")


if __name__ == "__main__":
    main()
