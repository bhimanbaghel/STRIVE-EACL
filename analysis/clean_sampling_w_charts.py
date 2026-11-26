import csv
import json
import math
import os
import glob
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, NoNorm

# ─────────────────────────────── chart constants ──────────────────────────────

STRATEGY_ORDER = [
    "v4_60",
    "v6_60",
    "v7_60",
    "v7_agent_60",
    "v6_verbalized_60",
    "v7_verbalized_60",
    "v7_agent_verbalized_60",
    "iterative_v6_verbalized_init_60",
    "iterative_v6_60",
    "v7_agent_iterative_60",
    "v7_agent_verbalized_iterative_60",
]
STRATEGY_LABELS = {
    "v4_60":                            "RB\n(w/oViz)",
    "v6_60":                            "RB\n(w/oRe)",
    "v7_60":                            "RB\n(w/oRe+ST)",
    "v7_agent_60":                      "RB\n(wRes+ST)",
    "v6_verbalized_60":                 "R2V\n(w/oRe)",
    "v7_verbalized_60":                 "R2V\n(w/oRe+ST)",
    "v7_agent_verbalized_60":           "R2V\n(wRes+ST)",
    "iterative_v6_verbalized_init_60":  "R2VR\n(w/oRe)",
    "iterative_v6_60":                  "R2R\n(w/oRe)",
    "v7_agent_iterative_60":            "R2R\n(wRe+ST)",
    "v7_agent_verbalized_iterative_60": "R2VR\n(wRe+ST)",
}
TIER_COLORS = {
    "GOLD":   "#BA7517",
    "SILVER": "#888780",
    "BRONZE": "#F0997B",
    "FAIL":   "#E8E6DF",
}

MODEL_ORDER = ["gpt-5.1", "sonnet-4.6", "Qwen3-Next-80B", "Qwen3-30B", "Ministral-14B", "Qwen3-4B"]

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ─────────────────────────────── data helpers ─────────────────────────────────

def parse_source_run(eval_run):
    """Extract clean model name and strategy from eval_run."""
    run = eval_run.replace("evalv3_", "")
    run = run.replace("evalv3.1_agent_", "")
    run = run.replace("evalv3.1_patient_", "")
    run = run.replace("evalv31agent_qwen30b-think_", "")
    run = run.replace("evalv31agent_sonnet-4.6_", "")

    model_map = {
        "gpt-5.1-2025-11-13":              "gpt-5.1",
        "Qwen3-Next-80B-A3B-Instruct":     "Qwen3-Next-80B",
        "Qwen3-30B-A3B-Instruct-2507":     "Qwen3-30B",
        "Ministral-3-14B-Instruct-2512":   "Ministral-14B",
        "Qwen3-4B-Instruct-2507":          "Qwen3-4B",
        "claude-sonnet-4-6":               "sonnet-4.6",
    }

    strategy = None
    for s in [
        "iterative_v6_verbalized_init_60", "iterative_v6_60", "iterative_v5",
        "v6_verbalized_60", "v6_60", "v5", "v4_60",
        "v7_agent_verbalized_iterative_60", "v7_patient_verbalized_iterative_60",
        "v7_patient_verbalized_60",
        "v7_agent_verbalized_60", "v7_verbalized_60",
        "v7_agent_iterative_60", "v7_patient_iterative_60",
        "v7_agent_60", "v7_patient_60", "v7_60",
    ]:
        if run.endswith(s) or run.endswith(s + "_finals"):
            strategy = s
            break

    model = None
    for messy, clean in model_map.items():
        if messy in run:
            model = clean
            break

    return model, strategy


GENERATION_PATHS = {
    # v4 single-shot
    ("gpt-5.1",        "v4_60"): ("intermediate", "gpt-5.1-2025-11-13_v4_60"),
    ("Qwen3-Next-80B", "v4_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v4_60"),
    ("Qwen3-30B",      "v4_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v4_60"),
    ("Ministral-14B",  "v4_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v4_60"),
    ("Qwen3-4B",       "v4_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v4_60"),
    # v6 single-shot
    ("gpt-5.1",        "v6_60"): ("intermediate", "gpt-5.1-2025-11-13_v6_60"),
    ("Qwen3-Next-80B", "v6_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v6_60"),
    ("Qwen3-30B",      "v6_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v6_60"),
    ("Ministral-14B",  "v6_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v6_60"),
    ("Qwen3-4B",       "v6_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v6_60"),
    # v7 single-shot
    ("gpt-5.1",        "v7_agent_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_agent_60"),
    ("sonnet-4.6",    "v7_agent_60"): ("intermediate", "claude-sonnet-4-6_v7_agent_60"),
    ("Qwen3-Next-80B", "v7_agent_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_agent_60"),
    ("Qwen3-30B",      "v7_agent_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_agent_60"),
    ("Ministral-14B",  "v7_agent_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_agent_60"),
    ("Qwen3-4B",       "v7_agent_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_agent_60"),
    # v7 single-shot No reasoning
    ("gpt-5.1",        "v7_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_60"),
    ("sonnet-4.6",    "v7_60"): ("intermediate", "claude-sonnet-4-6_v7_60"),
    ("Qwen3-Next-80B", "v7_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_60"),
    ("Qwen3-30B",      "v7_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_60"),
    ("Ministral-14B",  "v7_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_60"),
    ("Qwen3-4B",       "v7_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_60"),
    # v6_verbalized single-shot
    ("gpt-5.1",        "v6_verbalized_60"): ("intermediate", "gpt-5.1-2025-11-13_v6_verbalized_60"),
    ("Qwen3-Next-80B", "v6_verbalized_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v6_verbalized_60"),
    ("Qwen3-30B",      "v6_verbalized_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v6_verbalized_60"),
    ("Ministral-14B",  "v6_verbalized_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v6_verbalized_60"),
    ("Qwen3-4B",       "v6_verbalized_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v6_verbalized_60"),
    # v7_verbalized single-shot
    ("gpt-5.1",        "v7_agent_verbalized_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_agent_verbalized_60"),
    ("sonnet-4.6",    "v7_agent_verbalized_60"): ("intermediate", "claude-sonnet-4-6_v7_agent_verbalized_60"),
    ("Qwen3-Next-80B", "v7_agent_verbalized_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_agent_verbalized_60"),
    ("Qwen3-30B",      "v7_agent_verbalized_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_agent_verbalized_60"),
    ("Ministral-14B",  "v7_agent_verbalized_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_agent_verbalized_60"),
    ("Qwen3-4B",       "v7_agent_verbalized_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_agent_verbalized_60"),
    # v7_verbalized single-shot No reasoning
    ("gpt-5.1",        "v7_verbalized_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_verbalized_60"),
    ("sonnet-4.6",    "v7_verbalized_60"): ("intermediate", "claude-sonnet-4-6_v7_verbalized_60"),
    ("Qwen3-Next-80B", "v7_verbalized_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_verbalized_60"),
    ("Qwen3-30B",      "v7_verbalized_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_verbalized_60"),
    ("Ministral-14B",  "v7_verbalized_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_verbalized_60"),
    ("Qwen3-4B",       "v7_verbalized_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_verbalized_60"),
    # v7_patient single-shot (patient-gradient stimuli; same models as v7 / v7_agent)
    ("gpt-5.1",        "v7_patient_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_patient_60"),
    ("sonnet-4.6",    "v7_patient_60"): ("intermediate", "claude-sonnet-4-6_v7_patient_60"),
    ("Qwen3-Next-80B", "v7_patient_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_patient_60"),
    ("Qwen3-30B",      "v7_patient_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_patient_60"),
    ("Ministral-14B",  "v7_patient_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_patient_60"),
    ("Qwen3-4B",       "v7_patient_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_patient_60"),
    ("gpt-5.1",        "v7_patient_verbalized_60"): ("intermediate", "gpt-5.1-2025-11-13_v7_patient_verbalized_60"),
    ("sonnet-4.6",    "v7_patient_verbalized_60"): ("intermediate", "claude-sonnet-4-6_v7_patient_verbalized_60"),
    ("Qwen3-Next-80B", "v7_patient_verbalized_60"): ("intermediate", "Qwen3-Next-80B-A3B-Instruct_v7_patient_verbalized_60"),
    ("Qwen3-30B",      "v7_patient_verbalized_60"): ("intermediate", "Qwen3-30B-A3B-Instruct-2507_v7_patient_verbalized_60"),
    ("Ministral-14B",  "v7_patient_verbalized_60"): ("intermediate", "Ministral-3-14B-Instruct-2512_v7_patient_verbalized_60"),
    ("Qwen3-4B",       "v7_patient_verbalized_60"): ("intermediate", "Qwen3-4B-Instruct-2507_v7_patient_verbalized_60"),
    # iterative_v6
    ("gpt-5.1",        "iterative_v6_60"): ("iterative", "gpt-5.1-2025-11-13_iterative_v6_60"),
    ("Qwen3-Next-80B", "iterative_v6_60"): ("iterative", "Qwen3-Next-80B-A3B-Instruct_iterative_v6_60"),
    ("Qwen3-30B",      "iterative_v6_60"): ("iterative", "Qwen3-30B-A3B-Instruct-2507_iterative_v6_60"),
    ("Ministral-14B",  "iterative_v6_60"): ("iterative", "Ministral-3-14B-Instruct-2512_iterative_v6_60"),
    ("Qwen3-4B",       "iterative_v6_60"): ("iterative", "Qwen3-4B-Instruct-2507_iterative_v6_60"),
    # iterative_v6_verbalized_init
    ("gpt-5.1",        "iterative_v6_verbalized_init_60"): ("iterative", "gpt-5.1-2025-11-13_iterative_v6_verbalized_init_60"),
    ("Qwen3-Next-80B", "iterative_v6_verbalized_init_60"): ("iterative", "Qwen3-Next-80B-A3B-Instruct_iterative_v6_verbalized_init_60"),
    ("Qwen3-30B",      "iterative_v6_verbalized_init_60"): ("iterative", "Qwen3-30B-A3B-Instruct-2507_iterative_v6_verbalized_init_60"),
    ("Ministral-14B",  "iterative_v6_verbalized_init_60"): ("iterative", "Ministral-3-14B-Instruct-2512_iterative_v6_verbalized_init_60"),
    ("Qwen3-4B",       "iterative_v6_verbalized_init_60"): ("iterative", "Qwen3-4B-Instruct-2507_iterative_v6_verbalized_init_60"),
    # v7_agent_iterative
    ("gpt-5.1",        "v7_agent_iterative_60"): ("iterative", "gpt-5.1-2025-11-13_v7_agent_iterative_60"),
    ("sonnet-4.6",    "v7_agent_iterative_60"): ("iterative", "claude-sonnet-4-6_v7_agent_iterative_60"),
    ("Qwen3-Next-80B", "v7_agent_iterative_60"): ("iterative", "Qwen3-Next-80B-A3B-Instruct_v7_agent_iterative_60"),
    ("Qwen3-30B",      "v7_agent_iterative_60"): ("iterative", "Qwen3-30B-A3B-Instruct-2507_v7_agent_iterative_60"),
    ("Ministral-14B",  "v7_agent_iterative_60"): ("iterative", "Ministral-3-14B-Instruct-2512_v7_agent_iterative_60"),
    ("Qwen3-4B",       "v7_agent_iterative_60"): ("iterative", "Qwen3-4B-Instruct-2507_v7_agent_iterative_60"),
    # v7_agent_verbalized_iterative
    ("gpt-5.1",        "v7_agent_verbalized_iterative_60"): ("iterative", "gpt-5.1-2025-11-13_v7_agent_verbalized_iterative_60"),
    ("sonnet-4.6",    "v7_agent_verbalized_iterative_60"): ("iterative", "claude-sonnet-4-6_v7_agent_verbalized_iterative_60"),
    ("Qwen3-Next-80B", "v7_agent_verbalized_iterative_60"): ("iterative", "Qwen3-Next-80B-A3B-Instruct_v7_agent_verbalized_iterative_60"),
    ("Qwen3-30B",      "v7_agent_verbalized_iterative_60"): ("iterative", "Qwen3-30B-A3B-Instruct-2507_v7_agent_verbalized_iterative_60"),
    ("Ministral-14B",  "v7_agent_verbalized_iterative_60"): ("iterative", "Ministral-3-14B-Instruct-2512_v7_agent_verbalized_iterative_60"),
    ("Qwen3-4B",       "v7_agent_verbalized_iterative_60"): ("iterative", "Qwen3-4B-Instruct-2507_v7_agent_verbalized_iterative_60"),
    # v7_patient iterative (same layout as v7_agent_iterative under results/iterative/)
    ("gpt-5.1",        "v7_patient_iterative_60"): ("iterative", "gpt-5.1-2025-11-13_v7_patient_iterative_60"),
    ("sonnet-4.6",    "v7_patient_iterative_60"): ("iterative", "claude-sonnet-4-6_v7_patient_iterative_60"),
    ("Qwen3-Next-80B", "v7_patient_iterative_60"): ("iterative", "Qwen3-Next-80B-A3B-Instruct_v7_patient_iterative_60"),
    ("Qwen3-30B",      "v7_patient_iterative_60"): ("iterative", "Qwen3-30B-A3B-Instruct-2507_v7_patient_iterative_60"),
    ("Ministral-14B",  "v7_patient_iterative_60"): ("iterative", "Ministral-3-14B-Instruct-2512_v7_patient_iterative_60"),
    ("Qwen3-4B",       "v7_patient_iterative_60"): ("iterative", "Qwen3-4B-Instruct-2507_v7_patient_iterative_60"),
    ("gpt-5.1",        "v7_patient_verbalized_iterative_60"): ("iterative", "gpt-5.1-2025-11-13_v7_patient_verbalized_iterative_60"),
    ("sonnet-4.6",    "v7_patient_verbalized_iterative_60"): ("iterative", "claude-sonnet-4-6_v7_patient_verbalized_iterative_60"),
    ("Qwen3-Next-80B", "v7_patient_verbalized_iterative_60"): ("iterative", "Qwen3-Next-80B-A3B-Instruct_v7_patient_verbalized_iterative_60"),
    ("Qwen3-30B",      "v7_patient_verbalized_iterative_60"): ("iterative", "Qwen3-30B-A3B-Instruct-2507_v7_patient_verbalized_iterative_60"),
    ("Ministral-14B",  "v7_patient_verbalized_iterative_60"): ("iterative", "Ministral-3-14B-Instruct-2512_v7_patient_verbalized_iterative_60"),
    ("Qwen3-4B",       "v7_patient_verbalized_iterative_60"): ("iterative", "Qwen3-4B-Instruct-2507_v7_patient_verbalized_iterative_60"),
}


def load_generation(model, strategy, verb):
    """Load the generated output (scene + agents + sentences) for a given model/strategy/verb."""
    key = (model, strategy)
    if key not in GENERATION_PATHS:
        return None

    result_type, run_name = GENERATION_PATHS[key]

    if result_type == "intermediate":
        path = os.path.join("results", "intermediate", run_name, f"{verb}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        if "parsed" in data:
            return data["parsed"]
        elif "scene" in data:
            return data
        return None

    elif result_type == "iterative":
        path = os.path.join("results", "iterative", run_name, verb, "final_output.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    return None


# Eval-run prefixes for judge-specific aggregation (tables + per-judge strategy charts)
EVAL_RUN_PREFIX_GPT = "evalv3.1_agent_"
EVAL_RUN_PREFIX_QWEN = "evalv31agent_qwen30b-think_"
EVAL_RUN_PREFIX_SONNET = "evalv31agent_sonnet-4.6_"
JUDGE_TABLE_LABELS = {
    EVAL_RUN_PREFIX_GPT:    "GPT judge (`evalv3.1_agent_`)",
    EVAL_RUN_PREFIX_QWEN:   "Qwen3-30B-Thinking judge (`evalv31agent_qwen30b-think_`)",
    EVAL_RUN_PREFIX_SONNET: "Claude Sonnet 4.6 judge (`evalv31agent_sonnet-4.6_`)",
}


def _dedupe_rows_by_eval_prefix(rows, eval_prefix, strategies=None):
    """Filter merged CSV rows to one eval prefix; dedupe on (model, strategy, verb)."""
    if strategies is None:
        strategies = ANNOTATION_STRATEGIES
    seen, out = set(), []
    for r in rows:
        eval_run = r["eval_run"]
        if not eval_run.startswith(eval_prefix):
            continue
        model, strategy = parse_source_run(eval_run)
        if not model or not strategy or strategy not in strategies:
            continue
        key = (model, strategy, r["verb"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "model":    model,
            "strategy": strategy,
            "verb":     r["verb"],
            "tier":     r["tier"],
        })
    return out


def _build_strategy_and_model_groups(filtered_rows):
    strategy_groups, model_strategy_groups = {}, {}
    for r in filtered_rows:
        strategy_groups.setdefault(r["strategy"], []).append(r)
        model_strategy_groups.setdefault((r["model"], r["strategy"]), []).append(r)
    return strategy_groups, model_strategy_groups


def _fmt_count_pct(count, total):
    if total == 0:
        return "—"
    return f"{count} ({count / total * 100:.1f}%)"


def _strategy_tier_markdown_table(strategy_groups):
    """Markdown table: tier counts and % per strategy (all models combined)."""
    strategies = [s for s in STRATEGY_ORDER if s in strategy_groups]
    lines = [
        "| Strategy | Label | n | GOLD | SILVER | BRONZE | FAIL |",
        "|----------|-------|---|------|--------|--------|------|",
    ]
    for s in strategies:
        items = strategy_groups[s]
        n = len(items)
        c = Counter(r["tier"] for r in items)
        lines.append(
            "| "
            + " | ".join([
                s,
                STRATEGY_LABELS.get(s, s).replace("\n", " "),
                str(n),
                _fmt_count_pct(c.get("GOLD", 0), n),
                _fmt_count_pct(c.get("SILVER", 0), n),
                _fmt_count_pct(c.get("BRONZE", 0), n),
                _fmt_count_pct(c.get("FAIL", 0), n),
            ])
            + " |"
        )
    return "\n".join(lines)


def _model_heatmap_markdown_table(model_strategy_groups):
    """Markdown table: GOLD count/total (GOLD%) per model × strategy."""
    strategies = [
        s for s in STRATEGY_ORDER
        if any((m, s) in model_strategy_groups for m in MODEL_ORDER)
    ]
    col_labels = [STRATEGY_LABELS.get(s, s).replace("\n", " ") for s in strategies]
    lines = [
        "| Model | " + " | ".join(col_labels) + " |",
        "|-------| " + " | ".join("---:" for _ in strategies) + " |",
    ]
    for model in MODEL_ORDER:
        cells = [model]
        for s in strategies:
            key = (model, s)
            if key not in model_strategy_groups:
                cells.append("—")
                continue
            items = model_strategy_groups[key]
            n = len(items)
            gold = sum(1 for r in items if r["tier"] == "GOLD")
            cells.append(f"{gold}/{n} ({gold / n * 100:.1f}%)")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_judge_summary_markdown(
    judge_label, strategy_groups, model_strategy_groups, output_path,
):
    """Write strategy tier distribution + model GOLD heatmap tables for one judge."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n_rows = sum(len(v) for v in strategy_groups.values())
    body = [
        f"# {judge_label}",
        "",
        f"Rows: **{n_rows}** unique (model, strategy, verb) cells after deduplication.",
        "",
        "## Strategy tier distribution (all models combined)",
        "",
        "Counts and share of stimuli per tier.",
        "",
        _strategy_tier_markdown_table(strategy_groups),
        "",
        "## Model × strategy GOLD heatmap",
        "",
        "Each cell: GOLD count / total verbs (GOLD%). Missing combinations shown as —.",
        "",
        _model_heatmap_markdown_table(model_strategy_groups),
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(body))
    print(f"Saved judge summary tables: {output_path}")


# ─────────────────────────────── chart functions ──────────────────────────────

def _draw_strategy_bars(ax, strategy_groups, title=None):
    """100% stacked bar chart: tier distribution per strategy."""
    strategies = [s for s in STRATEGY_ORDER if s in strategy_groups]
    labels     = [STRATEGY_LABELS[s] for s in strategies]

    totals, pct = {}, {}
    for s in strategies:
        items    = strategy_groups[s]
        n        = len(items)
        totals[s] = n
        c        = Counter(r["tier"] for r in items)
        pct[s]   = {t: c.get(t, 0) / n * 100 for t in ["GOLD", "SILVER", "BRONZE", "FAIL"]}

    x       = np.arange(len(strategies))
    bar_w   = 0.56
    bottoms = np.zeros(len(strategies))

    for tier in ["GOLD", "SILVER", "BRONZE", "FAIL"]:
        vals = np.array([pct[s][tier] for s in strategies])
        bars = ax.bar(
            x, vals, bar_w, bottom=bottoms,
            color=TIER_COLORS[tier], label=tier,
            edgecolor="white", linewidth=0.8, zorder=2,
        )
        # GOLD% label inside segment when tall enough
        if tier == "GOLD":
            for i, (bar, v) in enumerate(zip(bars, vals)):
                if v >= 5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottoms[i] + v / 2,
                        f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white", zorder=3,
                    )
                elif v > 0:
                    # Small GOLD value: annotate above the full bar
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        100 + 1.5,
                        f"{v:.1f}%",
                        ha="center", va="bottom",
                        fontsize=7.5, color=TIER_COLORS["GOLD"], zorder=3,
                    )
        bottoms += vals

    # n= below each bar
    for i, s in enumerate(strategies):
        ax.text(x[i], -5, f"n={totals[s]}", ha="center", va="top",
                fontsize=7, color="#888780")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, linespacing=1.3)
    ax.set_ylim(-10, 114)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color="#888780")
    ax.set_ylabel("% of stimuli", fontsize=9, color="#555553")
    ax.yaxis.grid(True, color="#E8E6DF", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#E8E6DF")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, colors="#333333", pad=4)
    ax.tick_params(axis="y", length=0)

    handles = [mpatches.Patch(color=TIER_COLORS[t], label=t)
               for t in ["GOLD", "SILVER", "BRONZE", "FAIL"]]
    ax.legend(
        handles=handles, loc="upper left", fontsize=8, frameon=False,
        ncol=4, bbox_to_anchor=(0, 1.14), handlelength=1.2, handletextpad=0.5,
    )
    ax.set_title(
        title or "Tier distribution by strategy  (all models combined)",
        fontsize=9.5, color="#333333", loc="left", pad=28,
    )


def _draw_model_heatmap(ax, model_strategy_groups):
    """GOLD% heatmap: rows = models, columns = strategies."""
    strategies = [s for s in STRATEGY_ORDER
                  if any((m, s) in model_strategy_groups for m in MODEL_ORDER)]
    models     = MODEL_ORDER

    matrix = np.full((len(models), len(strategies)), np.nan)
    gold_n = np.full((len(models), len(strategies), 2), np.nan)  # [:,:,0]=gold, [:,:,1]=n
    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            key = (model, strategy)
            if key in model_strategy_groups:
                items    = model_strategy_groups[key]
                n        = len(items)
                gold     = sum(1 for r in items if r["tier"] == "GOLD")
                matrix[i, j] = gold / n * 100
                gold_n[i, j, 0] = gold
                gold_n[i, j, 1] = n

    # Colormap: very light amber → dark amber
    cmap = LinearSegmentedColormap.from_list(
        "amber_heat",
        ["#FAEEDA", "#FAC775", "#EF9F27", "#BA7517", "#854F0B"],
        N=256,
    )
    cmap.set_bad("#F1EFE8")   # NaN / missing → near-white

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Cell text
    for i in range(len(models)):
        for j in range(len(strategies)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10.5, color="#888780")
            else:
                g, ncell = int(gold_n[i, j, 0]), int(gold_n[i, j, 1])
                tc = "#412402"
                fw = "bold" if v >= 40 else "normal"
                ax.text(
                    j, i, f"{g}/{ncell}\n{v:.1f}%",
                    ha="center", va="center", fontsize=8, fontweight=fw, color=tc,
                    linespacing=1.25,
                )

    # Cell grid lines (white separators)
    for xi in np.arange(-0.5, len(strategies), 1):
        ax.axvline(xi, color="white", linewidth=2)
    for yi in np.arange(-0.5, len(models), 1):
        ax.axhline(yi, color="white", linewidth=2)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(
        [STRATEGY_LABELS.get(s, s) for s in strategies],
        fontsize=8.5, linespacing=1.3,
    )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.tick_params(length=0)
    ax.set_title("GOLD% by model × strategy", fontsize=9.5, color="#333333",
                 loc="left", pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label("GOLD %", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)


# Column order: agent / patient pairs for each modality (missing evals show "—").
V7_AGENT_VS_PATIENT_HEATMAP_COLUMNS = [
    ("v7_agent_60", "RB\n(agent)"),
    ("v7_patient_60", "RB\n(patient)"),
    ("v7_agent_verbalized_60", "R2V\n(agent)"),
    ("v7_patient_verbalized_60", "R2V\n(patient)"),
    ("v7_agent_iterative_60", "R2R\n(agent)"),
    ("v7_patient_iterative_60", "R2R\n(patient)"),
    ("v7_agent_verbalized_iterative_60", "R2VR\n(agent)"),
    ("v7_patient_verbalized_iterative_60", "R2VR\n(patient)"),
]

# (panel title, agent strategy, patient strategy) — used for per-verb GOLD overlap charts
V7_AGENT_PATIENT_FRAMING_PAIRS = [
    ("RB — single-shot", "v7_agent_60", "v7_patient_60"),
    ("R2V — verbalized single-shot", "v7_agent_verbalized_60", "v7_patient_verbalized_60"),
    ("R2R — iterative", "v7_agent_iterative_60", "v7_patient_iterative_60"),
    ("R2VR — verbalized iterative", "v7_agent_verbalized_iterative_60", "v7_patient_verbalized_iterative_60"),
]
# verb_framing_agent_vs_patient_gold.png: single-shot modalities only, no gpt-5.1
VERB_FRAMING_AGENT_VS_PATIENT_PAIRS = V7_AGENT_PATIENT_FRAMING_PAIRS[:2]
VERB_FRAMING_AGENT_VS_PATIENT_MODELS = [m for m in MODEL_ORDER if m != "gpt-5.1"]

# Discrete codes for agent-vs-patient GOLD comparison (see plot_verb_framing_agent_vs_patient)
_FRAMING_BOTH_GOLD = 0
_FRAMING_AGENT_GOLD_ONLY = 1
_FRAMING_PATIENT_GOLD_ONLY = 2
_FRAMING_NEITHER_GOLD = 3
_FRAMING_INCOMPLETE = 4
_FRAMING_COLORS = (
    "#117733",  # both GOLD
    "#0077BB",  # side-A GOLD only
    "#CC3311",  # side-B GOLD only
    "#E0E0E0",  # both evaluated, neither GOLD
    "#555555",  # missing eval on one side
)

# Judge comparison: evalv3.1_agent_ (GPT), evalv31agent_qwen30b-think_ (Qwen), evalv31agent_sonnet-4.6_ (Sonnet)
EVAL_JUDGE_GPT = "gpt"
EVAL_JUDGE_QWEN30B = "qwen30b"
EVAL_JUDGE_SONNET = "sonnet"
JUDGE_COMPARE_ORDER = [EVAL_JUDGE_GPT, EVAL_JUDGE_QWEN30B, EVAL_JUDGE_SONNET]
JUDGE_SHORT_NAMES = {
    EVAL_JUDGE_GPT: "GPT",
    EVAL_JUDGE_QWEN30B: "Qwen",
    EVAL_JUDGE_SONNET: "Sonnet",
}
JUDGE_DISPLAY_NAMES = {
    EVAL_JUDGE_GPT: "GPT",
    EVAL_JUDGE_QWEN30B: "Qwen3-30B-Thinking",
    EVAL_JUDGE_SONNET: "Claude Sonnet 4.6",
}
JUDGE_EVAL_PREFIXES = {
    EVAL_JUDGE_GPT: EVAL_RUN_PREFIX_GPT,
    EVAL_JUDGE_QWEN30B: EVAL_RUN_PREFIX_QWEN,
    EVAL_JUDGE_SONNET: EVAL_RUN_PREFIX_SONNET,
}
# Same generation strategies for all judges on v7_agent_* runs.
JUDGE_COMPARE_STRATEGIES = {
    "v7_agent_60",
    "v7_agent_verbalized_60",
    "v7_agent_iterative_60",
    "v7_agent_verbalized_iterative_60",
}
_JUDGE_HEATMAP_STRATEGY_ORDER = [
    "v7_agent_60",
    "v7_agent_verbalized_60",
    "v7_agent_iterative_60",
    "v7_agent_verbalized_iterative_60",
]
_JUDGE_HEATMAP_STRAT_SHORT = {
    "v7_agent_60": "RB",
    "v7_agent_verbalized_60": "R2V",
    "v7_agent_iterative_60": "R2R",
    "v7_agent_verbalized_iterative_60": "R2VR",
}


def _judge_heatmap_columns(judges):
    """Paired GOLD% heatmap columns: RB → R2V → R2R → R2VR × judges (strategy, judge_tag, xtick)."""
    cols = []
    for strategy in _JUDGE_HEATMAP_STRATEGY_ORDER:
        for judge_tag in judges:
            short = _JUDGE_HEATMAP_STRAT_SHORT[strategy]
            jname = JUDGE_SHORT_NAMES[judge_tag]
            cols.append((strategy, judge_tag, f"{short}\n({jname} judge)"))
    return cols


JUDGE_HEATMAP_COLUMNS = _judge_heatmap_columns([EVAL_JUDGE_GPT, EVAL_JUDGE_QWEN30B])
JUDGE_HEATMAP_COLUMNS_ALL = _judge_heatmap_columns(JUDGE_COMPARE_ORDER)
JUDGE_COMPARE_PAIRS = [
    (EVAL_JUDGE_GPT, EVAL_JUDGE_QWEN30B, "gpt_vs_qwen30b_think"),
    (EVAL_JUDGE_GPT, EVAL_JUDGE_SONNET, "gpt_vs_sonnet"),
    (EVAL_JUDGE_QWEN30B, EVAL_JUDGE_SONNET, "qwen30b_vs_sonnet"),
]
# Verb-level judge comparison: iterative modalities only (R2R / R2VR)
JUDGE_VERB_MODALITY_PAIRS = [
    ("R2R — iterative", "v7_agent_iterative_60"),
    ("R2VR — verbalized iterative", "v7_agent_verbalized_iterative_60"),
]
JUDGE_VERB_COMPARE_STRATEGIES = {s for _, s in JUDGE_VERB_MODALITY_PAIRS}
JUDGE_VERB_COMPARE_MODELS = ["gpt-5.1", "sonnet-4.6"]


def _draw_v7_agent_vs_patient_heatmap(ax, model_strategy_groups):
    """GOLD% heatmap: rows = models, columns = paired v7_agent vs v7_patient (RB/R2V/R2R/R2VR)."""
    col_specs = V7_AGENT_VS_PATIENT_HEATMAP_COLUMNS
    n_cols = len(col_specs)
    models = MODEL_ORDER

    matrix = np.full((len(models), n_cols), np.nan)
    gold_n = np.full((len(models), n_cols, 2), np.nan)

    for j, (strategy, _) in enumerate(col_specs):
        for i, model in enumerate(models):
            key = (model, strategy)
            if key not in model_strategy_groups:
                continue
            items = model_strategy_groups[key]
            n = len(items)
            gold = sum(1 for r in items if r["tier"] == "GOLD")
            matrix[i, j] = gold / n * 100
            gold_n[i, j, 0] = gold
            gold_n[i, j, 1] = n

    cmap = LinearSegmentedColormap.from_list(
        "amber_heat",
        ["#FAEEDA", "#FAC775", "#EF9F27", "#BA7517", "#854F0B"],
        N=256,
    )
    cmap.set_bad("#F1EFE8")

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(len(models)):
        for j in range(n_cols):
            strategy, _ = col_specs[j]
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10.5, color="#888780")
            else:
                g, ncell = int(gold_n[i, j, 0]), int(gold_n[i, j, 1])
                tc = "#412402"
                fw = "bold" if v >= 40 else "normal"
                ax.text(
                    j, i, f"{g}/{ncell}\n{v:.1f}%",
                    ha="center", va="center", fontsize=8, fontweight=fw, color=tc,
                    linespacing=1.25,
                )

    for xi in np.arange(-0.5, n_cols, 1):
        ax.axvline(xi, color="white", linewidth=2)
    for yi in np.arange(-0.5, len(models), 1):
        ax.axhline(yi, color="white", linewidth=2)

    xticklabels = [lab for _, lab in col_specs]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xticklabels, fontsize=8.5, linespacing=1.3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.tick_params(length=0)
    ax.set_title(
        "GOLD% — v7 agent vs v7 patient (by model)",
        fontsize=9.5, color="#333333", loc="left", pad=10,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label("GOLD %", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)


def plot_combined(strategy_groups, model_strategy_groups, output_path):
    """
    Single figure with two panels side by side:
      left  – stacked bar (tier distribution by strategy)
      right – GOLD% heatmap (model × strategy)
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 4.6))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[6, 5], wspace=0.30)

    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    _draw_strategy_bars(ax_bar,  strategy_groups)
    _draw_model_heatmap(ax_heat, model_strategy_groups)

    # Shared footnote
    fig.text(
        0.5, -0.04,
        "Intermediate results  ·  some strategy × model combinations not yet complete"
        "  ·  — = not yet run",
        ha="center", fontsize=7.5, color="#888780",
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved combined chart: {output_path}")


def plot_strategy_only(strategy_groups, output_path, title=None):
    fig, ax = plt.subplots(figsize=(15, 4.2))
    _draw_strategy_bars(ax, strategy_groups, title=title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved strategy chart: {output_path}")


def plot_heatmap_only(model_strategy_groups, output_path):
    fig, ax = plt.subplots(figsize=(12.5, 3.2))
    _draw_model_heatmap(ax, model_strategy_groups)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved heatmap: {output_path}")


def plot_v7_agent_vs_patient_heatmap(model_strategy_groups, output_path):
    fig, ax = plt.subplots(figsize=(16.5, 3.2))
    _draw_v7_agent_vs_patient_heatmap(ax, model_strategy_groups)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved v7 agent vs patient heatmap: {output_path}")


def _tier_lookup(filtered_rows):
    """(model, strategy, verb) -> tier string."""
    return {
        (r["model"], r["strategy"], r["verb"]): r["tier"]
        for r in filtered_rows
    }


def _eval_judge_tag(eval_run):
    if eval_run.startswith("evalv3.1_agent_"):
        return EVAL_JUDGE_GPT
    if eval_run.startswith("evalv31agent_qwen30b-think_"):
        return EVAL_JUDGE_QWEN30B
    if eval_run.startswith("evalv31agent_sonnet-4.6_"):
        return EVAL_JUDGE_SONNET
    return None


def _tier_lookup_by_judge(judge_rows):
    """(judge_tag, model, strategy, verb) -> tier string."""
    return {
        (r["judge"], r["model"], r["strategy"], r["verb"]): r["tier"]
        for r in judge_rows
    }


def _classify_agent_patient_gold(tier_agent, tier_patient):
    """Return discrete outcome code comparing GOLD under agent vs patient framing."""
    if tier_agent is None or tier_patient is None:
        return _FRAMING_INCOMPLETE
    ga = tier_agent == "GOLD"
    gp = tier_patient == "GOLD"
    if ga and gp:
        return _FRAMING_BOTH_GOLD
    if ga and not gp:
        return _FRAMING_AGENT_GOLD_ONLY
    if gp and not ga:
        return _FRAMING_PATIENT_GOLD_ONLY
    return _FRAMING_NEITHER_GOLD


def _build_framing_matrix(lookup, models, verbs, agent_strategy, patient_strategy):
    """Shape (len(models), len(verbs)) with integer outcome codes."""
    mat = np.empty((len(models), len(verbs)), dtype=np.int8)
    for i, model in enumerate(models):
        for j, verb in enumerate(verbs):
            ta = lookup.get((model, agent_strategy, verb))
            tp = lookup.get((model, patient_strategy, verb))
            mat[i, j] = _classify_agent_patient_gold(ta, tp)
    return mat


def _set_verb_xticklabels(ax, verbs):
    """Rotated verb names on the x-axis (same column order in every subplot)."""
    ax.set_xticklabels(verbs, rotation=68, ha="right", fontsize=6.9)
    ax.tick_params(axis="x", length=0, pad=1, labelbottom=True)


def _build_judge_matrix(lookup, models, verbs, strategy, judge_a, judge_b):
    """Shape (len(models), len(verbs)) comparing GOLD between two eval judges."""
    mat = np.empty((len(models), len(verbs)), dtype=np.int8)
    for i, model in enumerate(models):
        for j, verb in enumerate(verbs):
            ta = lookup.get((judge_a, model, strategy, verb))
            tb = lookup.get((judge_b, model, strategy, verb))
            mat[i, j] = _classify_agent_patient_gold(ta, tb)
    return mat


def _judge_model_strategy_groups(judge_rows):
    """(judge_tag, model, strategy) -> list of per-verb eval rows."""
    groups = defaultdict(list)
    for r in judge_rows:
        groups[(r["judge"], r["model"], r["strategy"])].append(r)
    return groups


def _draw_judge_heatmap(ax, judge_groups, col_specs, title):
    """GOLD% heatmap: rows = generator models, columns = paired judges per strategy."""
    n_cols = len(col_specs)
    models = MODEL_ORDER
    fontsize = 7.0 if n_cols > 8 else 7.5

    matrix = np.full((len(models), n_cols), np.nan)
    gold_n = np.full((len(models), n_cols, 2), np.nan)

    for j, (strategy, judge_tag, _) in enumerate(col_specs):
        for i, model in enumerate(models):
            key = (judge_tag, model, strategy)
            if key not in judge_groups:
                continue
            items = judge_groups[key]
            n = len(items)
            gold = sum(1 for r in items if r["tier"] == "GOLD")
            matrix[i, j] = gold / n * 100
            gold_n[i, j, 0] = gold
            gold_n[i, j, 1] = n

    cmap = LinearSegmentedColormap.from_list(
        "amber_heat",
        ["#FAEEDA", "#FAC775", "#EF9F27", "#BA7517", "#854F0B"],
        N=256,
    )
    cmap.set_bad("#F1EFE8")

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(len(models)):
        for j in range(n_cols):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10.5, color="#888780")
            else:
                g, ncell = int(gold_n[i, j, 0]), int(gold_n[i, j, 1])
                tc = "#412402"
                fw = "bold" if v >= 40 else "normal"
                ax.text(
                    j, i, f"{g}/{ncell}\n{v:.1f}%",
                    ha="center", va="center", fontsize=fontsize, fontweight=fw, color=tc,
                    linespacing=1.2,
                )

    for xi in np.arange(-0.5, n_cols, 1):
        ax.axvline(xi, color="white", linewidth=2)
    for yi in np.arange(-0.5, len(models), 1):
        ax.axhline(yi, color="white", linewidth=2)

    xticklabels = [lab for _, _, lab in col_specs]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xticklabels, fontsize=6.8 if n_cols > 8 else 7.5, linespacing=1.25)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=9.5, color="#333333", loc="left", pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label("GOLD %", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)


def plot_judge_model_heatmap(judge_rows, output_path, col_specs, title, footnote):
    """Summary heatmap: GOLD rates across generator models, strategies, and eval judges."""
    width = 8.5 + 2.2 * len(col_specs)
    fig, ax = plt.subplots(figsize=(width, 3.6))
    _draw_judge_heatmap(ax, _judge_model_strategy_groups(judge_rows), col_specs, title)
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=7.5, color="#888780")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved judge model heatmap: {output_path}")


def _judge_heatmap_footnote(judges):
    parts = []
    for j in judges:
        parts.append(f"{JUDGE_EVAL_PREFIXES[j]} ({JUDGE_SHORT_NAMES[j]} judge)")
    return (
        " vs ".join(parts) + " on v7_agent_* runs  ·  "
        "RB/R2V: GPT judge on all six models; Qwen judge on most models; Sonnet judge pending for singleshot  ·  "
        "R2R/R2VR: all three judges for gpt-5.1 & sonnet-4.6; GPT & Qwen also on Qwen/Ministral iterative runs  ·  "
        "— = no eval for that judge × model"
    )


def _latex_escape(text):
    """Escape characters that are special in LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in str(text):
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return center - margin, center + margin


def _judge_complete_pairs(judge_rows, judge_a, judge_b):
    """List of dicts with model, strategy, verb, and tier under each judge."""
    by_key = defaultdict(dict)
    for r in judge_rows:
        key = (r["model"], r["strategy"], r["verb"])
        by_key[key][r["judge"]] = r["tier"]
    pairs = []
    for (model, strategy, verb), tiers in sorted(by_key.items()):
        if judge_a not in tiers or judge_b not in tiers:
            continue
        pairs.append({
            "model": model,
            "strategy": strategy,
            "verb": verb,
            "a": tiers[judge_a],
            "b": tiers[judge_b],
            "judge_a": judge_a,
            "judge_b": judge_b,
        })
    return pairs


def _judge_agreement_stats(pairs):
    """Contingency and agreement metrics for complete judge pairs."""
    n = len(pairs)
    both = a_only = b_only = neither = 0
    for p in pairs:
        ga = p["a"] == "GOLD"
        gb = p["b"] == "GOLD"
        if ga and gb:
            both += 1
        elif ga:
            a_only += 1
        elif gb:
            b_only += 1
        else:
            neither += 1
    a_g = both + a_only
    b_g = both + b_only
    union_g = a_g + b_g - both

    p_a_given_b = both / b_g if b_g else float("nan")
    p_b_given_a = both / a_g if a_g else float("nan")
    jaccard = both / union_g if union_g else float("nan")
    po = (both + neither) / n if n else float("nan")
    p_a = a_g / n if n else float("nan")
    p_b = b_g / n if n else float("nan")
    pe = p_a * p_b + (1.0 - p_a) * (1.0 - p_b) if n else float("nan")
    kappa = (po - pe) / (1.0 - pe) if n and pe < 1.0 else float("nan")

    ci_lo, ci_hi = _wilson_ci(both, b_g) if b_g else (float("nan"), float("nan"))

    discordant = a_only + b_only
    if discordant:
        mcm_chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant
        mcm_z = (a_only - b_only) / math.sqrt(discordant)
    else:
        mcm_chi2 = mcm_z = float("nan")

    return {
        "n": n,
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "a_g": a_g,
        "b_g": b_g,
        "p_a_given_b": p_a_given_b,
        "p_b_given_a": p_b_given_a,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "jaccard": jaccard,
        "kappa": kappa,
        "po": po,
        "mcm_chi2": mcm_chi2,
        "mcm_z": mcm_z,
    }


def _fmt_pct(x, digits=1):
    if x is None or math.isnan(x):
        return "---"
    return f"{100.0 * x:.{digits}f}\\%"


def _fmt_prop(x, digits=3):
    if x is None or math.isnan(x):
        return "---"
    return f"{x:.{digits}f}"


def write_judge_agreement_tex(judge_rows, judge_a, judge_b, output_path, figure_relpath=None):
    """
    Write a standalone LaTeX report (Overleaf-ready) with metric definitions,
    reasoning, and computed agreement statistics for two eval judges.
    """
    name_a = JUDGE_DISPLAY_NAMES[judge_a]
    name_b = JUDGE_DISPLAY_NAMES[judge_b]
    prefix_a = JUDGE_EVAL_PREFIXES[judge_a].rstrip("_").replace("_", r"\_")
    prefix_b = JUDGE_EVAL_PREFIXES[judge_b].rstrip("_").replace("_", r"\_")
    short_a = JUDGE_SHORT_NAMES[judge_a]
    short_b = JUDGE_SHORT_NAMES[judge_b]

    if figure_relpath is None:
        figure_relpath = f"../results/figures/verb_judge_{short_a.lower()}_vs_{short_b.lower()}_gold.png"

    all_pairs = _judge_complete_pairs(judge_rows, judge_a, judge_b)
    if not all_pairs:
        print("WARNING: No complete judge pairs; skipping LaTeX report.")
        return

    overall = _judge_agreement_stats(all_pairs)

    strat_labels = {
        "v7_agent_60": "RB (v7\\_agent single-shot)",
        "v7_agent_verbalized_60": "R2V (v7\\_agent verbalized)",
        "v7_agent_iterative_60": "R2R (iterative)",
        "v7_agent_verbalized_iterative_60": "R2VR (verbalized iterative)",
    }
    by_strategy = {
        s: _judge_agreement_stats([p for p in all_pairs if p["strategy"] == s])
        for s in sorted({p["strategy"] for p in all_pairs})
    }
    compare_models = sorted({p["model"] for p in all_pairs})
    by_model = {
        m: _judge_agreement_stats([p for p in all_pairs if p["model"] == m])
        for m in compare_models
    }

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    csv_path = _latex_escape("results/aggregate/all_evaluations_merged_preview.csv")

    def stat_row(label, s):
        return (
            f"{_latex_escape(label)} & {s['n']} & {s['both']} & {s['a_only']} & {s['b_only']} "
            f"& {_fmt_pct(s['p_a_given_b'])} & {_fmt_pct(s['p_b_given_a'])} "
            f"& {_fmt_pct(s['jaccard'])} & {_fmt_prop(s['kappa'])} \\\\"
        )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{caption}",
        r"\graphicspath{{../results/figures/}}",
        "",
        rf"\title{{Evaluator Agreement: {name_a} vs.\ {name_b} Judge}}",
        r"\author{Auto-generated by \texttt{clean\_sampling\_w\_charts.py}}",
        f"\\date{{{generated}}}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section{Research question}",
        rf"We ask whether \textbf{{{name_b}}} as evaluator (\texttt{{{prefix_b}\_}}) "
        rf"is \emph{{conservative}} relative to the {name_a} judge (\texttt{{{prefix_a}\_}}): "
        rf"when {name_b} assigns \textsc{{Gold}}, does {name_a} usually agree?",
        rf"The complementary question---when {name_a} assigns \textsc{{Gold}}, does {name_b} agree?---"
        r"measures leniency asymmetry and is reported separately.",
        "",
        r"\section{Data unit and overlap}",
        r"\begin{itemize}",
        r"  \item \textbf{Unit of analysis:} one complete triple "
        r"(generator model, strategy, verb) where \emph{both} judges evaluated the same generation.",
        r"  \item \textbf{Generator models:} gpt-5.1, claude-sonnet-4.6 (sonnet-4.6).",
        r"  \item \textbf{Strategies (overlap only):} "
        r"\texttt{v7\_agent\_iterative\_60} (R2R) and \texttt{v7\_agent\_verbalized\_iterative\_60} (R2VR).",
        r"  \item \textbf{Source:} " + csv_path + r".",
        r"  \item \textbf{Complete pairs:} $N=" + str(overall["n"]) + r"$.",
        r"\end{itemize}",
        "",
        r"\section{Metric definitions}",
        "",
        rf"\subsection{{Primary metric (directional)}}",
        r"\begin{equation}",
        rf"  P(\text{{{name_a}}}=\textsc{{Gold}} \mid \text{{{name_b}}}=\textsc{{Gold}})",
        rf"  = \frac{{\#\{{\text{{both \textsc{{Gold}}}}\}}}}{{\#\{{\text{{{name_b}}}=\textsc{{Gold}}\}}}}",
        r"\end{equation}",
        rf"Each {name_b}-\textsc{{Gold}} cell is one Bernoulli trial ({name_a} also \textsc{{Gold}} or not). "
        r"We report a \textbf{Wilson score 95\% confidence interval} (recommended for proportions; "
        r"better than Wald when $p$ is far from 0.5 or $n$ is moderate):",
        r"\begin{equation}",
        r"  \frac{\hat{p} + z^2/(2n) \;\pm\; "
        r"z\sqrt{\hat{p}(1-\hat{p})/n + z^2/(4n^2)}}{1 + z^2/n}, \quad z=1.96.",
        r"\end{equation}",
        "",
        r"\subsection{Companion metric (reverse direction)}",
        r"\begin{equation}",
        rf"  P(\text{{{name_b}}}=\textsc{{Gold}} \mid \text{{{name_a}}}=\textsc{{Gold}})",
        rf"  = \frac{{\#\{{\text{{both \textsc{{Gold}}}}\}}}}{{\#\{{\text{{{name_a}}}=\textsc{{Gold}}\}}}}.",
        r"\end{equation}",
        "",
        r"\subsection{Secondary symmetric metrics}",
        r"\paragraph{Jaccard (intersection over union) on \textsc{Gold} sets.}",
        r"\begin{equation}",
        rf"  J = \frac{{|\text{{{name_a}}}_{{\textsc{{Gold}}}} \cap \text{{{name_b}}}_{{\textsc{{Gold}}}}|}}",
        rf"  {{|\text{{{name_a}}}_{{\textsc{{Gold}}}} \cup \text{{{name_b}}}_{{\textsc{{Gold}}}}|}}.",
        r"\end{equation}",
        r"Treats judges symmetrically; does not answer the directional hypothesis.",
        "",
        r"\paragraph{Cohen's $\kappa$ (chance-corrected agreement on \textsc{Gold} vs.\ not).}",
        r"\begin{equation}",
        r"  \kappa = \frac{P_o - P_e}{1 - P_e},",
        r"\end{equation}",
        r"where $P_o$ is observed agreement on the binary \textsc{Gold}/not label. "
        rf"Low $\kappa$ can coexist with high $P(\text{{{name_a}}}\mid\text{{{name_b}}})$ when judges use "
        r"different base rates (prevalence). \textbf{Do not use $\kappa$ as the headline metric} here.",
        "",
        r"\subsection{McNemar test (asymmetry of disagreements)}",
        rf"For discordant pairs, let $b=$ {name_a}-\textsc{{Gold}} only and $c=$ {name_b}-\textsc{{Gold}} only. "
        r"With continuity correction:",
        r"\begin{equation}",
        r"  \chi^2_{\text{McNemar}} = \frac{(|b-c|-1)^2}{b+c}.",
        r"\end{equation}",
        r"Large $\chi^2$ indicates systematic directional disagreement (one judge more lenient).",
        "",
        r"\section{Results}",
        "",
        r"\subsection{Contingency table (all complete pairs)}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Cross-classification of \textsc{Gold} labels ($N=" + str(overall["n"]) + r"$).}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        rf" & \textbf{{{name_b} not \textsc{{Gold}}}} & \textbf{{{name_b} \textsc{{Gold}}}} \\",
        r"\midrule",
        f"{name_a} not \\textsc{{Gold}} & {overall['neither'] + overall['b_only']} "
        f"& {overall['b_only']} \\\\",
        f"{name_a} \\textsc{{Gold}} & {overall['a_only']} & {overall['both']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\noindent Counts: both \textsc{Gold} $=" + str(overall["both"]) + r"$; "
        rf"{name_a}-only $=" + str(overall["a_only"]) + r"$; "
        rf"{name_b}-only $=" + str(overall["b_only"]) + r"$; "
        r"neither $=" + str(overall["neither"]) + r"$. "
        rf"{name_a} assigns \textsc{{Gold}} more often ($n_{{\text{{{name_a}}}}}=" + str(overall["a_g"]) + r"$ vs.\ "
        rf"$n_{{\text{{{name_b}}}}}=" + str(overall["b_g"]) + r"$).",
        "",
        r"\subsection{Primary result}",
        r"\begin{equation}",
        rf"  P(\text{{{name_a}}}=\textsc{{Gold}} \mid \text{{{name_b}}}=\textsc{{Gold}}) "
        f"= \\frac{{{overall['both']}}}{{{overall['b_g']}}} "
        f"= {_fmt_prop(overall['p_a_given_b'])} "
        f"\\;\\left[{_fmt_prop(overall['ci_lo'])},\\;{_fmt_prop(overall['ci_hi'])}\\right]"
        r" \quad \text{(Wilson 95\% CI)}.",
        r"\end{equation}",
        rf"\noindent In words: when {name_b} assigns \textsc{{Gold}}, {name_a} agrees in "
        + _fmt_pct(overall["p_a_given_b"]) + r" of cases "
        r"(95\% CI: " + _fmt_pct(overall["ci_lo"]) + r"--" + _fmt_pct(overall["ci_hi"]) + r").",
        "",
        r"\subsection{Companion and secondary results}",
        r"\begin{itemize}",
        rf"  \item $P(\text{{{name_b}}}=\textsc{{Gold}} \mid \text{{{name_a}}}=\textsc{{Gold}}) = "
        f"{overall['both']}/{overall['a_g']} = {_fmt_prop(overall['p_b_given_a'])}$ "
        r"(" + _fmt_pct(overall["p_b_given_a"]) + r").",
        r"  \item Jaccard $J = " + _fmt_prop(overall["jaccard"]) + r"$ (" + _fmt_pct(overall["jaccard"]) + r").",
        r"  \item Cohen's $\kappa = " + _fmt_prop(overall["kappa"]) + r"$ "
        r"(overall binary agreement $P_o=" + _fmt_prop(overall["po"]) + r"$).",
        r"  \item McNemar: $\chi^2_{\text{cc}} = " + _fmt_prop(overall["mcm_chi2"], 2) + r"$, "
        r"$Z = " + _fmt_prop(overall["mcm_z"], 2) + r"$ "
        r"(discordant $b=" + str(overall["a_only"]) + r"$, $c=" + str(overall["b_only"]) + r"$).",
        r"\end{itemize}",
        "",
        r"\subsection{Breakdown by strategy and model}",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Agreement metrics by subset. "
        rf"$P(\text{{{name_a}}}\mid\text{{{name_b}}})$ = primary directional metric.}}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        rf"Subset & $N$ & Both & {name_a}-only & {name_b}-only & "
        rf"$P(\text{{{name_a}}}\mid\text{{{name_b}}})$ & $P(\text{{{name_b}}}\mid\text{{{name_a}}})$ & Jaccard & $\kappa$ \\",
        r"\midrule",
        stat_row("All", overall),
        r"\midrule",
    ]
    for strat, s in sorted(by_strategy.items()):
        lines.append(stat_row(strat_labels.get(strat, strat), s))
    lines.append(r"\midrule")
    for model, s in by_model.items():
        lines.append(stat_row(model, s))
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\section{Interpretation for the paper}",
        r"\begin{itemize}",
        rf"  \item \textbf{{Supported:}} {name_b}-\textsc{{Gold}} largely overlaps {name_a}-\textsc{{Gold}} "
        f"({_fmt_pct(overall['p_a_given_b'])} concordance; Wilson CI reported above).",
        r"  \item \textbf{Not supported if overstated:} The judges are not interchangeable---"
        rf"$\kappa$ is low and $P(\text{{{name_b}}}\mid\text{{{name_a}}}) = {_fmt_prop(overall['p_b_given_a'])}$.",
        rf"  \item \textbf{{Recommended claim:}} ``When the {name_b} judge assigns \textsc{{Gold}}, "
        rf"the {name_a} judge assigns \textsc{{Gold}} in [CI] of comparable cases.''",
        r"  \item \textbf{Figure:} Per-verb heatmap (Figure~\ref{fig:judge-verb-heatmap}) "
        r"shows spatial structure of agreement/disagreement.",
        r"\end{itemize}",
        "",
        r"\begin{figure}[ht]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{"
        + os.path.basename(figure_relpath)
        + r"}",
        rf"\caption{{Per-verb $\times$ model \textsc{{Gold}} overlap between judges (R2R and R2VR). "
        rf"Green = both \textsc{{Gold}}; blue = {name_a}-only; red = {name_b}-only; gray = neither.}}",
        r"\label{fig:judge-verb-heatmap}",
        r"\end{figure}",
        "",
        r"\section{Metrics to avoid as primary evidence}",
        r"\begin{itemize}",
        rf"  \item \textbf{{Raw intersection count}} without conditioning on {name_b}-\textsc{{Gold}}.",
        r"  \item \textbf{Accuracy} on pooled binary labels (misleading when prevalence differs).",
        r"  \item \textbf{Cohen's $\kappa$ alone} for the directional hypothesis.",
        r"\end{itemize}",
        "",
        r"\end{document}",
        "",
    ])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved judge agreement LaTeX report: {output_path}")


def plot_verb_framing_agent_vs_patient(filtered_rows, output_path):
    """
    Per-verb × model view of GOLD overlap between agent and patient framing.

    RB and R2V only; excludes gpt-5.1. Rows = models, columns = verbs (sorted).
    Colors: both GOLD, agent-only GOLD, patient-only GOLD, both evaluated but
    neither GOLD, missing eval on one side.
    """
    from matplotlib.gridspec import GridSpec

    pairs = VERB_FRAMING_AGENT_VS_PATIENT_PAIRS
    models = VERB_FRAMING_AGENT_VS_PATIENT_MODELS

    lookup = _tier_lookup(filtered_rows)
    strat_needed = set()
    for _, ag, pt in pairs:
        strat_needed.add(ag)
        strat_needed.add(pt)
    verbs = sorted(
        {r["verb"] for r in filtered_rows if r["strategy"] in strat_needed}
    )
    if not verbs:
        print("WARNING: No verbs for framing overlap chart; skipping.")
        return

    cmap = ListedColormap(list(_FRAMING_COLORS))

    n_panels = len(pairs)
    fig = plt.figure(figsize=(22, 9.2))
    gs = GridSpec(
        n_panels + 1, 1, figure=fig,
        height_ratios=[1] * n_panels + [0.48],
        hspace=0.82,
    )
    legend_labels = (
        "Both GOLD",
        "Agent GOLD only",
        "Patient GOLD only",
        "Neither GOLD (evaluated)",
        "Incomplete (missing side)",
    )

    n_pairs = len(models) * len(verbs)
    for row, (title, ag_s, pt_s) in enumerate(pairs):
        ax = fig.add_subplot(gs[row, 0])
        mat = _build_framing_matrix(lookup, models, verbs, ag_s, pt_s)
        ax.imshow(
            mat,
            cmap=cmap,
            norm=NoNorm(),
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xticks(range(len(verbs)))
        ax.set_xlim(-0.5, len(verbs) - 0.5)
        ax.set_ylim(len(models) - 0.5, -0.5)
        _set_verb_xticklabels(ax, verbs)
        ax.tick_params(axis="y", length=0)

        complete = mat[mat != _FRAMING_INCOMPLETE]
        n_complete = complete.size
        n_inc = int((mat == _FRAMING_INCOMPLETE).sum())
        if n_complete:
            c = Counter(int(x) for x in complete.flat)
            pct = lambda code: 100.0 * c.get(code, 0) / n_complete
            # Each % is share of *complete* pairs only; four outcomes partition them → sum 100%.
            sub = (
                f"{n_complete} complete model×verb pairs "
                f"(both agent & patient eval present); "
                f"split → both GOLD {pct(_FRAMING_BOTH_GOLD):.0f}%, "
                f"agent-only GOLD {pct(_FRAMING_AGENT_GOLD_ONLY):.0f}%, "
                f"patient-only GOLD {pct(_FRAMING_PATIENT_GOLD_ONLY):.0f}%, "
                f"neither GOLD {pct(_FRAMING_NEITHER_GOLD):.0f}% "
                f"(four shares sum to 100%). "
                f"Incomplete={n_inc} / {n_pairs} cells."
            )
        else:
            sub = (
                "No complete agent+patient pairs in merged CSV for this modality. "
                f"Incomplete={n_inc} / {n_pairs} cells."
            )
        ax.set_title(
            f"{title}\n{sub}",
            fontsize=8.8,
            loc="left",
            pad=10,
            color="#333333",
            linespacing=1.38,
        )

        # Thin white grid on cell edges
        for xi in np.arange(-0.5, len(verbs), 1):
            ax.axvline(xi, color="white", linewidth=0.35, zorder=3)
        for yi in np.arange(-0.5, len(models), 1):
            ax.axhline(yi, color="white", linewidth=0.35, zorder=3)

    # Net model-level preference per verb: (# agent-only GOLD) − (# patient-only GOLD)
    ax_net = fig.add_subplot(gs[n_panels, 0])
    n_models = len(models)
    net_mat = np.zeros((len(pairs), len(verbs)), dtype=np.float32)
    for ridx, (_, ag_s, pt_s) in enumerate(pairs):
        mat = _build_framing_matrix(lookup, models, verbs, ag_s, pt_s)
        for j in range(len(verbs)):
            col = mat[:, j]
            net_mat[ridx, j] = float((col == _FRAMING_AGENT_GOLD_ONLY).sum()
                                     - (col == _FRAMING_PATIENT_GOLD_ONLY).sum())
    vmax = max(1, float(np.nanmax(np.abs(net_mat))))
    im_net = ax_net.imshow(
        net_mat,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_net.set_yticks(range(len(pairs)))
    ax_net.set_yticklabels(["RB", "R2V"], fontsize=9)
    ax_net.set_xticks(range(len(verbs)))
    _set_verb_xticklabels(ax_net, verbs)
    ax_net.tick_params(axis="y", length=0)
    ax_net.set_xlim(-0.5, len(verbs) - 0.5)
    ax_net.set_title(
        "Net preference across models (agent-only GOLD minus patient-only GOLD per verb)\n"
        "Blue → more models succeeded with agent framing only; red → patient framing only",
        fontsize=9.5,
        color="#333333",
        loc="left",
        pad=10,
    )
    cbar_net = plt.colorbar(im_net, ax=ax_net, shrink=0.92, pad=0.01, aspect=18)
    cbar_net.set_label(f"Δ (# of {n_models} models)", fontsize=8)
    cbar_net.ax.tick_params(labelsize=7.5)
    cbar_net.outline.set_visible(False)

    handles = [
        mpatches.Patch(facecolor=_FRAMING_COLORS[i], edgecolor="white", linewidth=0.6, label=legend_labels[i])
        for i in range(len(legend_labels))
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        fontsize=9,
        frameon=False,
        handlelength=1.35,
        handleheight=1.0,
        columnspacing=1.4,
    )

    fig.suptitle(
        "Agent vs patient framing — GOLD overlap by verb and model (RB & R2V only)\n"
        "(each column is one verb; each row is one generator model; gpt-5.1 excluded)",
        fontsize=11.5,
        fontweight="bold",
        color="#222222",
        y=0.99,
    )
    fig.text(
        0.5,
        0.022,
        "Success = tier GOLD. RB & R2V single-shot modalities only; gpt-5.1 excluded. "
        "Percentages in each panel are shares of complete pairs only "
        "(model×verb cells where both agent and patient evals exist); "
        "both GOLD / agent-only / patient-only / neither GOLD partition those pairs and add to 100%. "
        "Each column is one verb (same left-to-right order in every panel).",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )

    plt.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.35, facecolor="white")
    plt.close(fig)
    print(f"Saved verb framing overlap chart: {output_path}")


def plot_verb_judge_comparison(judge_rows, judge_a, judge_b, output_path):
    """
    Per-verb × model GOLD overlap (R2R / R2VR only) between two eval judges
    on the same v7_agent_* generation runs.
    """
    from matplotlib.gridspec import GridSpec

    name_a = JUDGE_DISPLAY_NAMES[judge_a]
    name_b = JUDGE_DISPLAY_NAMES[judge_b]
    prefix_a = JUDGE_EVAL_PREFIXES[judge_a]
    prefix_b = JUDGE_EVAL_PREFIXES[judge_b]
    models = JUDGE_VERB_COMPARE_MODELS
    cmap = ListedColormap(list(_FRAMING_COLORS))
    pairs = JUDGE_VERB_MODALITY_PAIRS
    n_mod = len(pairs)
    judge_rows = [
        r for r in judge_rows
        if r["strategy"] in JUDGE_VERB_COMPARE_STRATEGIES
        and r["model"] in JUDGE_VERB_COMPARE_MODELS
    ]
    lookup = _tier_lookup_by_judge(judge_rows)
    verbs = sorted({r["verb"] for r in judge_rows})
    if not verbs:
        print(f"WARNING: No verbs for {name_a} vs {name_b} judge comparison; skipping.")
        return

    fig = plt.figure(figsize=(22, 9.8))
    gs = GridSpec(n_mod + 1, 1, figure=fig,
                  height_ratios=[1] * n_mod + [0.42], hspace=0.82)
    legend_labels = (
        "Both GOLD",
        f"{name_a} judge GOLD only ({prefix_a})",
        f"{name_b} judge GOLD only ({prefix_b})",
        "Neither GOLD (evaluated)",
        "Incomplete (missing judge)",
    )

    n_pairs = len(models) * len(verbs)
    for row, (title, strategy) in enumerate(pairs):
        ax = fig.add_subplot(gs[row, 0])
        mat = _build_judge_matrix(
            lookup, models, verbs, strategy, judge_a, judge_b,
        )
        ax.imshow(mat, cmap=cmap, norm=NoNorm(), aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xticks(range(len(verbs)))
        ax.set_xlim(-0.5, len(verbs) - 0.5)
        ax.set_ylim(len(models) - 0.5, -0.5)
        _set_verb_xticklabels(ax, verbs)
        ax.tick_params(axis="y", length=0)

        complete = mat[mat != _FRAMING_INCOMPLETE]
        n_complete = complete.size
        n_inc = int((mat == _FRAMING_INCOMPLETE).sum())
        if n_complete:
            c = Counter(int(x) for x in complete.flat)
            pct = lambda code: 100.0 * c.get(code, 0) / n_complete
            sub = (
                f"{n_complete} complete model×verb pairs "
                f"(both judges present); "
                f"split → both GOLD {pct(_FRAMING_BOTH_GOLD):.0f}%, "
                f"{name_a}-only GOLD {pct(_FRAMING_AGENT_GOLD_ONLY):.0f}%, "
                f"{name_b}-only GOLD {pct(_FRAMING_PATIENT_GOLD_ONLY):.0f}%, "
                f"neither GOLD {pct(_FRAMING_NEITHER_GOLD):.0f}% "
                f"(four shares sum to 100%). "
                f"Incomplete={n_inc} / {n_pairs} cells."
            )
        else:
            sub = (
                "No complete judge pairs in merged CSV for this modality. "
                f"Incomplete={n_inc} / {n_pairs} cells."
            )
        ax.set_title(f"{title}\n{sub}", fontsize=8.8, loc="left", pad=10,
                     color="#333333", linespacing=1.38)

        for xi in np.arange(-0.5, len(verbs), 1):
            ax.axvline(xi, color="white", linewidth=0.35, zorder=3)
        for yi in np.arange(-0.5, len(models), 1):
            ax.axhline(yi, color="white", linewidth=0.35, zorder=3)

    ax_net = fig.add_subplot(gs[n_mod, 0])
    n_models = len(models)
    net_mat = np.zeros((len(pairs), len(verbs)), dtype=np.float32)
    for ridx, (_, strategy) in enumerate(pairs):
        mat = _build_judge_matrix(
            lookup, models, verbs, strategy, judge_a, judge_b,
        )
        for j in range(len(verbs)):
            col = mat[:, j]
            net_mat[ridx, j] = float((col == _FRAMING_AGENT_GOLD_ONLY).sum()
                                     - (col == _FRAMING_PATIENT_GOLD_ONLY).sum())
    vmax = max(1, float(np.nanmax(np.abs(net_mat))))
    im_net = ax_net.imshow(
        net_mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax_net.set_yticks(range(len(pairs)))
    ax_net.set_yticklabels(["R2R", "R2VR"], fontsize=9)
    ax_net.set_xticks(range(len(verbs)))
    _set_verb_xticklabels(ax_net, verbs)
    ax_net.tick_params(axis="y", length=0)
    ax_net.set_xlim(-0.5, len(verbs) - 0.5)
    ax_net.set_title(
        f"Net preference across models ({name_a}-judge-only GOLD minus {name_b}-judge-only GOLD per verb)\n"
        f"Blue → more models GOLD under {name_a} judge only; red → {name_b} judge only",
        fontsize=9.5, color="#333333", loc="left", pad=10,
    )
    cbar_net = plt.colorbar(im_net, ax=ax_net, shrink=0.92, pad=0.01, aspect=18)
    cbar_net.set_label(f"Δ (# of {n_models} models)", fontsize=8)
    cbar_net.ax.tick_params(labelsize=7.5)
    cbar_net.outline.set_visible(False)

    handles = [
        mpatches.Patch(facecolor=_FRAMING_COLORS[i], edgecolor="white", linewidth=0.6,
                       label=legend_labels[i])
        for i in range(len(legend_labels))
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
        ncol=5, fontsize=9, frameon=False, handlelength=1.35,
        handleheight=1.0, columnspacing=1.2,
    )

    fig.suptitle(
        f"{prefix_a} ({name_a} judge) vs {prefix_b} ({name_b} judge) — GOLD overlap by verb\n"
        "(each column is one verb; each row is one generator model)",
        fontsize=11.5, fontweight="bold", color="#222222", y=0.99,
    )
    fig.text(
        0.5, 0.022,
        "Iterative runs only (R2R / R2VR). Models: gpt-5.1 and sonnet-4.6 only. "
        f"Same v7_agent_* generations; evaluator differs ({prefix_a} vs {prefix_b}). "
        "Both judges evaluated all cells shown. "
        "Percentages in subtitles are shares of complete pairs (sum to 100%). "
        "Each column is one verb (same left-to-right order in every panel).",
        ha="center", fontsize=8.5, color="#555555",
    )

    plt.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.095)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.35, facecolor="white")
    plt.close(fig)
    print(f"Saved judge comparison verb heatmap: {output_path}")


# ─────────────────────────────── failure-cause analysis ─────────────────────────

FAILURE_CAUSE_ORDER = [
    "GOLD (success)",
    "Iterative: no pass (scene judged OK)",
    "SILVER (scene OK, not GOLD)",
    "Agent gradient wrong",
    "Scene needs redesign",
    "Visual identifiability",
    "Gradient incorrect",
]
FAILURE_CAUSE_COLORS = {
    "GOLD (success)":                      "#BA7517",
    "Iterative: no pass (scene judged OK)":  "#888780",
    "SILVER (scene OK, not GOLD)":           "#C4BFB5",
    "Agent gradient wrong":                  "#0077BB",
    "Scene needs redesign":                  "#CC3311",
    "Visual identifiability":                "#7A5195",
    "Gradient incorrect":                    "#E67E22",
}
FAILURE_CAUSE_SHORT_LABELS = {
    "GOLD (success)": "GOLD (success)",
    "Iterative: no pass (scene judged OK)": "Iterative: no pass (scene OK)",
    "SILVER (scene OK, not GOLD)": "SILVER (scene OK)",
    "Agent gradient wrong": "Agent gradient wrong",
    "Scene needs redesign": "Scene needs redesign",
    "Visual identifiability": "Visual identifiability",
    "Gradient incorrect": "Gradient incorrect",
}
AGENT_SLOT_LABELS = [
    ("plausible_easy",   "PE"),
    ("plausible_hard",   "PH"),
    ("implausible_easy", "IE"),
    ("implausible_hard", "IH"),
]

def agent_gpt_judge_modalities(model_eval_slug: str):
    """RB/R2V/R2R/R2VR evalv3.1_agent_* runs for one generator (GPT judge)."""
    p = f"evalv3.1_agent_{model_eval_slug}"
    return [
        ("RB",   f"{p}_v7_agent_60"),
        ("R2V",  f"{p}_v7_agent_verbalized_60"),
        ("R2R",  f"{p}_v7_agent_iterative_60"),
        ("R2VR", f"{p}_v7_agent_verbalized_iterative_60"),
    ]


# (display label, eval_run slug, short name for output files)
AGENT_GPT_JUDGE_FAILURE_MODELS = [
    ("gpt-5.1",    "gpt-5.1-2025-11-13",    "gpt51"),
    ("sonnet-4.6", "claude-sonnet-4-6",     "sonnet46"),
]

GPT51_AGENT_GPT_JUDGE_MODALITIES = agent_gpt_judge_modalities("gpt-5.1-2025-11-13")
SONNET_AGENT_GPT_JUDGE_MODALITIES = agent_gpt_judge_modalities("claude-sonnet-4-6")
GPT51_IR_AGENT_GPT_JUDGE_EVAL_RUN = GPT51_AGENT_GPT_JUDGE_MODALITIES[2][1]
SONNET_IR_AGENT_GPT_JUDGE_EVAL_RUN = SONNET_AGENT_GPT_JUDGE_MODALITIES[2][1]


def classify_failure_cause(row):
    """
    Map one eval row to a coarse failure-cause label (why not GOLD, or GOLD).
    Hierarchy matches eval semantics: scene redesign > iterative pass > agent > visual.
    """
    tier = row.get("tier", "")
    if tier == "GOLD":
        return "GOLD (success)"

    scene_verdict = row.get("scene_verdict", "")
    if scene_verdict == "NEEDS_REDESIGN":
        return "Scene needs redesign"

    summary = row.get("summary", "")
    is_iterative = summary.strip().startswith("Iterative:")
    if is_iterative and scene_verdict == "GOOD" and (
        tier == "SILVER" or "pass=False" in summary
    ):
        return "Iterative: no pass (scene judged OK)"
    if tier == "SILVER" and scene_verdict == "GOOD":
        return "SILVER (scene OK, not GOLD)"

    if scene_verdict == "AGENT_FIXABLE":
        return "Agent gradient wrong"

    if row.get("visual_verdict") == "PROBLEMATIC":
        return "Visual identifiability"

    if str(row.get("gradient_correct", "")).lower() == "false":
        return "Gradient incorrect"

    return f"Other ({tier})"


def _agent_slot_errors(row):
    """Return list of slot labels (PE/PH/IE/IH) with non-CORRECT agent verdicts."""
    bad = []
    for cond, label in AGENT_SLOT_LABELS:
        v = row.get(f"verdict_{cond}", "")
        if v and v != "CORRECT":
            bad.append((label, v))
    return bad


def _failure_cause_counts(eval_rows):
    """Return (n_total, cause_counts, plot_causes list)."""
    cause_counts = Counter(classify_failure_cause(r) for r in eval_rows)
    plot_causes = [c for c in FAILURE_CAUSE_ORDER if cause_counts.get(c, 0)]
    extra = [c for c in cause_counts if c not in FAILURE_CAUSE_ORDER]
    plot_causes.extend(sorted(extra))
    return len(eval_rows), cause_counts, plot_causes


def _draw_failure_cause_bars(ax, eval_rows, panel_title):
    """Left panel: horizontal bars of primary failure causes."""
    n_total, cause_counts, plot_causes = _failure_cause_counts(eval_rows)
    if n_total == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return cause_counts

    plot_causes_bar = list(reversed(plot_causes))
    y = np.arange(len(plot_causes_bar))
    counts = [cause_counts[c] for c in plot_causes_bar]
    colors = [FAILURE_CAUSE_COLORS.get(c, "#CCCCCC") for c in plot_causes_bar]
    ax.barh(y, counts, color=colors, edgecolor="white", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [FAILURE_CAUSE_SHORT_LABELS.get(c, c) for c in plot_causes_bar],
        fontsize=8,
    )
    ax.set_xlabel("Number of verbs", fontsize=8, color="#555553")
    ax.set_xlim(0, max(max(counts) * 1.2 + 1, 3))
    for i, (cnt, _) in enumerate(zip(counts, plot_causes_bar)):
        pct = cnt / n_total * 100
        ax.text(cnt + 0.35, i, f"{cnt} ({pct:.0f}%)", va="center", ha="left", fontsize=8)
    ax.set_title(panel_title, fontsize=9.5, loc="left", color="#333333", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#E8E6DF", linewidth=0.5)
    ax.set_axisbelow(True)
    return cause_counts


def _draw_agent_slot_bars(ax, eval_rows, n_agent_verbs):
    """Right panel: which plausibility slots are wrong for agent-gradient failures."""
    slot_counter = Counter()
    for r in eval_rows:
        if classify_failure_cause(r) != "Agent gradient wrong":
            continue
        for label, _ in _agent_slot_errors(r):
            slot_counter[label] += 1

    if not slot_counter:
        ax.text(0.5, 0.5, "No agent-gradient\nfailures", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888780")
        ax.set_axis_off()
        return

    slots = [lab for _, lab in AGENT_SLOT_LABELS]
    counts = [slot_counter[lab] for lab in slots]
    y = np.arange(len(slots))
    ax.barh(y, counts, color="#0077BB", edgecolor="white", height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(slots, fontsize=8)
    ax.set_xlabel("Incorrect / borderline slots", fontsize=8)
    ax.set_xlim(0, max(max(counts) * 1.25 + 0.5, 2))
    for i, c in enumerate(counts):
        if c:
            ax.text(c + 0.08, i, str(c), va="center", fontsize=8)
    ax.set_title(
        f"Agent slot errors ({n_agent_verbs} verbs)",
        fontsize=9.5, loc="left", color="#333333", pad=6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _print_failure_cause_breakdown(eval_rows, modality_label=None):
    """Console summary for one eval slice."""
    if not eval_rows:
        return
    _, cause_counts, plot_causes = _failure_cause_counts(eval_rows)
    n_total = len(eval_rows)
    header = modality_label or eval_rows[0]["eval_run"]
    print(f"\n--- Failure causes: {header} ---")
    for c in plot_causes:
        cnt = cause_counts[c]
        print(f"  {c:<40} {cnt:>3}  ({cnt/n_total*100:5.1f}%)")


def plot_failure_cause_distribution(
    eval_rows, output_path, title=None, modality_label=None, model_label="gpt-5.1",
):
    """Single modality: horizontal cause bars + agent-slot breakdown."""
    from matplotlib.gridspec import GridSpec

    if not eval_rows:
        print("WARNING: No rows for failure-cause chart; skipping.")
        return

    label = modality_label or "Outcome"
    fig = plt.figure(figsize=(13.5, 4.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1], wspace=0.42)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    cause_counts = _draw_failure_cause_bars(
        ax_a, eval_rows, f"{label} — primary cause (n={len(eval_rows)})",
    )
    _draw_agent_slot_bars(ax_b, eval_rows, cause_counts.get("Agent gradient wrong", 0))

    if title is None:
        title = (
            f"Why not GOLD? — {model_label} {label}, judged by GPT (evalv3.1_agent_)"
        )
    fig.suptitle(title, fontsize=11, fontweight="bold", color="#222222", y=1.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved failure-cause chart: {output_path}")
    _print_failure_cause_breakdown(eval_rows, modality_label)


def plot_failure_causes_model_all_modalities(
    all_rows, output_path, model_label, modalities,
):
    """RB, R2V, R2R, R2VR — v7_agent_* for one generator, judged by GPT (evalv3.1_agent_)."""
    from matplotlib.gridspec import GridSpec

    by_run = {run: [] for _, run in modalities}
    for r in all_rows:
        if r["eval_run"] in by_run:
            by_run[r["eval_run"]].append(r)

    n_mod = len(modalities)
    fig = plt.figure(figsize=(13.5, 2.35 * n_mod))
    gs = GridSpec(n_mod, 2, figure=fig, width_ratios=[1.15, 1], hspace=0.55, wspace=0.38)

    for row, (label, run) in enumerate(modalities):
        eval_rows = by_run.get(run, [])
        ax_a = fig.add_subplot(gs[row, 0])
        ax_b = fig.add_subplot(gs[row, 1])
        cause_counts = _draw_failure_cause_bars(
            ax_a, eval_rows, f"{label} — primary cause (n={len(eval_rows)})",
        )
        _draw_agent_slot_bars(ax_b, eval_rows, cause_counts.get("Agent gradient wrong", 0))
        _print_failure_cause_breakdown(eval_rows, label)

    fig.suptitle(
        f"Why not GOLD? — {model_label} v7_agent strategies, judged by GPT (evalv3.1_agent_)\n"
        "RB / R2V = single-shot · R2R / R2VR = iterative",
        fontsize=11, fontweight="bold", color="#222222", y=0.995,
    )
    fig.text(
        0.5, 0.01,
        "Left: primary cause per verb. Right: which agent slot (PE/PH/IE/IH) is wrong when "
        "scene_verdict=AGENT_FIXABLE. Data: all_evaluations_merged_preview.csv",
        ha="center", fontsize=7.5, color="#888780",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    print(f"Saved failure-cause chart (all modalities): {output_path}")


def plot_failure_causes_gpt51_all_modalities(all_rows, output_path):
    """Backward-compatible wrapper for gpt-5.1 all-modality chart."""
    plot_failure_causes_model_all_modalities(
        all_rows, output_path, "gpt-5.1", GPT51_AGENT_GPT_JUDGE_MODALITIES,
    )


# ─────────────────────────────── v7_agent paper examples table ─────────────────

# v7_agent framing only; wRes+ST (single-shot) and wRe+ST (iterative) modalities.
V7_AGENT_TABLE_STRATEGIES = (
    "v7_agent_60",
    "v7_agent_verbalized_60",
    "v7_agent_iterative_60",
    "v7_agent_verbalized_iterative_60",
)
V7_AGENT_TABLE_STRATEGY_ORDER = list(V7_AGENT_TABLE_STRATEGIES)
V7_AGENT_TABLE_STRATEGY_LABELS = {
    "v7_agent_60":                      "RB (wRes+ST)",
    "v7_agent_verbalized_60":           "R2V (wRes+ST)",
    "v7_agent_iterative_60":            "R2R (wRe+ST)",
    "v7_agent_verbalized_iterative_60": "R2VR (wRe+ST)",
}
V7_AGENT_TABLE_EVAL_PREFIX = EVAL_RUN_PREFIX_GPT

# Qualitative table (US letter / ACL table*, ~8pt, four sentence columns):
# Full-page qualitative table (~8pt, four sentence columns): ~24 rows.
PAPER_TABLE_TARGET_ROWS = 24
PAPER_TABLE_TIER_QUOTA = {"GOLD": 6, "SILVER": 6, "BRONZE": 4, "FAIL": 8}
PAPER_TABLE_CONTRAST_ROWS = {"method": 12, "model": 8, "verb": 4}
PAPER_TABLE_SAMPLE_SEED = 42

PAPER_EXAMPLES_CSV = "results/aggregate/v7_agent_paper_examples_table.csv"
PAPER_EXAMPLES_MD = "results/aggregate/v7_agent_paper_examples_metadata.md"


def _pool_key(r):
    return (r["model"], r["strategy"], r["verb"])


def _build_v7_agent_table_pool(rows):
    """GPT-judge eval rows for v7_agent wRes+ST / wRe+ST strategies, with generations joined."""
    seen, filtered = set(), []
    for r in rows:
        eval_run = r["eval_run"]
        if not eval_run.startswith(V7_AGENT_TABLE_EVAL_PREFIX):
            continue
        model, strategy = parse_source_run(eval_run)
        if not model or strategy not in V7_AGENT_TABLE_STRATEGIES:
            continue
        key = (model, strategy, r["verb"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append({
            "eval_run":     eval_run,
            "model":        model,
            "strategy":     strategy,
            "strategy_label": V7_AGENT_TABLE_STRATEGY_LABELS[strategy],
            "verb":         r["verb"],
            "tier":         r["tier"],
            "score_strict": r.get("score_strict", ""),
            "visual_verdict": r.get("visual_verdict", ""),
            "scene_verdict":  r.get("scene_verdict", ""),
            "summary":      r.get("summary", ""),
        })

    output_rows = []
    for r in filtered:
        gen = load_generation(r["model"], r["strategy"], r["verb"])
        if not gen:
            continue
        scene = gen.get("scene", {})
        sentences = gen.get("sentences", {})
        agents = gen.get("agents", {})
        output_rows.append({
            **r,
            "verb_form":  scene.get("verb_form", ""),
            "patient":    scene.get("patient", ""),
            "instrument": scene.get("instrument", ""),
            "location":   scene.get("location", ""),
            "agent_PE":   agents.get("plausible_easy", {}).get("agent", ""),
            "agent_PH":   agents.get("plausible_hard", {}).get("agent", ""),
            "agent_IH":   agents.get("implausible_hard", {}).get("agent", ""),
            "agent_IE":   agents.get("implausible_easy", {}).get("agent", ""),
            "sentence_PE": sentences.get("plausible_easy", ""),
            "sentence_PH": sentences.get("plausible_hard", ""),
            "sentence_IH": sentences.get("implausible_hard", ""),
            "sentence_IE": sentences.get("implausible_easy", ""),
        })
    return output_rows


def _index_pool(pool):
    by_key = {_pool_key(r): r for r in pool}
    by_tier = defaultdict(list)
    for r in pool:
        by_tier[r["tier"]].append(r)
    return by_key, by_tier


def _tier_count(selected, tier):
    return sum(1 for r in selected if r["tier"] == tier)


def _pick_method_contrast(by_key, tier, model, verb, strategies, group_id, rng):
    """Same model + verb + tier across strategies (method contrast)."""
    picked = []
    for s in strategies:
        r = by_key.get((model, s, verb))
        if r and r["tier"] == tier:
            picked.append({**r, "contrast_type": "method", "contrast_group": group_id})
    return picked


def _pick_model_contrast(by_key, tier, strategy, verb, models, group_id):
    """Same strategy + verb + tier across models (model contrast)."""
    picked = []
    for m in models:
        r = by_key.get((m, strategy, verb))
        if r and r["tier"] == tier:
            picked.append({**r, "contrast_type": "model", "contrast_group": group_id})
    return picked


def _pick_verb_contrast(by_key, tier, model, strategy, verbs, group_id):
    """Same model + strategy + tier across verbs (verb contrast)."""
    picked = []
    for v in verbs:
        r = by_key.get((model, strategy, v))
        if r and r["tier"] == tier:
            picked.append({**r, "contrast_type": "verb", "contrast_group": group_id})
    return picked


def _method_contrast_candidates(by_key, tier, min_strategies=3):
    """(model, verb) -> sorted strategies available at tier."""
    groups = defaultdict(set)
    for (model, strategy, verb), r in by_key.items():
        if r["tier"] == tier:
            groups[(model, verb)].add(strategy)
    out = []
    for (model, verb), strats in groups.items():
        if len(strats) >= min_strategies:
            ordered = [s for s in V7_AGENT_TABLE_STRATEGY_ORDER if s in strats]
            out.append((model, verb, ordered))
    return out


def _model_contrast_candidates(by_key, tier, min_models=3):
    """(strategy, verb) -> sorted models available at tier."""
    groups = defaultdict(set)
    for (model, strategy, verb), r in by_key.items():
        if r["tier"] == tier:
            groups[(strategy, verb)].add(model)
    out = []
    for (strategy, verb), models in groups.items():
        if len(models) >= min_models:
            ordered = [m for m in MODEL_ORDER if m in models]
            out.append((strategy, verb, ordered))
    return out


def _verb_contrast_candidates(by_key, tier, min_verbs=3):
    """(model, strategy) -> sorted verbs available at tier."""
    groups = defaultdict(set)
    for (model, strategy, verb), r in by_key.items():
        if r["tier"] == tier:
            groups[(model, strategy)].add(verb)
    out = []
    for (model, strategy), verbs in groups.items():
        if len(verbs) >= min_verbs:
            out.append((model, strategy, sorted(verbs)))
    return out


def _fill_tier_quota(selected, used, by_tier, tier, quota, rng, contrast_type="representative"):
    """Stratified fill: balance strategy and model within tier."""
    need = quota - _tier_count(selected, tier)
    if need <= 0:
        return
    candidates = [r for r in by_tier[tier] if _pool_key(r) not in used]
    if not candidates:
        return

    strat_counts = Counter(r["strategy"] for r in selected if r["tier"] == tier)
    model_counts = Counter(r["model"] for r in selected if r["tier"] == tier)

    def score(r):
        return strat_counts[r["strategy"]] + model_counts[r["model"]]

    rng.shuffle(candidates)
    candidates.sort(key=score)
    group_id = f"{tier.lower()}_fill"
    for r in candidates:
        if need <= 0:
            break
        k = _pool_key(r)
        if k in used:
            continue
        used.add(k)
        selected.append({
            **r,
            "contrast_type": contrast_type,
            "contrast_group": group_id,
        })
        strat_counts[r["strategy"]] += 1
        model_counts[r["model"]] += 1
        need -= 1


def select_v7_agent_paper_examples(pool, target_rows=PAPER_TABLE_TARGET_ROWS,
                                   tier_quota=None, seed=PAPER_TABLE_SAMPLE_SEED):
    """
    Stratified sample for a full-page paper table with contrast blocks.

    Priority: method contrast (fix model+verb, vary strategy) > model contrast
    (fix strategy+verb, vary model) > verb contrast (fix model+strategy, vary verb)
    > tier-representative fill to hit per-tier quotas.
    """
    if tier_quota is None:
        tier_quota = dict(PAPER_TABLE_TIER_QUOTA)
    assert sum(tier_quota.values()) == target_rows

    rng = random.Random(seed)
    by_key, by_tier = _index_pool(pool)
    selected, used = [], set()
    group_serial = 0

    def commit(block):
        nonlocal group_serial
        for r in block:
            k = _pool_key(r)
            if k in used:
                continue
            if _tier_count(selected, r["tier"]) >= tier_quota[r["tier"]]:
                continue
            used.add(k)
            selected.append(r)

    preferred_models = ["gpt-5.1", "sonnet-4.6"]

    def _sort_method_cands(cands):
        rng.shuffle(cands)
        cands.sort(key=lambda x: (-len(x[2]), 0 if x[0] in preferred_models else 1, x[1]))
        return cands

    def _sort_model_cands(cands, prefer_ss=False):
        rng.shuffle(cands)
        if prefer_ss:
            cands.sort(key=lambda x: (
                0 if x[0] in ("v7_agent_60", "v7_agent_verbalized_60") else 1,
                -len(x[2]),
            ))
        else:
            cands.sort(key=lambda x: (
                0 if x[0] in ("v7_agent_iterative_60", "v7_agent_verbalized_iterative_60") else 1,
                -len(x[2]),
            ))
        return cands

    def _add_method_blocks(tier, label, min_strategies=3, prefer_full_four=False):
        nonlocal group_serial
        cands = _method_contrast_candidates(by_key, tier, min_strategies)
        if prefer_full_four:
            full = [c for c in cands if len(c[2]) == 4]
            rest = [c for c in cands if len(c[2]) < 4]
            cands = _sort_method_cands(full) + _sort_method_cands(rest)
        else:
            cands = _sort_method_cands(cands)
        for model, verb, strats in cands:
            if _tier_count(selected, tier) >= tier_quota[tier]:
                break
            group_serial += 1
            commit(_pick_method_contrast(
                by_key, tier, model, verb, strats,
                f"{label}_method_{group_serial}", rng,
            ))

    def _add_model_blocks(tier, label, min_models=3, prefer_ss=False, max_per_block=6):
        nonlocal group_serial
        cands = _sort_model_cands(_model_contrast_candidates(by_key, tier, min_models), prefer_ss)
        for strategy, verb, models in cands:
            if _tier_count(selected, tier) >= tier_quota[tier]:
                break
            take = min(len(models), max_per_block, tier_quota[tier] - _tier_count(selected, tier))
            if take < 2:
                continue
            group_serial += 1
            commit(_pick_model_contrast(
                by_key, tier, strategy, verb, models[:take],
                f"{label}_model_{group_serial}",
            ))

    def _add_verb_blocks(tier, label, min_verbs=3, max_per_block=6):
        nonlocal group_serial
        cands = _verb_contrast_candidates(by_key, tier, min_verbs)
        rng.shuffle(cands)
        cands.sort(key=lambda x: -len(x[2]))
        for model, strategy, verbs in cands:
            if _tier_count(selected, tier) >= tier_quota[tier]:
                break
            take = min(len(verbs), max_per_block, tier_quota[tier] - _tier_count(selected, tier))
            if take < 2:
                continue
            group_serial += 1
            commit(_pick_verb_contrast(
                by_key, tier, model, strategy, verbs[:take],
                f"{label}_verb_{group_serial}",
            ))

    # ── GOLD: 4-strategy method blocks, then 3-strategy, then model contrast ──
    _add_method_blocks("GOLD", "gold", min_strategies=4, prefer_full_four=True)
    _add_method_blocks("GOLD", "gold", min_strategies=3)
    _add_model_blocks("GOLD", "gold", min_models=3, max_per_block=6)

    # ── SILVER: model then verb contrasts ──
    _add_model_blocks("SILVER", "silver", min_models=3, prefer_ss=True, max_per_block=6)
    _add_verb_blocks("SILVER", "silver", min_verbs=3, max_per_block=6)

    # ── BRONZE: verb contrast (sparse; RB/R2V only) ──
    _add_verb_blocks("BRONZE", "bronze", min_verbs=3, max_per_block=8)

    # ── FAIL: method, model, verb contrasts ──
    _add_method_blocks("FAIL", "fail", min_strategies=3)
    _add_model_blocks("FAIL", "fail", min_models=3, prefer_ss=True, max_per_block=6)
    _add_verb_blocks("FAIL", "fail", min_verbs=3, max_per_block=6)

    # Per-tier stratified fill
    for tier in ["GOLD", "SILVER", "BRONZE", "FAIL"]:
        _fill_tier_quota(selected, used, by_tier, tier, tier_quota[tier], rng)

    # Global top-up if any quota still short (shouldn't happen)
    if len(selected) < target_rows:
        rest = [r for r in pool if _pool_key(r) not in used]
        rng.shuffle(rest)
        for r in rest:
            if len(selected) >= target_rows:
                break
            if _tier_count(selected, r["tier"]) >= tier_quota.get(r["tier"], 0):
                continue
            used.add(_pool_key(r))
            selected.append({**r, "contrast_type": "representative", "contrast_group": "extra"})

    # Stable table order: tier → contrast type → group → model → strategy → verb
    contrast_order = {"method": 0, "model": 1, "verb": 2, "representative": 3}
    tier_order = {"GOLD": 0, "SILVER": 1, "BRONZE": 2, "FAIL": 3}
    selected.sort(key=lambda r: (
        tier_order.get(r["tier"], 9),
        contrast_order.get(r.get("contrast_type", "representative"), 9),
        r.get("contrast_group", ""),
        MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
        V7_AGENT_TABLE_STRATEGY_ORDER.index(r["strategy"])
        if r["strategy"] in V7_AGENT_TABLE_STRATEGY_ORDER else 99,
        r["verb"],
    ))
    for i, r in enumerate(selected, start=1):
        r["table_row_id"] = i
    return selected[:target_rows]


def write_v7_agent_paper_examples_table(selected, pool, output_csv, output_md):
    """Write CSV table + markdown metadata (sizing rationale, contrast map)."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    fieldnames = [
        "table_row_id", "contrast_type", "contrast_group", "tier",
        "model", "strategy", "strategy_label", "verb",
        "score_strict", "scene_verdict", "visual_verdict",
        "verb_form", "patient", "instrument", "location",
        "agent_PE", "agent_PH", "agent_IH", "agent_IE",
        "sentence_PE", "sentence_PH", "sentence_IH", "sentence_IE",
        "eval_run",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    pool_tiers = Counter(r["tier"] for r in pool)
    sel_tiers = Counter(r["tier"] for r in selected)
    sel_contrast = Counter(r.get("contrast_type", "") for r in selected)
    sel_strat = Counter(r["strategy_label"] for r in selected)
    sel_model = Counter(r["model"] for r in selected)

    lines = [
        "# v7_agent paper examples table — metadata",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Scope",
        "",
        "- **Framing:** `v7_agent` (agent-varying gradient)",
        "- **Strategies:** RB / R2V (`wRes+ST`) and R2R / R2VR (`wRe+ST`) only",
        "- **Judge:** GPT (`evalv3.1_agent_`)",
        "- **CSV:** `" + output_csv + "`",
        "",
        "## Full-page table capacity estimate",
        "",
        "For a US-letter / ACL-style **full-page `table*`** with ~8pt text and four",
        "sentence columns (PE / PH / IE / IH) plus metadata columns:",
        "",
        "| Layout | Approx. rows | Notes |",
        "|--------|-------------|-------|",
        "| Full sentences, wrapped (1 page) | **20–24** | Single full-page `table*` |",
        "| Full sentences, wrapped (1 page) | **20–24** | Current target (used here) |",
        "| Truncated sentences (~12 words) | 56–64 | One landscape page with `\\resizebox` |",
        "| One example sentence + tier badge | 70–80 | Loses gradient contrast |",
        "",
        f"**Target rows:** {PAPER_TABLE_TARGET_ROWS} "
        f"(tier quotas: {', '.join(f'{t}={n}' for t, n in PAPER_TABLE_TIER_QUOTA.items())}).",
        "",
        "## Sampling pool (before selection)",
        "",
        f"- Total cells with generations: **{len(pool)}**",
        f"- Tier counts: " + ", ".join(f"{t}={pool_tiers[t]}" for t in ["GOLD", "SILVER", "BRONZE", "FAIL"]),
        "",
        "## Selected table",
        "",
        f"- Rows written: **{len(selected)}**",
        f"- Selected tiers: " + ", ".join(f"{t}={sel_tiers[t]}" for t in ["GOLD", "SILVER", "BRONZE", "FAIL"]),
        f"- Contrast types: " + ", ".join(f"{k}={v}" for k, v in sorted(sel_contrast.items())),
        "",
        "### Contrast blocks (fix one factor, vary another)",
        "",
        "| `contrast_type` | Fixed | Varied | Purpose |",
        "|-----------------|-------|--------|---------|",
        "| `method` | model, verb, tier | strategy (RB→R2VR) | Show effect of reasoning / verbalized / iterative pipeline |",
        "| `model` | strategy, verb, tier | generator model | Show capability gap across model scale |",
        "| `verb` | model, strategy, tier | verb | Show event difficulty spread at same pipeline |",
        "| `representative` | — | stratified fill | Round out per-tier quotas |",
        "",
        "Rows sharing the same `contrast_group` belong to one contrast block.",
        "",
        "### Strategy coverage in selection",
        "",
        "| Strategy | Rows |",
        "|----------|------|",
    ]
    for s in V7_AGENT_TABLE_STRATEGY_ORDER:
        lab = V7_AGENT_TABLE_STRATEGY_LABELS[s]
        lines.append(f"| {lab} | {sel_strat.get(lab, 0)} |")
    lines.extend([
        "",
        "### Model coverage in selection",
        "",
        "| Model | Rows |",
        "|-------|------|",
    ])
    for m in MODEL_ORDER:
        if sel_model.get(m, 0):
            lines.append(f"| {m} | {sel_model[m]} |")
    lines.extend([
        "",
        "## Contrast groups in this extract",
        "",
    ])
    by_group = defaultdict(list)
    for r in selected:
        by_group[r.get("contrast_group", "")].append(r)
    for gid in sorted(by_group):
        if not gid:
            continue
        rows_g = by_group[gid]
        ct = rows_g[0].get("contrast_type", "")
        tier = rows_g[0]["tier"]
        lines.append(f"### `{gid}` ({ct}, tier={tier}, n={len(rows_g)})")
        lines.append("")
        lines.append("| row | model | strategy | verb | tier |")
        lines.append("|-----|-------|----------|------|------|")
        for r in rows_g:
            lines.append(
                f"| {r['table_row_id']} | {r['model']} | {r['strategy_label']} "
                f"| {r['verb']} | {r['tier']} |"
            )
        lines.append("")

    lines.extend([
        "## Reproducibility",
        "",
        f"- Seed: `{PAPER_TABLE_SAMPLE_SEED}`",
        "- Script: `analysis/clean_sampling_w_charts.py` → `select_v7_agent_paper_examples()`",
        "",
    ])

    with open(output_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved v7_agent paper examples CSV: {output_csv} ({len(selected)} rows)")
    print(f"Saved v7_agent paper examples metadata: {output_md}")


# ─────────────────────────────── main pipeline ────────────────────────────────

ANNOTATION_STRATEGIES = {
    "v4_60", "v6_60", "v6_verbalized_60", "v7_agent_60", "v7_agent_verbalized_60",
    "iterative_v6_60", "iterative_v6_verbalized_init_60", "v7_60", "v7_verbalized_60",
    "v7_agent_iterative_60", "v7_agent_verbalized_iterative_60",
    "v7_patient_60", "v7_patient_verbalized_60",
    "v7_patient_iterative_60", "v7_patient_verbalized_iterative_60",
}

with open("results/aggregate/all_evaluations_merged_preview.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter and deduplicate
seen, filtered = set(), []
for r in rows:
    eval_run = r["eval_run"]
    if (
        eval_run.startswith("evalv3_")
        or eval_run.startswith("evalv3.1_agent_")
        or eval_run.startswith("evalv3.1_patient_")
    ):
        model, strategy = parse_source_run(eval_run)
        if not model or not strategy or strategy not in ANNOTATION_STRATEGIES:
            continue
        key = (model, strategy, r["verb"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append({
            "eval_run":     eval_run,
            "model":        model,
            "strategy":     strategy,
            "verb":         r["verb"],
            "tier":         r["tier"],
            "score_strict": r.get("score_strict", ""),
            "visual_verdict": r.get("visual_verdict", ""),
            "scene_verdict":  r.get("scene_verdict", ""),
        })

# Join with actual sentences
output_rows = []
for r in filtered:
    gen = load_generation(r["model"], r["strategy"], r["verb"])
    if not gen:
        print(f"WARNING: No generation found for {r['model']} / {r['strategy']} / {r['verb']}")
        continue
    scene     = gen.get("scene", {})
    sentences = gen.get("sentences", {})
    agents    = gen.get("agents", {})
    output_rows.append({
        "model":    r["model"],
        "strategy": r["strategy"],
        "verb":     r["verb"],
        "tier":     r["tier"],
        "score_strict":   r["score_strict"],
        "visual_verdict": r["visual_verdict"],
        "scene_verdict":  r["scene_verdict"],
        "verb_form":  scene.get("verb_form", ""),
        "patient":    scene.get("patient", ""),
        "instrument": scene.get("instrument", ""),
        "location":   scene.get("location", ""),
        "agent_PE":   agents.get("plausible_easy",   {}).get("agent", ""),
        "agent_PH":   agents.get("plausible_hard",   {}).get("agent", ""),
        "agent_IH":   agents.get("implausible_hard", {}).get("agent", ""),
        "agent_IE":   agents.get("implausible_easy", {}).get("agent", ""),
        "sentence_PE": sentences.get("plausible_easy",   ""),
        "sentence_PH": sentences.get("plausible_hard",   ""),
        "sentence_IH": sentences.get("implausible_hard", ""),
        "sentence_IE": sentences.get("implausible_easy", ""),
    })

# Write CSV
OUTPUT_CSV = "results/aggregate/annotation_sampling_pool.csv"
if output_rows:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

print(f"\nWrote {len(output_rows)} items to {OUTPUT_CSV}")
print(f"  GOLD:   {sum(1 for r in output_rows if r['tier'] == 'GOLD')}")
print(f"  SILVER: {sum(1 for r in output_rows if r['tier'] == 'SILVER')}")
print(f"  FAIL:   {sum(1 for r in output_rows if r['tier'] == 'FAIL')}")

# v7_agent full-page paper examples (GOLD / SILVER / BRONZE / FAIL, stratified contrasts)
v7_agent_pool = _build_v7_agent_table_pool(rows)
if v7_agent_pool:
    paper_examples = select_v7_agent_paper_examples(v7_agent_pool)
    write_v7_agent_paper_examples_table(
        paper_examples,
        v7_agent_pool,
        PAPER_EXAMPLES_CSV,
        PAPER_EXAMPLES_MD,
    )
    print(f"\n--- v7_agent paper examples ({len(paper_examples)} rows) ---")
    for tier in ["GOLD", "SILVER", "BRONZE", "FAIL"]:
        n = sum(1 for r in paper_examples if r["tier"] == tier)
        print(f"  {tier}: {n}")
    for ct in ["method", "model", "verb", "representative"]:
        n = sum(1 for r in paper_examples if r.get("contrast_type") == ct)
        print(f"  contrast {ct}: {n}")
else:
    print("WARNING: Empty v7_agent table pool; skipping paper examples export.")

# Build aggregation groups
model_strategy_groups: dict = {}
strategy_groups: dict       = {}
for r in filtered:
    model_strategy_groups.setdefault((r["model"], r["strategy"]), []).append(r)
    strategy_groups.setdefault(r["strategy"], []).append(r)

# ── Text tables (unchanged from original) ──
print("\n\n--- Per-Strategy Tier Distribution (across all models) ---")
print(f"{'Strategy':<40} {'GOLD':>6} {'SILVER':>6} {'BRONZE':>6} {'FAIL':>6} {'Total':>6} {'GOLD%':>7}")
print("-" * 80)
for strategy in sorted(strategy_groups):
    items = strategy_groups[strategy]
    c     = Counter(r["tier"] for r in items)
    n     = len(items)
    print(f"{strategy:<40} {c.get('GOLD',0):>6} {c.get('SILVER',0):>6} "
          f"{c.get('BRONZE',0):>6} {c.get('FAIL',0):>6} {n:>6} "
          f"{c.get('GOLD',0)/n*100:>6.1f}%")

print("\n\n--- Per-Model x Strategy Tier Distribution ---")
print(f"{'Model':<16} {'Strategy':<40} {'GOLD':>6} {'SILVER':>6} {'BRONZE':>6} "
      f"{'FAIL':>6} {'Total':>6} {'GOLD%':>7}")
print("-" * 95)
for (model, strategy), items in sorted(model_strategy_groups.items()):
    c = Counter(r["tier"] for r in items)
    n = len(items)
    print(f"{model:<16} {strategy:<40} {c.get('GOLD',0):>6} {c.get('SILVER',0):>6} "
          f"{c.get('BRONZE',0):>6} {c.get('FAIL',0):>6} {n:>6} "
          f"{c.get('GOLD',0)/n*100:>6.1f}%")

print("\n--- GOLD Items Preview ---")
golds = [r for r in output_rows if r["tier"] == "GOLD"]
for r in sorted(golds[:10], key=lambda x: (x["model"], x["strategy"], x["verb"])):
    print(f"\n  {r['model']} / {r['strategy']} / {r['verb']}")
    for cond, key in [("PE", "sentence_PE"), ("PH", "sentence_PH"),
                      ("IH", "sentence_IH"), ("IE", "sentence_IE")]:
        print(f"    {cond}: {r[key]}")

# ── Chart generation ──
os.makedirs("results/figures", exist_ok=True)

plot_combined(
    strategy_groups,
    model_strategy_groups,
    "results/figures/results_combined.png",
)
plot_strategy_only(
    strategy_groups,
    "results/figures/strategy_tier_distribution.png",
)

# Judge-specific aggregation (GPT / Qwen / Sonnet eval prefixes)
filtered_gpt = _dedupe_rows_by_eval_prefix(rows, EVAL_RUN_PREFIX_GPT)
filtered_qwen = _dedupe_rows_by_eval_prefix(rows, EVAL_RUN_PREFIX_QWEN)
filtered_sonnet = _dedupe_rows_by_eval_prefix(rows, EVAL_RUN_PREFIX_SONNET)
strategy_groups_gpt, model_strategy_groups_gpt = _build_strategy_and_model_groups(filtered_gpt)
strategy_groups_qwen, model_strategy_groups_qwen = _build_strategy_and_model_groups(filtered_qwen)
strategy_groups_sonnet, model_strategy_groups_sonnet = _build_strategy_and_model_groups(filtered_sonnet)

os.makedirs("results/aggregate", exist_ok=True)
for prefix, strategy_groups_j, model_strategy_groups_j, md_name in [
    (EVAL_RUN_PREFIX_GPT, strategy_groups_gpt, model_strategy_groups_gpt, "tier_tables_gpt_judge.md"),
    (EVAL_RUN_PREFIX_QWEN, strategy_groups_qwen, model_strategy_groups_qwen, "tier_tables_qwen30b_think_judge.md"),
    (EVAL_RUN_PREFIX_SONNET, strategy_groups_sonnet, model_strategy_groups_sonnet, "tier_tables_sonnet_judge.md"),
]:
    write_judge_summary_markdown(
        JUDGE_TABLE_LABELS[prefix],
        strategy_groups_j,
        model_strategy_groups_j,
        f"results/aggregate/{md_name}",
    )

_judge_strategy_charts = [
    (strategy_groups_qwen, "strategy_tier_distribution_qwen30b_think_judge.png",
     "Tier distribution by strategy  (all models combined, Qwen3-30B-Thinking judge)"),
    (strategy_groups_sonnet, "strategy_tier_distribution_sonnet_judge.png",
     "Tier distribution by strategy  (all models combined, Claude Sonnet 4.6 judge)"),
]
for strategy_groups_j, fig_name, title in _judge_strategy_charts:
    if strategy_groups_j:
        plot_strategy_only(
            strategy_groups_j,
            f"results/figures/{fig_name}",
            title=title,
        )
    else:
        print(f"WARNING: No rows for judge strategy tier chart ({fig_name}); skipping.")
plot_heatmap_only(
    model_strategy_groups,
    "results/figures/model_heatmap.png",
)
plot_v7_agent_vs_patient_heatmap(
    model_strategy_groups,
    "results/figures/model_heatmap_v7_agent_vs_patient.png",
)
plot_verb_framing_agent_vs_patient(
    filtered,
    "results/figures/verb_framing_agent_vs_patient_gold.png",
)

# Judge comparison rows (kept separate — do not mix into main aggregation)
seen_judge, judge_compare_rows = set(), []
for r in rows:
    judge = _eval_judge_tag(r["eval_run"])
    if judge is None:
        continue
    model, strategy = parse_source_run(r["eval_run"])
    if not model or strategy not in JUDGE_COMPARE_STRATEGIES:
        continue
    key = (judge, model, strategy, r["verb"])
    if key in seen_judge:
        continue
    seen_judge.add(key)
    judge_compare_rows.append({
        "judge":    judge,
        "model":    model,
        "strategy": strategy,
        "verb":     r["verb"],
        "tier":     r["tier"],
        "eval_run": r["eval_run"],
    })

if judge_compare_rows:
    all_judge_names = " / ".join(JUDGE_DISPLAY_NAMES[j] for j in JUDGE_COMPARE_ORDER)
    plot_judge_model_heatmap(
        judge_compare_rows,
        "results/figures/model_heatmap_judge_gpt_qwen_sonnet.png",
        JUDGE_HEATMAP_COLUMNS_ALL,
        f"GOLD% by generator model × strategy — {all_judge_names} judges",
        _judge_heatmap_footnote(JUDGE_COMPARE_ORDER),
    )
    for judge_a, judge_b, pair_slug in JUDGE_COMPARE_PAIRS:
        col_specs = _judge_heatmap_columns([judge_a, judge_b])
        name_a = JUDGE_DISPLAY_NAMES[judge_a]
        name_b = JUDGE_DISPLAY_NAMES[judge_b]
        plot_judge_model_heatmap(
            judge_compare_rows,
            f"results/figures/model_heatmap_judge_{pair_slug}.png",
            col_specs,
            f"GOLD% by generator model × strategy — {name_a} vs {name_b} judge",
            _judge_heatmap_footnote([judge_a, judge_b]),
        )
        short_a = JUDGE_SHORT_NAMES[judge_a].lower()
        short_b = JUDGE_SHORT_NAMES[judge_b].lower()
        verb_fig = f"results/figures/verb_judge_{short_a}_vs_{short_b}_gold.png"
        plot_verb_judge_comparison(judge_compare_rows, judge_a, judge_b, verb_fig)
        write_judge_agreement_tex(
            judge_compare_rows,
            judge_a,
            judge_b,
            f"latex_figures/judge_agreement_{pair_slug}.tex",
            figure_relpath=f"../results/figures/verb_judge_{short_a}_vs_{short_b}_gold.png",
        )
else:
    print("WARNING: No rows for judge comparison chart; skipping.")

# Failure-cause: v7_agent RB/R2V/R2R/R2VR per model, GPT judge (evalv3.1_agent_)
for model_label, eval_slug, file_slug in AGENT_GPT_JUDGE_FAILURE_MODELS:
    modalities = agent_gpt_judge_modalities(eval_slug)
    eval_runs = {run for _, run in modalities}
    model_rows = [r for r in rows if r["eval_run"] in eval_runs]
    plot_failure_causes_model_all_modalities(
        model_rows,
        f"results/figures/failure_causes_{file_slug}_v7_agent_all_modalities_gpt_judge.png",
        model_label,
        modalities,
    )
    ir_run = modalities[2][1]
    ir_rows = [r for r in model_rows if r["eval_run"] == ir_run]
    plot_failure_cause_distribution(
        ir_rows,
        f"results/figures/failure_causes_{file_slug}_v7_agent_iter_IR_gpt_judge.png",
        modality_label="R2R",
        model_label=model_label,
    )