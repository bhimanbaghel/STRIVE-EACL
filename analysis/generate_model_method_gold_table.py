"""
Generate paper Table 2: GOLD (%) per (model, method).

Cells are reported as GPT / Qwen30B-Think / Qwen3.8-27B-xhigh.
GPT and Qwen30B values come from the merged eval CSV; Qwen3.8-27B is read
from results/evaluations/evalv31agent_qwen38-27b-xhigh_* (and the CSV if present)
so the table can update as Slurm jobs finish, before re-merging.

Run from project root:
    uv run python analysis/generate_model_method_gold_table.py
    uv run python analysis/generate_model_method_gold_table.py --check

Outputs:
    latex_figures/tab_model_method_gold.tex
    results/aggregate/tab_model_method_gold.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "aggregate" / "all_evaluations_merged_preview.csv"
DEFAULT_TEX = ROOT / "latex_figures" / "tab_model_method_gold.tex"
DEFAULT_OUT_CSV = ROOT / "results" / "aggregate" / "tab_model_method_gold.csv"
DEFAULT_EVAL_ROOT = ROOT / "results" / "evaluations"

EVAL_RUN_PREFIX_GPT = "evalv3.1_agent_"
EVAL_RUN_PREFIX_QWEN = "evalv31agent_qwen30b-think_"
# Distinct from AAT (...-xhigh-rN_) and ACE (..._qwen38-27b_{MODEL}_{METHOD}_15).
EVAL_RUN_PREFIX_QWEN38 = "evalv31agent_qwen38-27b-xhigh_"

# Paper methods: v7_agent (wRes+ST) RB/R2V and iterative R2R/R2VR.
METHODS = [
    ("RB", "v7_agent_60"),
    ("R2V", "v7_agent_verbalized_60"),
    ("R2R", "v7_agent_iterative_60"),
    ("R2VR", "v7_agent_verbalized_iterative_60"),
]
METHOD_STRATEGIES = {s for _, s in METHODS}

MODEL_ORDER = [
    "gpt-5.1",
    "sonnet-4.6",
    "Qwen3-Next-80B",
    "Qwen3-30B",
    "Ministral-14B",
    "Qwen3-4B",
]
MODEL_DISPLAY = {
    "gpt-5.1": "GPT-5.1",
    "sonnet-4.6": "Sonnet 4.6",
    "Qwen3-Next-80B": "Qwen3-80B",
    "Qwen3-30B": "Qwen3-30B",
    "Ministral-14B": "Ministral-14B",
    "Qwen3-4B": "Qwen3-4B",
}
CLOSED_SOURCE = {"gpt-5.1", "sonnet-4.6"}
# Paper omits GPT-judge for open generators on iterative methods (budget).
GPT_MASKED_STRATEGIES = {
    "v7_agent_iterative_60",
    "v7_agent_verbalized_iterative_60",
}

# Reference cells from the paper TeX (GPT / Qwen30 GOLD% only).
EXPECTED = {
    "GPT-5.1": {
        "RB": ("28.3", "51.7"),
        "R2V": ("13.3", "56.7"),
        "R2R": ("75.0", "41.7"),
        "R2VR": ("71.7", "60.0"),
    },
    "Sonnet 4.6": {
        "RB": ("28.3", "51.7"),
        "R2V": ("21.7", "51.7"),
        "R2R": ("74.5", "56.4"),
        "R2VR": ("81.7", "45.0"),
    },
    "Qwen3-80B": {
        "RB": ("3.3", "31.7"),
        "R2V": ("6.7", "33.3"),
        "R2R": (None, "81.7"),
        "R2VR": (None, "66.7"),
    },
    "Qwen3-30B": {
        "RB": ("6.7", "41.7"),
        "R2V": ("10.0", "31.7"),
        "R2R": (None, "73.3"),
        "R2VR": (None, "83.3"),
    },
    "Ministral-14B": {
        "RB": ("5.1", "30.5"),
        "R2V": ("8.3", "20.0"),
        "R2R": (None, "70.0"),
        "R2VR": (None, "81.7"),
    },
    "Qwen3-4B": {
        "RB": ("6.7", "11.7"),
        "R2V": ("10.0", "28.3"),
        "R2R": (None, "66.7"),
        "R2VR": (None, "71.7"),
    },
    "All models": {
        "RB": ("13.1", "36.5"),
        "R2V": ("11.7", "36.9"),
        "R2R": ("74.8*", "65.1"),
        "R2VR": ("76.7*", "68.1"),
    },
}


def parse_source_run(eval_run: str) -> tuple[str | None, str | None]:
    """Extract clean model name and strategy from eval_run (same rules as charts)."""
    run = eval_run.replace("evalv3_", "")
    run = run.replace("evalv3.1_agent_", "")
    run = run.replace("evalv3.1_patient_", "")
    # Longer prefixes first so they are not partially stripped.
    run = run.replace("evalv31agent_qwen38-27b-xhigh_", "")
    run = run.replace("evalv31agent_qwen30b-think_", "")
    run = run.replace("evalv31agent_sonnet-4.6_", "")

    model_map = {
        "gpt-5.1-2025-11-13": "gpt-5.1",
        "Qwen3-Next-80B-A3B-Instruct": "Qwen3-Next-80B",
        "Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B",
        "Ministral-3-14B-Instruct-2512": "Ministral-14B",
        "Qwen3-4B-Instruct-2507": "Qwen3-4B",
        "claude-sonnet-4-6": "sonnet-4.6",
    }

    strategy = None
    for s in [
        "iterative_v6_verbalized_init_60",
        "iterative_v6_60",
        "iterative_v5",
        "v6_verbalized_60",
        "v6_60",
        "v5",
        "v4_60",
        "v7_agent_verbalized_iterative_60",
        "v7_patient_verbalized_iterative_60",
        "v7_patient_verbalized_60",
        "v7_agent_verbalized_60",
        "v7_verbalized_60",
        "v7_agent_iterative_60",
        "v7_patient_iterative_60",
        "v7_agent_60",
        "v7_patient_60",
        "v7_60",
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


def assign_tier_from_eval_json(data: dict) -> str:
    """Match scripts/merge_iterative_to_evals.py assign_tier."""
    if not data or not data.get("success"):
        return "FAIL"
    scores = data.get("scores") or {}
    try:
        score = int(scores.get("score_strict", 0) or 0)
    except (TypeError, ValueError):
        return "FAIL"
    if score < 4:
        return "FAIL"
    ev = data.get("evaluation") or {}
    scene = (ev.get("scene_assessment") or {}).get("scene_verdict", "")
    visual = (ev.get("visual_evaluation") or {}).get("visual_verdict", "")
    if scene == "GOOD" and visual == "ALL_DISTINCT":
        return "GOLD"
    if scene == "GOOD" and visual == "MOSTLY_DISTINCT":
        return "SILVER"
    if score == 4:
        return "BRONZE"
    return "FAIL"


def load_judge_cells(csv_path: Path, eval_prefix: str) -> dict[tuple[str, str], tuple[int, int]]:
    """Return {(model, strategy): (gold, n)} for one judge prefix from merged CSV."""
    seen: set[tuple[str, str, str]] = set()
    gold_n: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    with csv_path.open() as f:
        for r in csv.DictReader(f):
            eval_run = r["eval_run"]
            if not eval_run.startswith(eval_prefix):
                continue
            # Exclude AAT replicate tags accidentally sharing a shorter prefix.
            if eval_prefix == EVAL_RUN_PREFIX_QWEN38 and "-xhigh-r" in eval_run:
                continue
            model, strategy = parse_source_run(eval_run)
            if not model or strategy not in METHOD_STRATEGIES:
                continue
            key = (model, strategy, r["verb"])
            if key in seen:
                continue
            seen.add(key)
            cell = gold_n[(model, strategy)]
            cell[1] += 1
            if r["tier"] == "GOLD":
                cell[0] += 1

    return {k: (v[0], v[1]) for k, v in gold_n.items()}


def load_judge_cells_from_eval_dirs(
    eval_root: Path, eval_prefix: str,
) -> dict[tuple[str, str], tuple[int, int]]:
    """Load GOLD rates directly from results/evaluations/<prefix>* dirs."""
    seen: set[tuple[str, str, str]] = set()
    gold_n: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    if not eval_root.is_dir():
        return {}

    for d in sorted(eval_root.iterdir()):
        if not d.is_dir() or not d.name.startswith(eval_prefix):
            continue
        if "-xhigh-r" in d.name:  # skip AAT replicates
            continue
        model, strategy = parse_source_run(d.name)
        if not model or strategy not in METHOD_STRATEGIES:
            continue
        for fp in d.glob("*.json"):
            verb = fp.stem.lower()
            key = (model, strategy, verb)
            if key in seen:
                continue
            try:
                data = json.loads(fp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not data.get("success"):
                continue
            seen.add(key)
            cell = gold_n[(model, strategy)]
            cell[1] += 1
            if assign_tier_from_eval_json(data) == "GOLD":
                cell[0] += 1

    return {k: (v[0], v[1]) for k, v in gold_n.items()}


def merge_cell_dicts(
    primary: dict[tuple[str, str], tuple[int, int]],
    secondary: dict[tuple[str, str], tuple[int, int]],
) -> dict[tuple[str, str], tuple[int, int]]:
    """Prefer primary counts; fill gaps from secondary."""
    out = dict(primary)
    for k, v in secondary.items():
        if k not in out:
            out[k] = v
    return out


def fmt_pct(gold: int, n: int) -> str:
    return f"{100.0 * gold / n:.1f}"


def _sum_cells(cells: dict[tuple[str, str], tuple[int, int]], models: list[str], strategy: str):
    g = n = 0
    for m in models:
        if (m, strategy) not in cells:
            continue
        cg, cn = cells[(m, strategy)]
        g += cg
        n += cn
    return g, n


def lookup_cell(
    cells: dict[tuple[str, str], tuple[int, int]], model: str, strategy: str,
) -> tuple[str | None, int, int]:
    if (model, strategy) not in cells:
        return None, 0, 0
    g, n = cells[(model, strategy)]
    return fmt_pct(g, n), g, n


def build_table_rows(
    gpt: dict[tuple[str, str], tuple[int, int]],
    qwen: dict[tuple[str, str], tuple[int, int]],
    qwen38: dict[tuple[str, str], tuple[int, int]],
) -> list[dict]:
    rows: list[dict] = []

    for model in MODEL_ORDER:
        display = MODEL_DISPLAY[model]
        cells = {}
        for method, strategy in METHODS:
            report_gpt = not (
                strategy in GPT_MASKED_STRATEGIES and model not in CLOSED_SOURCE
            )
            gpt_pct, gpt_gold, gpt_n = (
                lookup_cell(gpt, model, strategy) if report_gpt else (None, 0, 0)
            )
            qwen_pct, qwen_gold, qwen_n = lookup_cell(qwen, model, strategy)
            q38_pct, q38_gold, q38_n = lookup_cell(qwen38, model, strategy)
            cells[method] = {
                "gpt": gpt_pct,
                "qwen": qwen_pct,
                "qwen38": q38_pct,
                "gpt_star": False,
                "gpt_gold": gpt_gold,
                "gpt_n": gpt_n,
                "qwen_gold": qwen_gold,
                "qwen_n": qwen_n,
                "qwen38_gold": q38_gold,
                "qwen38_n": q38_n,
            }
        rows.append({"model": display, "cells": cells})

    agg_cells = {}
    for method, strategy in METHODS:
        gpt_only_closed = strategy in GPT_MASKED_STRATEGIES
        gpt_models = [m for m in MODEL_ORDER if m in CLOSED_SOURCE] if gpt_only_closed else MODEL_ORDER
        g_g, g_n = _sum_cells(gpt, gpt_models, strategy)
        q_g, q_n = _sum_cells(qwen, MODEL_ORDER, strategy)
        q38_g, q38_n = _sum_cells(qwen38, MODEL_ORDER, strategy)
        gpt_pct = fmt_pct(g_g, g_n) if g_n else None
        qwen_pct = fmt_pct(q_g, q_n) if q_n else None
        q38_pct = fmt_pct(q38_g, q38_n) if q38_n else None
        agg_cells[method] = {
            "gpt": gpt_pct,
            "qwen": qwen_pct,
            "qwen38": q38_pct,
            "gpt_star": gpt_only_closed and gpt_pct is not None,
            "gpt_gold": g_g,
            "gpt_n": g_n,
            "qwen_gold": q_g,
            "qwen_n": q_n,
            "qwen38_gold": q38_g,
            "qwen38_n": q38_n,
        }
    rows.append({"model": "All models", "cells": agg_cells})
    return rows


def format_triple_cell(cell: dict, *, latex: bool) -> str:
    def one(val: str | None, star: bool = False) -> str:
        if val is None:
            return "—"
        if star:
            return f"{val}\\textsuperscript{{*}}" if latex else f"{val}*"
        return val

    return (
        f"{one(cell['gpt'], cell.get('gpt_star', False))} / "
        f"{one(cell['qwen'])} / "
        f"{one(cell['qwen38'])}"
    )


def render_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{RB} & \textbf{R2V} & \textbf{R2R} & \textbf{R2VR} \\",
        r"\midrule",
    ]
    for row in rows:
        if row["model"] == "All models":
            lines.append(r"\midrule")
        cells = [format_triple_cell(row["cells"][m], latex=True) for m, _ in METHODS]
        model = row["model"]
        pad = max(0, 14 - len(model))
        lines.append(f"{model}{' ' * pad} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{GOLD (\%) per (model, method) cell, reported as",
        r"\emph{GPT / Qwen3-30B-Thinking / Qwen3.8-27B-xhigh}.",
        r"``—'' indicates unavailable GPT-judge cells (budget) or pending",
        r"Qwen3.8-27B evaluations.",
        r"\textsuperscript{*}GPT-judge aggregate covers only the two",
        r"closed-source generators ($n{=}115$ for R2R, $n{=}120$ for R2VR)}",
        r"\label{tab:model_method}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "method",
        "gpt_gold", "gpt_n", "gpt_pct", "gpt_star",
        "qwen_gold", "qwen_n", "qwen_pct",
        "qwen38_gold", "qwen38_n", "qwen38_pct",
        "cell",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            for method, _ in METHODS:
                cell = row["cells"][method]
                w.writerow({
                    "model": row["model"],
                    "method": method,
                    "gpt_gold": cell["gpt_gold"],
                    "gpt_n": cell["gpt_n"],
                    "gpt_pct": cell["gpt"] or "",
                    "gpt_star": "1" if cell.get("gpt_star") else "0",
                    "qwen_gold": cell["qwen_gold"],
                    "qwen_n": cell["qwen_n"],
                    "qwen_pct": cell["qwen"] or "",
                    "qwen38_gold": cell["qwen38_gold"],
                    "qwen38_n": cell["qwen38_n"],
                    "qwen38_pct": cell["qwen38"] or "",
                    "cell": format_triple_cell(cell, latex=False),
                })


def check_against_expected(rows: list[dict]) -> list[str]:
    """Validate GPT / Qwen30 cells against the published dual-judge table."""
    errors: list[str] = []
    by_model = {r["model"]: r for r in rows}
    for model, expected_methods in EXPECTED.items():
        if model not in by_model:
            errors.append(f"missing row: {model}")
            continue
        for method, (exp_gpt, exp_qwen) in expected_methods.items():
            cell = by_model[model]["cells"][method]
            got_gpt = cell["gpt"]
            if cell.get("gpt_star") and got_gpt is not None:
                got_gpt = f"{got_gpt}*"
            got_qwen = cell["qwen"]
            if got_gpt != exp_gpt or got_qwen != exp_qwen:
                errors.append(
                    f"{model} {method}: got {got_gpt!r}/{got_qwen!r}, "
                    f"expected {exp_gpt!r}/{exp_qwen!r}"
                )
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if GPT/Qwen30 cells differ from the paper reference",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: missing input CSV: {args.csv}", file=sys.stderr)
        return 1

    gpt = load_judge_cells(args.csv, EVAL_RUN_PREFIX_GPT)
    qwen = load_judge_cells(args.csv, EVAL_RUN_PREFIX_QWEN)
    qwen38_csv = load_judge_cells(args.csv, EVAL_RUN_PREFIX_QWEN38)
    qwen38_dirs = load_judge_cells_from_eval_dirs(args.eval_root, EVAL_RUN_PREFIX_QWEN38)
    # Prefer live eval dirs (fresher during Slurm runs) over possibly-stale CSV.
    qwen38 = merge_cell_dicts(qwen38_dirs, qwen38_csv)

    rows = build_table_rows(gpt, qwen, qwen38)

    print("Model × method GOLD%  (GPT / Qwen30B-Think / Qwen3.8-27B-xhigh)\n")
    header = f"{'Model':<14} " + " ".join(f"{m:>22}" for m, _ in METHODS)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(
            f"{format_triple_cell(row['cells'][m], latex=False):>22}" for m, _ in METHODS
        )
        print(f"{row['model']:<14} {cells}")

    q38_cells = sum(1 for r in rows if r["model"] != "All models"
                    for m, _ in METHODS if r["cells"][m]["qwen38"] is not None)
    print(f"\nQwen3.8-27B-xhigh cells filled: {q38_cells}/24")

    agg = next(r for r in rows if r["model"] == "All models")
    for method in ("R2R", "R2VR"):
        c = agg["cells"][method]
        print(f"All-models GPT {method}: n={c['gpt_n']} gold={c['gpt_gold']} ({c['gpt']}*)")

    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(render_latex(rows))
    write_csv(rows, args.out_csv)
    print(f"\nWrote {args.tex}")
    print(f"Wrote {args.out_csv}")

    errors = check_against_expected(rows)
    if errors:
        print("\nReference check FAILED:")
        for e in errors:
            print(f"  - {e}")
        if args.check:
            return 1
        print("(re-run with --check to treat mismatches as errors)")
    else:
        print("\nReference check OK: GPT/Qwen30 cells match the paper table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
