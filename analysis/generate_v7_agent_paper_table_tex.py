"""
Generate Overleaf-ready LaTeX from v7_agent paper example CSV.

Usage (from project root):
    uv run python analysis/generate_v7_agent_paper_table_tex.py

Outputs:
    latex_figures/v7_agent_paper_examples_table.tex
    latex_figures/v7_agent_paper_examples_companion.tex
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INPUT_CSV = Path("results/aggregate/v7_agent_paper_examples_table.csv")
DEFAULT_TABLE_TEX = Path("latex_figures/v7_agent_paper_examples_table.tex")
DEFAULT_COMPANION_TEX = Path("latex_figures/v7_agent_paper_examples_companion.tex")

MODEL_DISPLAY_NAMES = {
    "Qwen3-Next-80B": "Qwen3-80B",
}


def latex_escape(text: str) -> str:
    repl = {
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
    return "".join(repl.get(ch, ch) for ch in str(text))


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0\\%"
    return f"{100.0 * count / total:.1f}\\%"


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


# Row background tints (match analysis/clean_sampling_w_charts.py tier palette, lightened).
TIER_ROW_COLOR = {
    "GOLD":   "tierGold",
    "SILVER": "tierSilver",
    "BRONZE": "tierBronze",
    "FAIL":   "tierFail",
}
TIER_COLOR_HTML = {
    "GOLD":   "BA7517",
    "SILVER": "888780",
    "BRONZE": "F0997B",
    "FAIL":   "E8E6DF",
}


def short_method_label(label: str) -> str:
    """RB (wRes+ST) -> RB."""
    text = str(label).strip()
    if " (" in text:
        return text.split(" (", 1)[0].strip()
    return text


def display_model_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def fail_highlight_sentence_index(row: dict) -> int:
    """
    Choose one sentence cell to emphasize in FAIL rows.
    Heuristic:
      - scene-level failure: highlight PE (core event sentence)
      - otherwise: highlight IE (extreme implausible endpoint often driving failure)
    Returns 0..3 for PE/PH/IH/IE.
    """
    scene = str(row.get("scene_verdict", "")).upper()
    # If failure is driven by scene-level verdict, do not add any sentence-level highlighting.
    if scene and scene != "GOOD":
        return -1
    return 3


ROWS_PER_TABLE = 24


def _append_single_table(lines: list[str], chunk: list[dict], table_idx: int, n_tables: int) -> None:
    """Append one table* chunk."""
    caption_suffix = ""
    if n_tables > 1:
        caption_suffix = f" (part {table_idx + 1}/{n_tables})"

    lines.append(r"\begin{table*}[p]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(
        r"\caption{Selected qualitative examples from v7\_agent runs (24 rows)."
        + caption_suffix
        + r"}"
    )
    if table_idx == 0:
        lines.append(r"\label{tab:v7-agent-paper-examples}")
    lines.append(r"\begin{tabularx}{\textwidth}{l l l X X X X}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & Method & Verb & PE sentence & PH sentence & IH sentence & IE sentence \\"
    )
    lines.append(r"\midrule")

    for r in chunk:
        tier = r.get("tier", "")
        row_color = TIER_ROW_COLOR.get(tier, "tierFail")
        row_shade = {"GOLD": 12, "SILVER": 24, "BRONZE": 24, "FAIL": 18}.get(tier, 18)
        verb_label = str(r.get("verb", ""))

        sentence_cells = [
            latex_escape(r.get("sentence_PE", "")),
            latex_escape(r.get("sentence_PH", "")),
            latex_escape(r.get("sentence_IH", "")),
            latex_escape(r.get("sentence_IE", "")),
        ]
        if tier == "FAIL":
            fail_idx = fail_highlight_sentence_index(r)
            if fail_idx >= 0:
                sentence_cells[fail_idx] = r"\textbf{" + sentence_cells[fail_idx] + "}"
        cells = [
            latex_escape(display_model_name(r.get("model", ""))),
            latex_escape(short_method_label(r.get("strategy_label", ""))),
            latex_escape(verb_label),
            *sentence_cells,
        ]
        lines.append(rf"\rowcolor{{{row_color}!{row_shade}}}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\vspace{3pt}")
    lines.append(r"{\footnotesize")
    lines.append(
        r"\colorbox{tierGold!35}{\strut\phantom{Gg}} GOLD\quad"
        r"\colorbox{tierSilver!35}{\strut\phantom{Gg}} SILVER\quad"
        r"\colorbox{tierBronze!35}{\strut\phantom{Gg}} BRONZE\quad"
        r"\colorbox{tierFail!55}{\strut\phantom{Gg}} FAIL"
    )
    lines.append(r"\\")
    lines.append(
        r"\textit{FAIL rows:} sentence highlighting is applied only when `scene_verdict=GOOD`."
    )
    lines.append(r"}")
    lines.append(r"\end{table*}")
    lines.append("")


def build_main_table_tex(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("% Auto-generated by generate_v7_agent_paper_table_tex.py")
    lines.append("% Required packages: booktabs, tabularx, colortbl, xcolor")
    lines.append(r"\definecolor{tierGold}{HTML}{" + TIER_COLOR_HTML["GOLD"] + "}")
    lines.append(r"\definecolor{tierSilver}{HTML}{" + TIER_COLOR_HTML["SILVER"] + "}")
    lines.append(r"\definecolor{tierBronze}{HTML}{" + TIER_COLOR_HTML["BRONZE"] + "}")
    lines.append(r"\definecolor{tierFail}{HTML}{" + TIER_COLOR_HTML["FAIL"] + "}")
    chunks = [rows[i:i + ROWS_PER_TABLE] for i in range(0, len(rows), ROWS_PER_TABLE)]
    for idx, chunk in enumerate(chunks):
        _append_single_table(lines, chunk, idx, len(chunks))

    return "\n".join(lines)


def build_companion_tex(rows: list[dict], input_csv: Path) -> str:
    n = len(rows)
    tier_counts = Counter(r.get("tier", "") for r in rows)
    strategy_counts = Counter(r.get("strategy_label", "") for r in rows)
    model_counts = Counter(display_model_name(r.get("model", "")) for r in rows)
    contrast_counts = Counter(r.get("contrast_type", "") for r in rows)
    verb_counts = Counter(r.get("verb", "") for r in rows)

    unique_verbs = len(verb_counts)
    unique_models = len(model_counts)
    unique_methods = len(strategy_counts)

    top_verbs = verb_counts.most_common(5)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("% Auto-generated by generate_v7_agent_paper_table_tex.py")
    lines.append("% Required package: booktabs")
    lines.append(r"\subsection{Companion Notes for Table~\ref{tab:v7-agent-paper-examples}}")
    lines.append(
        r"\textbf{Data source:} "
        + latex_escape(str(input_csv))
        + r" (\texttt{v7\_agent\_paper\_examples\_table.csv}); generated "
        + latex_escape(generated)
        + r"."
    )
    lines.append("")
    lines.append(r"\paragraph{Coverage.}")
    lines.append(
        f"The table contains {n} rows spanning {unique_verbs} verbs, "
        f"{unique_models} generator models, and {unique_methods} methods."
    )
    lines.append("")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Tier distribution in the selected table.}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Tier & Count & Share \\")
    lines.append(r"\midrule")
    for tier in ("GOLD", "SILVER", "BRONZE", "FAIL"):
        c = tier_counts.get(tier, 0)
        lines.append(f"{latex_escape(tier)} & {c} & {pct(c, n)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Method and contrast-type coverage.}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Group & Count & Share \\")
    lines.append(r"\midrule")
    for method in sorted(strategy_counts):
        c = strategy_counts[method]
        lines.append(f"{latex_escape(method)} & {c} & {pct(c, n)} \\\\")
    lines.append(r"\midrule")
    for ctype in sorted(contrast_counts):
        c = contrast_counts[ctype]
        lines.append(
            f"{latex_escape('contrast: ' + ctype)} & {c} & {pct(c, n)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\paragraph{Model balance.}")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Model coverage in the selected table.}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Count & Share \\")
    lines.append(r"\midrule")
    for model, c in sorted(model_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"{latex_escape(model)} & {c} & {pct(c, n)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\paragraph{Insights.}")
    lines.append(r"\begin{itemize}")
    lines.append(
        r"\item The selection is tier-stratified (12/12/8/16 across GOLD/SILVER/BRONZE/FAIL), so qualitative comparisons are not dominated by one outcome bucket."
    )
    lines.append(
        r"\item Contrast groups deliberately hold one factor fixed (method, model, or verb) while varying another, supporting cleaner attribution in qualitative discussion."
    )
    if top_verbs:
        top_blob = ", ".join(
            f"{latex_escape(v)} ({c})" for v, c in top_verbs
        )
        lines.append(
            r"\item Frequently recurring verbs in this sample are: "
            + top_blob
            + r", providing repeated anchors for cross-method and cross-model contrast."
        )
    lines.append(
        r"\item The table mixes single-shot (wRes+ST) and iterative (wRe+ST) methods, enabling side-by-side analysis of how refinement changes plausibility-gradient quality."
    )
    lines.append(r"\end{itemize}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert v7_agent paper CSV into Overleaf-ready LaTeX."
    )
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output_table_tex", default=str(DEFAULT_TABLE_TEX))
    parser.add_argument("--output_companion_tex", default=str(DEFAULT_COMPANION_TEX))
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_table_tex = Path(args.output_table_tex)
    output_companion_tex = Path(args.output_companion_tex)

    rows = read_rows(input_csv)
    output_table_tex.parent.mkdir(parents=True, exist_ok=True)
    output_companion_tex.parent.mkdir(parents=True, exist_ok=True)

    output_table_tex.write_text(build_main_table_tex(rows))
    output_companion_tex.write_text(build_companion_tex(rows, input_csv))

    print(f"Wrote table tex: {output_table_tex}")
    print(f"Wrote companion tex: {output_companion_tex}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
