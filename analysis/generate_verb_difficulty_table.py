"""
Generate per-verb difficulty table from full 60-verb evaluation pool.

Run from project root:
    uv run python analysis/generate_verb_difficulty_table.py

Outputs:
    results/figures/verb_difficulty_table.tex
    results/figures/verb_difficulty.csv
"""

import os
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
EVAL_CSV        = "results/aggregate/annotation_sampling_pool.csv"
OUT_TEX         = "results/figures/verb_difficulty_table.tex"
OUT_CSV         = "results/figures/verb_difficulty.csv"
EASY_THRESH     = 20.0   # GOLD% >= this  → Easy
MODERATE_THRESH = 10.0   # GOLD% >= this  → Moderate; below → Hard
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)

df = pd.read_csv(EVAL_CSV)

verb_stats = df.groupby("verb").agg(
    total  = ("tier", "count"),
    gold   = ("tier", lambda x: (x == "GOLD").sum()),
    silver = ("tier", lambda x: (x == "SILVER").sum()),
    bronze = ("tier", lambda x: (x == "BRONZE").sum()),
    fail   = ("tier", lambda x: (x == "FAIL").sum()),
).reset_index()

verb_stats["gold_pct"] = (verb_stats["gold"] / verb_stats["total"] * 100).round(1)
verb_stats = verb_stats.sort_values("gold_pct", ascending=False).reset_index(drop=True)


def label(pct):
    if pct >= EASY_THRESH:
        return "Easy"
    elif pct >= MODERATE_THRESH:
        return "Moderate"
    else:
        return "Hard"


verb_stats["difficulty"] = verb_stats["gold_pct"].apply(label)

# ── Print to console ──────────────────────────────────────────────────────────
print(f"\nPer-verb difficulty ({len(verb_stats)} verbs):\n")
print(f"{'Verb':<15} {'GOLD':>5} {'SILVER':>7} {'FAIL':>6} {'n':>5} {'GOLD%':>7}  {'Level':<10}")
print("-" * 60)
prev_label = None
for _, row in verb_stats.iterrows():
    if row["difficulty"] != prev_label:
        if prev_label is not None:
            print()
        prev_label = row["difficulty"]
    print(
        f"{row['verb']:<15} {int(row['gold']):>5} {int(row['silver']):>7} "
        f"{int(row['fail']):>6} {int(row['total']):>5} "
        f"{row['gold_pct']:>6.1f}%  {row['difficulty']:<10}"
    )

# ── LaTeX table ───────────────────────────────────────────────────────────────
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\resizebox{\textwidth}{!}{%")
lines.append(r"\begin{tabular}{l r r r r r l}")
lines.append(r"\toprule")
lines.append(
    r"\textbf{Verb} & \textbf{GOLD} & \textbf{SILVER} & "
    r"\textbf{FAIL} & \textbf{$n$} & \textbf{GOLD\%} & "
    r"\textbf{Difficulty} \\"
)
lines.append(r"\midrule")

prev_label = None
for _, row in verb_stats.iterrows():
    if row["difficulty"] != prev_label:
        if prev_label is not None:
            lines.append(r"\addlinespace[3pt]")
        prev_label = row["difficulty"]
    lines.append(
        f"\\textit{{{row['verb']}}} & {int(row['gold'])} & {int(row['silver'])} & "
        f"{int(row['fail'])} & {int(row['total'])} & {row['gold_pct']} & "
        f"{row['difficulty']} \\\\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}}")
lines.append(
    r"\caption{Per-verb GOLD rate across all models and methods "
    r"(60 verbs, sorted by GOLD\%). "
    r"Difficulty: Easy ($\geq$20\%), Moderate (10--19\%), Hard ($<$10\%).}"
)
lines.append(r"\label{tab:verb_difficulty}")
lines.append(r"\end{table*}")

with open(OUT_TEX, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nWrote LaTeX table to {OUT_TEX}")

# ── CSV ───────────────────────────────────────────────────────────────────────
verb_stats.to_csv(OUT_CSV, index=False)
print(f"Wrote CSV to {OUT_CSV}")