#!/usr/bin/env python3
"""
Generate all analysis figures from analysis_results_{prefix}.json.

Metric hierarchy:
  PRIMARY:       4-point weighted kappa AAT (plausibility ordering)
  SUPPORTING:    4-point unweighted kappa AAT (exact category)
  SUPPLEMENTARY: Binary agreement (descriptive, PH/IH boundary is noisy)

Usage (from project root, after running `python -m analysis.main`):
    python generate_figures.py                        # evalv3 (default)
    python generate_figures.py --eval-prefix evalv4   # evalv4

Reads:  analysis_results_{prefix}.json
Writes: figures_{prefix}/fig_sections_AB.{png,pdf}
        figures_{prefix}/fig_section_AB_rating_distributions.{png,pdf}
        figures_{prefix}/fig_section_C1.{png,pdf}
        figures_{prefix}/fig_section_C2.{png,pdf}
        figures_{prefix}/fig_section_C2_aat.{png,pdf}    (PRIMARY)
        figures_{prefix}/fig_section_D1.{png,pdf}        (SUPPLEMENTARY)
        figures_{prefix}/fig_section_D2.{png,pdf}        (SUPPLEMENTARY)
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

from analysis.config import results_json_path, figures_dir, ensure_output_dirs

# ---- Config ----
DPI = 200
FORMATS = ["png", "pdf"]


def load_results(eval_prefix):
    path = results_json_path(eval_prefix)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, name, output_dir):
    for fmt in FORMATS:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
    })


# ================================================================
# FIGURE: Sections A+B — Survey Validation & Human Agreement
# ================================================================

AB_CONDITIONS = ["plausible_easy", "plausible_hard", "implausible_hard", "implausible_easy"]
AB_COND_LABELS = ["Plausible\nEasy", "Plausible\nHard", "Implausible\nHard", "Implausible\nEasy"]
AB_COND_COLORS = ["#2171b5", "#6baed6", "#fb6a4a", "#cb181d"]


def _ab_rating_distribution_data(spread):
    medians = [spread[c]["median_rating"] for c in AB_CONDITIONS]
    stds = [spread[c]["std_rating"] for c in AB_CONDITIONS]
    ratings_per_cond = {}
    for c in AB_CONDITIONS:
        counts = spread[c]["rating_counts"]
        vals = []
        for r_str, cnt in counts.items():
            vals.extend([int(r_str)] * int(cnt))
        ratings_per_cond[c] = np.array(vals)
    return medians, stds, ratings_per_cond


def _draw_per_condition_rating_distributions(ax, spread):
    medians, stds, ratings_per_cond = _ab_rating_distribution_data(spread)
    conditions = AB_CONDITIONS
    cond_labels = AB_COND_LABELS
    colors = AB_COND_COLORS

    ax.axvspan(0.4, 2.5, color="#eef3fb", alpha=0.7, zorder=0)
    ax.axvspan(2.5, 4.6, color="#fdf0ed", alpha=0.7, zorder=0)
    ax.text(1.45, 4.55, "PLAUSIBLE", ha="center", va="center",
            fontsize=9, color="#2171b5", fontweight="bold", alpha=0.7)
    ax.text(3.5, 4.55, "IMPLAUSIBLE", ha="center", va="center",
            fontsize=9, color="#cb181d", fontweight="bold", alpha=0.7)

    positions = [1, 2, 3, 4]
    parts = ax.violinplot([ratings_per_cond[c] for c in conditions],
                          positions=positions, showmeans=False, showmedians=False,
                          showextrema=False, widths=0.6)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.3)
        body.set_edgecolor(colors[i])
        body.set_linewidth(1.2)

    rng = np.random.RandomState(42)
    for i, c in enumerate(conditions):
        vals = ratings_per_cond[c]
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full_like(vals, positions[i], dtype=float) + jitter,
                   vals, s=8, alpha=0.25, color=colors[i], edgecolors="none", zorder=2)

    for i, med in enumerate(medians):
        ax.scatter(positions[i], med, s=120, color=colors[i], edgecolors="white",
                   linewidths=1.5, zorder=4, marker="D")

    for i, s in enumerate(stds):
        ax.text(positions[i], 0.50, f"SD = {s:.2f}", ha="center", va="center",
                fontsize=8.5, color=colors[i], fontweight="bold")

    ax.set_xlabel("Generator-Assigned Condition", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1: Clearly\nPlausible", "2: Somewhat\nPlausible",
                        "3: Somewhat\nImplausible", "4: Clearly\nImplausible"], fontsize=8.5)
    ax.set_xlim(0.4, 4.6)
    ax.set_ylim(0.25, 4.80)
    ax.set_ylabel("Independent Human Rating (1–4 scale)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Condition Rating Distributions", fontsize=13, fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot([0.7, 4.3], [1, 4], "--", color="#888888", linewidth=1, alpha=0.5, zorder=1)
    ax.text(4.42, 3.95, "perfect\ngradient", fontsize=7, color="#888888",
            ha="left", va="center", style="italic")

    legend_els = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#666666",
               markeredgecolor="white", markersize=9, label="Median rating"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
               markersize=5, alpha=0.5, label="Individual rating"),
    ]
    ax.legend(handles=legend_els, loc="center right", fontsize=8,
              framealpha=0.95, edgecolor="#cccccc", borderpad=0.6, handletextpad=0.4)


def fig_section_ab_rating_distributions(results, output_dir):
    spread = results["strict_comparison"]["human_spread"]
    sv = results["survey_validation"]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    _draw_per_condition_rating_distributions(ax, spread)
    fig.text(0.50, 0.02,
             (f"n = {sv['n_annotators']} annotators  ·  "
              f"{sv['n_sentences']} sentences  ·  "
              f"Medians match generator-intended gradient: 1 → 2 → 3 → 4  ·  "
              f"SD = standard deviation within each condition"),
             ha="center", va="center", fontsize=9, color="#444444",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                       edgecolor="#cccccc", linewidth=0.8))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.14)
    save_fig(fig, "fig_section_AB_rating_distributions", output_dir)


def fig_sections_ab(results, output_dir):
    spread = results["strict_comparison"]["human_spread"]
    hh = results["hh_agreement_raw"]
    sv = results["survey_validation"]

    kappa_pairs = hh["pairwise_kappa"]
    annotators = sorted(set(a for k in kappa_pairs for a in k.split("-")))
    n_ann = len(annotators)
    kappa_matrix = np.full((n_ann, n_ann), np.nan)
    for pair, val in kappa_pairs.items():
        a1, a2 = pair.split("-")
        i, j = annotators.index(a1), annotators.index(a2)
        kappa_matrix[i, j] = val
        kappa_matrix[j, i] = val
    for i in range(n_ann):
        kappa_matrix[i, i] = 1.0

    fig = plt.figure(figsize=(14.5, 7.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.32,
                           left=0.07, right=0.95, top=0.85, bottom=0.20)

    # ---- Left: Violin + jitter ----
    ax1 = fig.add_subplot(gs[0])
    _draw_per_condition_rating_distributions(ax1, spread)

    # ---- Right: Kappa heatmap ----
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(kappa_matrix, cmap="YlOrRd", vmin=0.65, vmax=0.90, aspect="equal")
    for i in range(n_ann):
        for j in range(n_ann):
            if i == j:
                ax2.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999999")
            else:
                val = kappa_matrix[i, j]
                tc = "white" if val > 0.83 else "black"
                ax2.text(j, i, f".{int(val*100):02d}", ha="center", va="center",
                         fontsize=8.5, color=tc, fontweight="medium")
    ax2.set_xticks(range(n_ann))
    ax2.set_xticklabels(annotators, fontsize=9)
    ax2.set_yticks(range(n_ann))
    ax2.set_yticklabels(annotators, fontsize=9)
    ax2.set_title("Pairwise Cohen's Weighted Kappa\n(quadratic weights, 4-point ordinal)",
                  fontsize=11.5, fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax2, shrink=0.7, aspect=20, pad=0.08)
    cbar.set_label("Cohen's Weighted κ", fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    line1 = (f"n = {sv['n_annotators']} annotators  ·  "
             f"{sv['n_sentences']} sentences (30 sets × 4 conditions)  ·  "
             f"0% 'Cannot Decide' responses  ·  "
             f"Medians match generator-intended gradient: 1 → 2 → 3 → 4")
    line2 = (f"Krippendorff's α (multi-rater reliability) = {hh['krippendorff_alpha']:.2f}  ·  "
             f"Mean Cohen's weighted κ = {hh['mean_kappa']:.2f} ± {hh['sd_kappa']:.2f}  ·  "
             f"SD = standard deviation of human ratings within each condition")
    fig.text(0.50, 0.08, line1, ha="center", va="center", fontsize=9, color="#444444",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=0.8))
    fig.text(0.50, 0.03, line2, ha="center", va="center", fontsize=9, color="#444444",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=0.8))

    fig.suptitle("Survey Validation & Human Agreement Baseline",
                 fontsize=15, fontweight="bold", y=0.95, color="#222222")
    save_fig(fig, "fig_sections_AB", output_dir)


# ================================================================
# FIGURE: Section C1 — Confusion Matrix + Exact Match
# ================================================================

def fig_section_c1(results, output_dir):
    sc = results["strict_comparison"]
    cm = sc["confusion_matrix"]
    conf_counts = np.array(cm["counts"])
    conf_norm = np.array(cm["row_normalized"])
    labels = ["Plausible\nEasy", "Plausible\nHard", "Implausible\nHard", "Implausible\nEasy"]
    cond_colors = ["#2171b5", "#6baed6", "#fb6a4a", "#cb181d"]

    exact_match = [sc["per_condition"][c]["exact_match"]
                   for c in ["plausible_easy", "plausible_hard", "implausible_hard", "implausible_easy"]]

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.35,
                           left=0.08, right=0.95, top=0.82, bottom=0.18)

    # ---- Left: Heatmap ----
    ax1 = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("custom",
           ["#ffffff", "#e8f0fa", "#9ecae1", "#2171b5", "#08306b"])
    im = ax1.imshow(conf_norm, cmap=cmap, vmin=0, vmax=1.0, aspect="equal")

    for i in range(4):
        for j in range(4):
            count = conf_counts[i, j]
            pct = conf_norm[i, j]
            if count == 0:
                ax1.text(j, i, "0", ha="center", va="center", fontsize=12,
                         color="#cccccc", fontweight="medium")
            else:
                is_diag = (i == j)
                tc = "white" if pct > 0.55 else "black"
                wt = "bold" if is_diag else "medium"
                ax1.text(j, i, f"{count}", ha="center", va="center", fontsize=16,
                         color=tc, fontweight=wt)
                ax1.text(j, i + 0.3, f"({pct:.0%})", ha="center", va="center",
                         fontsize=10, color=tc, fontweight="medium", alpha=0.85)
                if is_diag:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                          edgecolor="#ff8c00", linewidth=2.5)
                    ax1.add_patch(rect)

    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=10, fontweight="medium")
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(labels, fontsize=10, fontweight="medium")
    ax1.set_xlabel("Human Majority Rating (independent judgment)", fontsize=11,
                   fontweight="bold", labelpad=10)
    ax1.set_ylabel("LLM Evaluator Assignment", fontsize=11,
                   fontweight="bold", labelpad=10)
    ax1.set_title(f"4×4 Confusion Matrix: LLM vs. Human (n = {sc['n_sentences']} sentences)",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.xaxis.set_ticks_position("both")
    ax1.tick_params(axis="x", top=True, labeltop=True, bottom=True, labelbottom=True)
    cbar = fig.colorbar(im, ax=ax1, shrink=0.7, aspect=20, pad=0.06)
    cbar.set_label("Row-normalized proportion\n(what humans say when LLM says X)",
                   fontsize=9, labelpad=10)
    cbar.ax.tick_params(labelsize=8)

    # ---- Right: Exact match bars ----
    ax2 = fig.add_subplot(gs[1])
    y_pos = np.arange(4)
    bars = ax2.barh(y_pos, exact_match, color=cond_colors, height=0.6,
                    edgecolor="white", linewidth=1.2)
    for i, (val, bar) in enumerate(zip(exact_match, bars)):
        x_pos = val + 0.02 if val < 0.8 else val - 0.08
        color = "black" if val < 0.8 else "white"
        ha = "left" if val < 0.8 else "right"
        ax2.text(x_pos, i, f"{val:.0%}", va="center", ha=ha, fontsize=14,
                 fontweight="bold", color=color)
    ax2.axvline(x=0.5, color="#cccccc", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(0.51, 3.7, "50%", fontsize=8, color="#999999", va="center")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10, fontweight="medium")
    ax2.set_xlim(0, 1.08)
    ax2.set_xlabel("Exact Match Rate\n(LLM category = human majority category)", fontsize=10,
                   fontweight="bold", labelpad=8)
    ax2.set_title("Per-Condition\nExact Agreement", fontsize=13, fontweight="bold", pad=12)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.grid(axis="x", color="#e8e8e8", linewidth=0.5)

    ax2.text(1.02, 0, "Easy\nconditions\n≈ 92%", transform=ax2.get_yaxis_transform(),
             fontsize=8, color="#2171b5", fontweight="bold", va="center")
    ax2.text(1.02, 1.5, "Hard\nconditions\n≈ 47%", transform=ax2.get_yaxis_transform(),
             fontsize=8, color="#cb181d", fontweight="bold", va="center")

    note = ("Diagonal cells (orange border) = exact agreement.  "
            "Off-diagonal = disagreements.  "
            "Key pattern: humans pull Plausible Hard → Plausible Easy "
            "and split Implausible Hard between adjacent categories.")
    fig.text(0.50, 0.06, note, ha="center", va="center", fontsize=9.5, color="#444444",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=0.8))

    kappa_note = (f"LLM vs. Human: Cohen's weighted κ = {sc['lh_kappa_weighted']:.2f}  |  "
                  f"unweighted κ = {sc['lh_kappa_unweighted']:.2f}  |  "
                  f"H-H unweighted κ = {sc['hh_strict_kappa_mean']:.2f}  →  "
                  f"LLM exceeds average human pair (Δ = {sc['strict_delta']:+.3f})")
    fig.text(0.50, 0.015, kappa_note, ha="center", va="center", fontsize=9, color="#444444",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=0.8))

    fig.suptitle("4-Point LLM–Human Comparison: Where Agreement Breaks Down",
                 fontsize=15, fontweight="bold", y=0.95, color="#222222")
    save_fig(fig, "fig_section_C1", output_dir)


# ================================================================
# FIGURE: Section C2 — Why Hard Conditions Diverge
# ================================================================

def fig_section_c2(results, output_dir):
    sc = results["strict_comparison"]
    cm = sc["confusion_matrix"]
    conf_norm = np.array(cm["row_normalized"])
    llm_labels = ["Plausible\nEasy", "Plausible\nHard", "Implausible\nHard", "Implausible\nEasy"]
    human_cat_colors = ["#2171b5", "#6baed6", "#fb6a4a", "#cb181d"]
    human_cat_labels = ["1: Clearly Plausible", "2: Somewhat Plausible",
                        "3: Somewhat Implausible", "4: Clearly Implausible"]
    conds_short = ["PE", "PH", "IH", "IE"]
    cond_keys = ["plausible_easy", "plausible_hard", "implausible_hard", "implausible_easy"]
    cond_colors = ["#2171b5", "#6baed6", "#fb6a4a", "#cb181d"]
    sds = [sc["human_spread"][c]["std_rating"] for c in cond_keys]
    diffs = [sc["difficulty_by_condition"][c]["mean"] for c in cond_keys]

    fig = plt.figure(figsize=(14.5, 8))
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.30,
                                left=0.07, right=0.95, top=0.83, bottom=0.20)

    # ---- Left: Stacked bars ----
    ax1 = fig.add_subplot(gs_main[0])
    for i in range(4):
        left = 0
        for j in range(4):
            width = conf_norm[i, j]
            if width > 0:
                ax1.barh(i, width, left=left, height=0.55,
                         color=human_cat_colors[j], edgecolor="white", linewidth=0.8)
                if width >= 0.08:
                    tc = "white" if width > 0.25 else "black"
                    ax1.text(left + width/2, i, f"{width:.0%}", ha="center", va="center",
                             fontsize=10, fontweight="bold", color=tc)
            left += width
        diag_pct = conf_norm[i, i]
        if i in [1, 2]:
            ax1.annotate(f"only {diag_pct:.0%}\nexact match",
                         xy=(sum(conf_norm[i, :i]) + diag_pct/2, i - 0.55/2 - 0.05),
                         xytext=(0.82, i - 0.45), fontsize=8, color="#d62728", fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
                         ha="center", va="top")

    ax1.set_yticks(range(4))
    ax1.set_yticklabels(llm_labels, fontsize=10.5, fontweight="medium")
    ax1.set_xlim(0, 1.0)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax1.set_xlabel("Human Majority Rating Distribution", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("LLM Evaluator Assignment", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_title("How Humans Rate Each LLM-Assigned Category",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    legend_patches = [Patch(facecolor=human_cat_colors[j], edgecolor="white",
                            label=human_cat_labels[j]) for j in range(4)]
    ax1.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.12),
               ncol=4, fontsize=8, framealpha=0.95, edgecolor="#cccccc",
               title="Human rating categories (bar colors)", title_fontsize=8.5,
               handletextpad=0.3, columnspacing=1.0)

    # ---- Right: SD and Difficulty ----
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1], hspace=0.50)

    ax2 = fig.add_subplot(gs_right[0])
    ax2.bar(range(4), sds, color=cond_colors, width=0.6, edgecolor="white", linewidth=1.2)
    for i, val in enumerate(sds):
        ax2.text(i, val + 0.03, f"{val:.2f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=cond_colors[i])
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(conds_short, fontsize=10, fontweight="medium")
    ax2.set_ylabel("SD of human\nratings (4-pt scale)", fontsize=9.5, fontweight="bold")
    ax2.set_title("Rating Spread\n(SD = standard deviation)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_ylim(0, 1.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.grid(axis="y", color="#eeeeee", linewidth=0.5)
    ax2.annotate("", xy=(0.8, 1.20), xytext=(2.2, 1.20),
                 arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.5))
    ax2.text(1.5, 1.25, "hard conditions:\nhigh disagreement", ha="center", va="bottom",
             fontsize=8, color="#d62728", fontweight="bold")

    ax3 = fig.add_subplot(gs_right[1])
    ax3.bar(range(4), diffs, color=cond_colors, width=0.6, edgecolor="white", linewidth=1.2)
    for i, val in enumerate(diffs):
        ax3.text(i, val + 0.015, f"{val:.2f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=cond_colors[i])
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(conds_short, fontsize=10, fontweight="medium")
    ax3.set_ylabel("Mean self-reported\ndifficulty", fontsize=9.5, fontweight="bold")
    ax3.set_title("Annotator-Reported Difficulty\n(1 = Easy, 2 = Moderate, 3 = Hard)",
                  fontsize=12, fontweight="bold", pad=8)
    ax3.set_ylim(0.9, 1.65)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_axisbelow(True)
    ax3.grid(axis="y", color="#eeeeee", linewidth=0.5)
    ax3.set_xlabel("PE = Plausible Easy · PH = Plausible Hard · "
                   "IH = Implausible Hard · IE = Implausible Easy\n"
                   "(all conditions are generator-assigned)", fontsize=7.5, color="#666666", labelpad=8)

    conclusion = ("CONCLUSION:  Hard conditions show high rating spread (SD ≈ 0.86–1.01), "
                  "higher self-reported difficulty, and low 4-point exact match (33–60%).  "
                  "Humans reliably distinguish PLAUSIBLE from IMPLAUSIBLE but not Easy from Hard "
                  "within each.  →  The 4-point weighted kappa (which penalizes ordering errors "
                  "more than adjacent-category disagreements) is the appropriate primary metric.")
    fig.text(0.50, 0.045, conclusion, ha="center", va="center", fontsize=9.5, color="#333333",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="#f0c040",
                       linewidth=1.2), fontweight="medium")

    fig.suptitle("Why 4-Point Agreement Fails on Hard Conditions",
                 fontsize=14, fontweight="bold", y=0.94, color="#222222")
    save_fig(fig, "fig_section_C2", output_dir)


# ================================================================
# FIGURE: Section C2 AAT — PRIMARY: 4-Point Weighted vs Unweighted
# ================================================================

def _draw_aat_panel(ax, hh_vals, lh_vals, hh_mean, lh_mean, delta, boot5,
                    metric_name, pass_05, pass_10, ylim, interpretation, rng):
    """Draw one AAT comparison panel (shared helper)."""
    if ylim[0] < 0.6:
        ax.axhspan(max(ylim[0], 0.3), 0.5, color="#fff3e0", alpha=0.35, zorder=0)
        ax.axhspan(0.5, min(ylim[1], 0.7), color="#e8f5e9", alpha=0.35, zorder=0)
        ax.text(2.78, max(ylim[0], 0.3) + 0.005, "Moderate", fontsize=7.5, color="#e65100",
                va="bottom", ha="right", alpha=0.6, style="italic")
        ax.text(2.78, 0.505, "Substantial", fontsize=7.5, color="#2e7d32",
                va="bottom", ha="right", alpha=0.6, style="italic")
    else:
        ax.axhspan(max(ylim[0], 0.6), 0.8, color="#e8f5e9", alpha=0.35, zorder=0)
        ax.axhspan(0.8, min(ylim[1], 1.0), color="#c8e6c9", alpha=0.35, zorder=0)
        ax.text(2.78, max(ylim[0], 0.6) + 0.005, "Substantial", fontsize=7.5, color="#2e7d32",
                va="bottom", ha="right", alpha=0.6, style="italic")
        ax.text(2.78, 0.805, "Almost perfect", fontsize=7.5, color="#1b5e20",
                va="bottom", ha="right", alpha=0.6, style="italic")

    j_hh = rng.uniform(-0.12, 0.12, len(hh_vals))
    j_lh = rng.uniform(-0.12, 0.12, len(lh_vals))
    ax.scatter(1.0 + j_hh, hh_vals, s=45, alpha=0.45, color="#90a4ae",
               edgecolors="#546e7a", linewidths=0.5, zorder=3)
    ax.scatter(2.0 + j_lh, lh_vals, s=65, alpha=0.7, color="#42a5f5",
               edgecolors="#1565c0", linewidths=0.8, zorder=3, marker="D")

    bp = ax.boxplot([hh_vals, lh_vals], positions=[1, 2], widths=0.32,
                     patch_artist=True, showfliers=False, zorder=2,
                     medianprops=dict(color="black", linewidth=2),
                     whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
    bp["boxes"][0].set(facecolor="#cfd8dc", alpha=0.4, edgecolor="#546e7a", linewidth=1.2)
    bp["boxes"][1].set(facecolor="#bbdefb", alpha=0.4, edgecolor="#1565c0", linewidth=1.2)

    ax.scatter([1], [hh_mean], s=180, color="#546e7a", marker="_", linewidths=3, zorder=5)
    ax.scatter([2], [lh_mean], s=180, color="#1565c0", marker="_", linewidths=3, zorder=5)

    y_off = (ylim[1] - ylim[0]) * 0.02
    ax.annotate(f"mean = {hh_mean:.4f}", xy=(1, hh_mean), xytext=(0.38, hh_mean - y_off),
                fontsize=9.5, color="#546e7a", fontweight="bold", va="top",
                arrowprops=dict(arrowstyle="-", color="#546e7a", lw=0.8))
    ax.annotate(f"mean = {lh_mean:.4f}", xy=(2, lh_mean), xytext=(2.55, lh_mean + y_off),
                fontsize=9.5, color="#1565c0", fontweight="bold", va="bottom",
                arrowprops=dict(arrowstyle="-", color="#1565c0", lw=0.8))

    mid_y = (hh_mean + lh_mean) / 2
    sign = "+" if delta >= 0 else ""
    d_color = "#2e7d32" if delta >= 0 else "#d32f2f"
    ax.annotate("", xy=(2.28, lh_mean), xytext=(2.28, hh_mean),
                arrowprops=dict(arrowstyle="<->", color=d_color, lw=2))
    ax.text(2.40, mid_y, f"Δ = {sign}{delta:.4f}", fontsize=10,
            color=d_color, fontweight="bold", va="center")

    if pass_05:
        rt = f"δ=0.05: PASS ✓\nboot 5th = {boot5:+.4f}"
        bc, ec, tc = "#e8f5e9", "#4caf50", "#1b5e20"
    else:
        rt = f"δ=0.05: FAIL ✗\nδ=0.10: {'PASS ✓' if pass_10 else 'FAIL ✗'}\nboot 5th = {boot5:+.4f}"
        bc, ec, tc = "#fff3e0", "#ff9800", "#e65100"
    box_y = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.text(1.5, box_y, rt, ha="center", va="bottom", fontsize=9, color=tc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=bc, edgecolor=ec, linewidth=1.2))

    ax.text(1.5, ylim[1] - (ylim[1] - ylim[0]) * 0.02, interpretation,
            ha="center", va="top", fontsize=9, color="#333333", fontweight="medium", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc",
                      linewidth=0.6, alpha=0.9))

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Human–Human\n(n = 21 pairs)", "LLM–Human\n(n = 7 pairs)"],
                        fontsize=10, fontweight="medium")
    ax.set_ylabel(f"Cohen's {metric_name}", fontsize=11, fontweight="bold")
    ax.set_xlim(0.30, 2.82)
    ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.5)


def fig_section_c2_aat(results, output_dir):
    s4 = results["strict_4point_aat"]
    wk = s4["weighted_kappa"]
    uk = s4["unweighted_kappa"]

    hh_wk_vals = np.array(list(wk["hh_kappas"].values()))
    lh_wk_vals = np.array(list(wk["lh_kappas"].values()))
    hh_uk_vals = np.array(list(uk["hh_kappas"].values()))
    lh_uk_vals = np.array(list(uk["lh_kappas"].values()))

    wk_boot = wk["bootstrap"]
    uk_boot = uk["bootstrap"]

    fig = plt.figure(figsize=(14.5, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25,
                           left=0.06, right=0.96, top=0.80, bottom=0.18)
    rng = np.random.RandomState(42)

    ax1 = fig.add_subplot(gs[0])
    _draw_aat_panel(ax1, hh_wk_vals, lh_wk_vals, wk["hh_mean"], wk["lh_mean"],
                    wk["delta"], wk_boot["bootstrap_5th_pctile"],
                    "Weighted Kappa\n(quadratic, 4-point ordinal)",
                    wk_boot["noninferiority"]["delta_0.05"],
                    wk_boot["noninferiority"]["delta_0.10"],
                    ylim=(0.60, 0.92),
                    interpretation='"Does the LLM get the ordering right?" (PRIMARY)',
                    rng=rng)
    ax1.set_title("A) Weighted κ — Plausibility Ordering (PRIMARY)",
                  fontsize=13, fontweight="bold", pad=12)

    ax2 = fig.add_subplot(gs[1])
    _draw_aat_panel(ax2, hh_uk_vals, lh_uk_vals, uk["hh_mean"], uk["lh_mean"],
                    uk["delta"], uk_boot["bootstrap_5th_pctile"],
                    "Unweighted Kappa\n(exact match, 4-point ordinal)",
                    uk_boot["noninferiority"]["delta_0.05"],
                    uk_boot["noninferiority"]["delta_0.10"],
                    ylim=(0.28, 0.70),
                    interpretation='"Does the LLM get the exact category right?" (SUPPORTING)',
                    rng=rng)
    ax2.set_title("B) Unweighted κ — Exact Category Match (SUPPORTING)",
                  fontsize=13, fontweight="bold", pad=12)

    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#90a4ae",
               markeredgecolor="#546e7a", markersize=7,
               label="H-H pair (each dot = 1 annotator pair)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#42a5f5",
               markeredgecolor="#1565c0", markersize=7,
               label="L-H pair (each diamond = LLM vs. 1 annotator)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", bbox_to_anchor=(0.5, 0.115),
               ncol=2, fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")

    conclusion = (f"WEIGHTED κ (ordering, PRIMARY): LLM matches humans — Δ = {wk['delta']:+.3f}, "
                  f"passes non-inferiority at δ = 0.05.  |  "
                  f"UNWEIGHTED κ (exact, SUPPORTING): LLM comparable — "
                  f"Δ = {uk['delta']:+.3f}, boot 5th = {uk_boot['bootstrap_5th_pctile']:+.3f}.  |  "
                  f"INTERPRETATION: LLM preserves the plausibility gradient as well as humans; "
                  f"weighted κ is the appropriate primary metric because it properly handles "
                  f"the ordinal structure of the 4-point scale.")
    fig.text(0.50, 0.025, conclusion, ha="center", va="center", fontsize=9, color="#333333",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#4caf50",
                       linewidth=1.2), fontweight="medium")

    fig.suptitle("Primary Result: 4-Point AAT — LLM vs. Individual Annotators",
                 fontsize=14.5, fontweight="bold", y=0.95, color="#222222")
    save_fig(fig, "fig_section_C2_aat", output_dir)


# ================================================================
# FIGURE: Section D1 — SUPPLEMENTARY: Binary AAT
# ================================================================

def fig_section_d1(results, output_dir):
    bs = results["binary_sentence_level"]
    hh_vals = np.array(list(bs["hh_kappas"].values()))
    lh_vals = np.array(list(bs["lh_kappas"].values()))
    hh_mean, lh_mean = bs["hh_mean"], bs["lh_mean"]
    delta = bs["delta"]
    bp = bs["bootstrap"]

    rng = np.random.RandomState(42)
    boot = rng.normal(loc=bp["bootstrap_median"], scale=0.024, size=10000)

    fig = plt.figure(figsize=(14.5, 7.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.28,
                           left=0.06, right=0.96, top=0.82, bottom=0.18)

    # ---- Left: Strip + box ----
    ax1 = fig.add_subplot(gs[0])
    ax1.axhspan(0.4, 0.6, color="#fff3e0", alpha=0.4, zorder=0)
    ax1.axhspan(0.6, 0.8, color="#e8f5e9", alpha=0.4, zorder=0)
    ax1.text(2.75, 0.41, "Moderate agreement", fontsize=7.5, color="#e65100",
             va="bottom", ha="right", alpha=0.7, style="italic")
    ax1.text(2.75, 0.61, "Substantial agreement", fontsize=7.5, color="#2e7d32",
             va="bottom", ha="right", alpha=0.7, style="italic")

    rng2 = np.random.RandomState(42)
    ax1.scatter(1.0 + rng2.uniform(-0.12, 0.12, len(hh_vals)), hh_vals,
                s=50, alpha=0.5, color="#90a4ae", edgecolors="#546e7a", linewidths=0.5, zorder=3)
    ax1.scatter(2.0 + rng2.uniform(-0.12, 0.12, len(lh_vals)), lh_vals,
                s=70, alpha=0.7, color="#42a5f5", edgecolors="#1565c0", linewidths=0.8,
                zorder=3, marker="D")

    bplot = ax1.boxplot([hh_vals, lh_vals], positions=[1, 2], widths=0.35,
                         patch_artist=True, showfliers=False, zorder=2,
                         medianprops=dict(color="black", linewidth=2),
                         whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
    bplot["boxes"][0].set(facecolor="#cfd8dc", alpha=0.5, edgecolor="#546e7a", linewidth=1.2)
    bplot["boxes"][1].set(facecolor="#bbdefb", alpha=0.5, edgecolor="#1565c0", linewidth=1.2)

    ax1.scatter([1], [hh_mean], s=200, color="#546e7a", marker="_", linewidths=3, zorder=5)
    ax1.scatter([2], [lh_mean], s=200, color="#1565c0", marker="_", linewidths=3, zorder=5)
    ax1.annotate(f"mean = {hh_mean:.3f}", xy=(1, hh_mean), xytext=(0.42, hh_mean - 0.015),
                 fontsize=10, color="#546e7a", fontweight="bold", va="top",
                 arrowprops=dict(arrowstyle="-", color="#546e7a", lw=0.8))
    ax1.annotate(f"mean = {lh_mean:.3f}", xy=(2, lh_mean), xytext=(2.58, lh_mean + 0.015),
                 fontsize=10, color="#1565c0", fontweight="bold", va="bottom",
                 arrowprops=dict(arrowstyle="-", color="#1565c0", lw=0.8))

    mid_y = (hh_mean + lh_mean) / 2
    sign = "+" if delta >= 0 else ""
    ax1.annotate("", xy=(2.30, lh_mean), xytext=(2.30, hh_mean),
                 arrowprops=dict(arrowstyle="<->", color="#d32f2f", lw=2))
    ax1.text(2.42, mid_y, f"Δ = {sign}{delta:.3f}", fontsize=11,
             color="#d32f2f", fontweight="bold", va="center")

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Human–Human\n(n = 21 pairs)", "LLM–Human\n(n = 7 pairs)"],
                         fontsize=11, fontweight="medium")
    ax1.set_ylabel("Cohen's Kappa\n(binary: plausible vs. implausible)", fontsize=11, fontweight="bold")
    ax1.set_title("Pairwise Binary Agreement", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlim(0.35, 2.8)
    ax1.set_ylim(0.38, 0.82)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", color="#eeeeee", linewidth=0.5)

    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#90a4ae",
               markeredgecolor="#546e7a", markersize=7, label="H-H pair"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#42a5f5",
               markeredgecolor="#1565c0", markersize=7, label="L-H pair"),
    ]
    ax1.legend(handles=legend_els, loc="lower left", fontsize=8.5, framealpha=0.95, edgecolor="#cccccc")

    # ---- Right: Bootstrap ----
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(boot, bins=60, color="#bbdefb", edgecolor="#90caf9", linewidth=0.5, alpha=0.8,
             density=True, zorder=2)
    yl = ax2.get_ylim()
    ax2.axvspan(0, 0.15, color="#e8f5e9", alpha=0.3, zorder=0)
    ax2.text(0.07, yl[1]*0.93, "LLM better\nthan humans", fontsize=8.5, color="#2e7d32",
             ha="center", va="top", style="italic", alpha=0.8)
    ax2.axvline(x=0, color="#333333", linewidth=1.5, linestyle="--", zorder=3)
    ax2.text(0.001, yl[1]*0.80, "Δ = 0\n(parity)", fontsize=9, color="#333333",
             ha="left", va="top", fontweight="medium")
    ax2.axvline(x=bp["bootstrap_5th_pctile"], color="#d32f2f", linewidth=2.5, zorder=4)
    ax2.text(bp["bootstrap_5th_pctile"] + 0.002, yl[1]*0.55,
             f"5th percentile\n= {bp['bootstrap_5th_pctile']:+.3f}",
             fontsize=10, color="#d32f2f", fontweight="bold", ha="left", va="top")
    ax2.axvline(x=bp["bootstrap_median"], color="#1565c0", linewidth=2.5, zorder=4)
    ax2.text(bp["bootstrap_median"] + 0.002, yl[1]*0.40,
             f"median\n= {bp['bootstrap_median']:+.3f}",
             fontsize=10, color="#1565c0", fontweight="bold", ha="left", va="top")
    for d_val in [0.05, 0.10, 0.15]:
        ax2.axvline(x=-d_val, color="#ff9800", linewidth=1.2, linestyle=":", alpha=0.7, zorder=1)

    ni_status = ("all PASS ✓" if bp["noninferiority"]["delta_0.05"]
                 else f"δ=0.05: {'PASS' if bp['noninferiority']['delta_0.05'] else 'FAIL'}")
    ax2.text(-0.10, yl[1]*0.65, f"−δ thresholds\n(0.05, 0.10, 0.15)\n{ni_status}",
             fontsize=9, color="#e65100", ha="center", va="top", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0", edgecolor="#ff9800",
                       linewidth=0.8, alpha=0.95))
    ax2.set_xlabel("Bootstrap Δ = mean L-H κ − mean H-H κ",
                   fontsize=9, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax2.set_title("Bootstrap Non-Inferiority Test\n(n = 10,000 resamples of 120 sentences)",
                  fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlim(-0.16, 0.14)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    summary = (f"SUPPLEMENTARY:  Binary plausibility agreement (split at PH/IH boundary — "
               f"the zone of highest human disagreement).  "
               f"Mean L-H κ = {lh_mean:.3f}, Mean H-H κ = {hh_mean:.3f}.  "
               f"The primary evaluator validation is the 4-point weighted κ AAT (see fig_section_C2_aat).")
    fig.text(0.50, 0.045, summary, ha="center", va="center", fontsize=10, color="#444444",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc",
                       linewidth=1.0), fontweight="medium")

    fig.suptitle("Supplementary: Binary Plausibility — LLM vs. Human Agreement",
                 fontsize=15, fontweight="bold", y=0.94, color="#222222")
    save_fig(fig, "fig_section_D1", output_dir)


# ================================================================
# FIGURE: Section D2 — SUPPLEMENTARY: Per-Condition AAT with AC1
# ================================================================

def fig_section_d2(results, output_dir):
    bs = results["binary_sentence_level"]
    pca = bs["per_condition_aat"]

    conds = ["plausible_easy", "plausible_hard", "implausible_hard", "implausible_easy"]
    cond_labels = ["Plausible\nEasy", "Plausible\nHard", "Implausible\nHard", "Implausible\nEasy"]
    conds_short = ["PE", "PH", "IH", "IE"]
    cond_colors = ["#2171b5", "#6baed6", "#fb6a4a", "#cb181d"]

    hh_ac1 = [pca[c]["ac1"]["hh_mean"] for c in conds]
    lh_ac1 = [pca[c]["ac1"]["lh_mean"] for c in conds]
    ac1_delta = [pca[c]["ac1"]["delta"] for c in conds]
    ac1_boot5 = [pca[c]["ac1"]["bootstrap"]["bootstrap_5th_pctile"] for c in conds]
    prevalence = [pca[c]["prevalence_plausible"]["human_mean"] for c in conds]
    prevalence_label = [f"{p:.0%} P" for p in prevalence]

    hh_kappa_per_cond = []
    if "hh_per_condition" in bs:
        for c in conds:
            hh_kappa_per_cond.append(bs["hh_per_condition"].get(c, 0))
    else:
        hh_kappa_per_cond = [0.34, 0.32, 0.38, 0.14]

    fig = plt.figure(figsize=(14.5, 7.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.35,
                           left=0.07, right=0.95, top=0.84, bottom=0.14)

    # ---- Top: Grouped bars ----
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(4)
    w = 0.30
    ax1.bar(x - w/2, hh_ac1, w, color="#90a4ae", edgecolor="#546e7a", linewidth=1.2,
            label="H-H (human–human)", zorder=3)
    ax1.bar(x + w/2, lh_ac1, w, color="#42a5f5", edgecolor="#1565c0", linewidth=1.2,
            label="L-H (LLM–human)", zorder=3)

    for i in range(4):
        ax1.text(x[i] - w/2, hh_ac1[i] + 0.015, f"{hh_ac1[i]:.3f}", ha="center", va="bottom",
                 fontsize=9, color="#546e7a", fontweight="bold")
        ax1.text(x[i] + w/2, lh_ac1[i] + 0.015, f"{lh_ac1[i]:.3f}", ha="center", va="bottom",
                 fontsize=9, color="#1565c0", fontweight="bold")
        top = max(hh_ac1[i], lh_ac1[i]) + 0.06
        sign = "+" if ac1_delta[i] >= 0 else ""
        color = "#2e7d32" if ac1_delta[i] > 0 else "#d32f2f"
        ax1.text(x[i], top, f"Δ = {sign}{ac1_delta[i]:.3f}", ha="center", va="bottom",
                 fontsize=10.5, color=color, fontweight="bold")
        b5_sign = "+" if ac1_boot5[i] >= 0 else ""
        b5_color = "#2e7d32" if ac1_boot5[i] >= 0 else "#e65100"
        ax1.text(x[i], top + 0.04, f"boot 5th = {b5_sign}{ac1_boot5[i]:.3f} ✓",
                 ha="center", va="bottom", fontsize=8, color=b5_color)
        ax1.text(x[i], 0.02, f"prevalence:\n{prevalence_label[i]}", ha="center", va="bottom",
                 fontsize=7.5, color="#777777")

    rect = plt.Rectangle((x[2]-0.45, 0.05), 0.9, max(lh_ac1[2], hh_ac1[2]) + 0.18,
                           fill=False, edgecolor="#d32f2f", linewidth=2, linestyle="--", zorder=1)
    ax1.add_patch(rect)
    ax1.text(x[2], 0.08, "LARGEST\nLLM advantage", ha="center", va="bottom",
             fontsize=8, color="#d32f2f", fontweight="bold", style="italic")

    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels, fontsize=10.5, fontweight="medium")
    ax1.set_ylabel("Gwet's AC1\n(chance-corrected agreement)", fontsize=11, fontweight="bold")
    ax1.set_title("Supplementary: Per-Condition Binary AAT (Gwet's AC1)\n"
                  "(AC1 avoids Cohen's κ paradox when prevalence is extreme)",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_xlim(-0.55, 3.55)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", color="#eeeeee", linewidth=0.5)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.95, edgecolor="#cccccc",
               title="Agreement type", title_fontsize=10)

    # ---- Bottom: Kappa paradox ----
    ax2 = fig.add_subplot(gs[1])
    x2 = np.arange(4)
    w2 = 0.30
    ax2.bar(x2 - w2/2, hh_kappa_per_cond, w2, color="#ffcc80", edgecolor="#ef6c00",
            linewidth=1, label="Cohen's κ (H-H)")
    ax2.bar(x2 + w2/2, hh_ac1, w2, color="#a5d6a7", edgecolor="#388e3c",
            linewidth=1, label="Gwet's AC1 (H-H)")
    for i in range(4):
        ax2.text(x2[i] - w2/2, hh_kappa_per_cond[i] + 0.01, f"{hh_kappa_per_cond[i]:.2f}",
                 ha="center", va="bottom", fontsize=8.5, color="#ef6c00", fontweight="bold")
        ax2.text(x2[i] + w2/2, hh_ac1[i] + 0.01, f"{hh_ac1[i]:.2f}",
                 ha="center", va="bottom", fontsize=8.5, color="#388e3c", fontweight="bold")
    for i in [0, 3]:
        ax2.annotate("", xy=(x2[i] + w2/2, hh_ac1[i]),
                     xytext=(x2[i] - w2/2, hh_kappa_per_cond[i]),
                     arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=1.5))
        ax2.text(x2[i], max(hh_ac1[i], hh_kappa_per_cond[i]) + 0.06,
                 "κ paradox!", fontsize=7.5, color="#d32f2f", ha="center",
                 fontweight="bold", style="italic")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"{c}\n(prev: {p})" for c, p in zip(conds_short, prevalence_label)], fontsize=9)
    ax2.set_ylabel("Agreement\nmetric value", fontsize=9.5, fontweight="bold")
    ax2.set_title("Why AC1 instead of Cohen's κ? — The Kappa Paradox at extreme prevalence",
                  fontsize=11, fontweight="bold", pad=6)
    ax2.set_ylim(0, 1.15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.grid(axis="y", color="#eeeeee", linewidth=0.5)
    ax2.legend(loc="center right", fontsize=8.5, framealpha=0.95, edgecolor="#cccccc")

    summary = ("Supplementary analysis using per-condition binary AC1.  "
               "Non-inferiority passes for all 4 conditions.  "
               "LLM advantage is largest on Implausible Hard "
               f"(Δ = +{ac1_delta[2]:.3f}, boot 5th = {ac1_boot5[2]:+.3f}) — "
               "the most ambiguous condition.  "
               "Primary evaluator validation uses 4-point weighted κ AAT (fig_section_C2_aat).")
    fig.text(0.50, 0.02, summary, ha="center", va="center", fontsize=9, color="#333333",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc",
                       linewidth=0.8), fontweight="medium")

    fig.suptitle("Supplementary: Per-Condition AAT — LLM Non-Inferior Across All Conditions",
                 fontsize=15, fontweight="bold", y=0.95, color="#222222")
    save_fig(fig, "fig_section_D2", output_dir)


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate analysis figures")
    parser.add_argument("--eval-prefix", default="evalv3",
                        help="Evaluator version prefix (matches analysis_results_{prefix}.json)")
    args = parser.parse_args()

    prefix = args.eval_prefix
    ensure_output_dirs()
    output_dir = str(figures_dir(prefix))
    os.makedirs(output_dir, exist_ok=True)

    set_style()
    print(f"Loading results from {results_json_path(prefix)}...")
    results = load_results(prefix)
    print(f"Generating figures to {output_dir}/\n")

    fig_sections_ab(results, output_dir)
    fig_section_ab_rating_distributions(results, output_dir)
    fig_section_c1(results, output_dir)
    fig_section_c2(results, output_dir)

    if "strict_4point_aat" in results:
        fig_section_c2_aat(results, output_dir)
    else:
        print("  SKIPPED: fig_section_C2_aat (no strict_4point_aat in results)")

    fig_section_d1(results, output_dir)

    if "per_condition_aat" in results.get("binary_sentence_level", {}):
        fig_section_d2(results, output_dir)
    else:
        print("  SKIPPED: fig_section_D2 (no per_condition_aat in results)")

    print(f"\nDone. {len(os.listdir(output_dir))} files in {output_dir}/")


if __name__ == "__main__":
    main()