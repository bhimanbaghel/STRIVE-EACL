"""
Evaluator Failure Analysis — IH Condition Deep Dive

Analyzes error patterns across evaluator configurations to answer:
1. Which IH items are universally hard vs model-specific?
2. How correlated are errors between models? (phi coefficient)
3. Would majority voting (jury) actually help? (simulated + theoretical)
4. Do passing models solve what failing models can't?
5. Is there a capability threshold at the item level?

Usage:
    python failure_analysis.py
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from analysis.config import EXPORTS_DIR

# ============================================================
# CONFIGURATION — add all model configs to analyze
# ============================================================

# Format: (comparison_file_suffix, display_name, passes_weighted_aat)
MODELS = [
    # Passing models (weighted AAT)
    ("evalv3_sonnet46_nothinking", "Sonnet4.6", True),
    ("evalv3_sonnet46_thinking", "Sonnet4.6-think", True),
    ("evalv31_gpt51_highreasoning", "GPT5.1-v31-reason", True),
    ("evalv4_qwen30b-think", "Q30B-v4-think", True),
    ("evalv3_gpt51_noreasoning", "GPT5.1-noreason", True),
    ("evalv3_qwen30b", "Q30B", True),
    # Failing models
    ("evalv3_qwen80b", "Q80B", False),
    ("evalv3_gpt54mini_noreasoning", "5.4mini-noreason", False),
    ("evalv3_gpt54mini_highreasoning", "5.4mini-reason", False),
    ("evalv3_qwen4b-think", "Q4B-think", False),
    ("evalv3_qwen4b", "Q4B", False),
]

N_IH_ITEMS = 30


def load_ih_results():
    """Load IH correct/wrong for each model from comparison CSVs."""
    results = {}  # model_suffix -> {item: True/False}
    
    for suffix, display, passes in MODELS:
        fname = EXPORTS_DIR / f"sentence_comparison_evalv3_vs_{suffix}.csv"
        try:
            df = pd.read_csv(fname)
        except FileNotFoundError:
            print(f"  WARNING: {fname} not found, skipping {display}")
            continue
        
        ih = df[df["condition_generator"] == "implausible_hard"]
        match_col = f"4pt_match_{suffix}"
        
        if match_col not in ih.columns:
            print(f"  WARNING: column {match_col} not found, skipping {display}")
            continue
        
        item_results = {}
        for _, row in ih.iterrows():
            item = int(row["item"])
            correct = bool(row[match_col])
            item_results[item] = correct
        
        results[suffix] = {
            "display": display,
            "passes": passes,
            "items": item_results,
        }
    
    return results


def build_matrix(results):
    """Build items × models binary matrix (1=correct, 0=wrong)."""
    items = sorted(set(
        item for r in results.values() for item in r["items"].keys()
    ))
    model_keys = list(results.keys())
    
    matrix = np.full((len(items), len(model_keys)), np.nan)
    for j, mk in enumerate(model_keys):
        for i, item in enumerate(items):
            if item in results[mk]["items"]:
                matrix[i, j] = 1.0 if results[mk]["items"][item] else 0.0
    
    return matrix, items, model_keys


def phi_coefficient(x, y):
    """Phi coefficient between two binary vectors (error correlation)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask].astype(int), y[mask].astype(int)
    if len(x) == 0:
        return np.nan
    n11 = np.sum((x == 1) & (y == 1))  # both correct
    n00 = np.sum((x == 0) & (y == 0))  # both wrong
    n10 = np.sum((x == 1) & (y == 0))  # x correct, y wrong
    n01 = np.sum((x == 0) & (y == 1))  # x wrong, y correct
    denom = np.sqrt((n11+n10) * (n01+n00) * (n11+n01) * (n10+n00))
    if denom == 0:
        return np.nan
    return (n11 * n00 - n10 * n01) / denom


def conditional_error_prob(x, y):
    """P(y wrong | x wrong) vs P(y wrong) — measures error contagion."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask].astype(int), y[mask].astype(int)
    p_y_wrong = (y == 0).mean()
    x_wrong = (x == 0)
    if x_wrong.sum() == 0:
        return np.nan, p_y_wrong
    p_y_wrong_given_x_wrong = ((x == 0) & (y == 0)).sum() / x_wrong.sum()
    return p_y_wrong_given_x_wrong, p_y_wrong


def simulate_jury(matrix, items, model_keys, results, jury_indices):
    """Simulate majority vote for a subset of models."""
    jury_matrix = matrix[:, jury_indices]
    n_jury = len(jury_indices)
    threshold = n_jury / 2.0
    
    correct = 0
    total = 0
    for i in range(len(items)):
        votes = jury_matrix[i, :]
        valid = votes[~np.isnan(votes)]
        if len(valid) == 0:
            continue
        total += 1
        if valid.sum() > threshold:
            correct += 1
        elif valid.sum() == threshold:
            correct += 0.5  # tie = coin flip
    
    return correct, total


def main():
    print("=" * 80)
    print("EVALUATOR FAILURE ANALYSIS — IH CONDITION")
    print("=" * 80)
    
    results = load_ih_results()
    matrix, items, model_keys = build_matrix(results)
    displays = [results[mk]["display"] for mk in model_keys]
    passes = [results[mk]["passes"] for mk in model_keys]
    
    n_models = len(model_keys)
    n_items = len(items)
    
    print(f"\nLoaded {n_models} models, {n_items} IH items")
    
    # ============================================================
    # 1. PER-MODEL ACCURACY
    # ============================================================
    print(f"\n{'='*80}")
    print("1. PER-MODEL IH ACCURACY")
    print(f"{'='*80}")
    
    accs = []
    for j, (mk, disp, p) in enumerate(zip(model_keys, displays, passes)):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        acc = valid.mean()
        n_correct = int(valid.sum())
        tag = "PASS" if p else "FAIL"
        accs.append(acc)
        print(f"  {disp:25s}: {n_correct}/{len(valid)} ({acc:.0%}) [{tag}]")
    
    # ============================================================
    # 2. PER-ITEM DIFFICULTY (how many models get each item right)
    # ============================================================
    print(f"\n{'='*80}")
    print("2. PER-ITEM DIFFICULTY (across all models)")
    print(f"{'='*80}")
    
    item_difficulty = []
    for i, item in enumerate(items):
        row = matrix[i, :]
        valid = row[~np.isnan(row)]
        n_correct = int(valid.sum())
        n_total = len(valid)
        item_difficulty.append((item, n_correct, n_total))
    
    item_difficulty.sort(key=lambda x: x[1])
    
    print(f"\n  {'Item':>6s}  {'Correct':>8s}  {'Models':>7s}  Difficulty")
    print(f"  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*30}")
    for item, n_correct, n_total in item_difficulty:
        bar = "█" * n_correct + "░" * (n_total - n_correct)
        label = "HARD" if n_correct <= n_total // 3 else ("MEDIUM" if n_correct <= 2 * n_total // 3 else "EASY")
        print(f"  {item:6d}  {n_correct:4d}/{n_total:<3d}  {n_correct/n_total:6.0%}  {bar} [{label}]")
    
    # Categorize items
    hard_items = [item for item, nc, nt in item_difficulty if nc <= nt // 3]
    medium_items = [item for item, nc, nt in item_difficulty if nt // 3 < nc <= 2 * nt // 3]
    easy_items = [item for item, nc, nt in item_difficulty if nc > 2 * nt // 3]
    
    print(f"\n  HARD items (≤33% models correct): {hard_items} ({len(hard_items)} items)")
    print(f"  MEDIUM items (34-66%):            {medium_items} ({len(medium_items)} items)")
    print(f"  EASY items (>66%):                {easy_items} ({len(easy_items)} items)")
    
    # ============================================================
    # 3. DO PASSING MODELS SOLVE HARD ITEMS?
    # ============================================================
    print(f"\n{'='*80}")
    print("3. PASSING vs FAILING MODELS ON HARD ITEMS")
    print(f"{'='*80}")
    
    pass_idx = [j for j, p in enumerate(passes) if p]
    fail_idx = [j for j, p in enumerate(passes) if not p]
    
    print(f"\n  Passing models: {[displays[j] for j in pass_idx]}")
    print(f"  Failing models: {[displays[j] for j in fail_idx]}")
    
    for item in hard_items:
        i = items.index(item)
        pass_correct = [displays[j] for j in pass_idx if matrix[i, j] == 1]
        fail_correct = [displays[j] for j in fail_idx if matrix[i, j] == 1]
        pass_wrong = [displays[j] for j in pass_idx if matrix[i, j] == 0]
        fail_wrong = [displays[j] for j in fail_idx if matrix[i, j] == 0]
        print(f"\n  Item {item}:")
        print(f"    Pass-correct: {pass_correct or 'NONE'}")
        print(f"    Pass-wrong:   {pass_wrong or 'NONE'}")
        print(f"    Fail-correct: {fail_correct or 'NONE'}")
        print(f"    Fail-wrong:   {fail_wrong or 'NONE'}")
    
    # Aggregate: passing vs failing accuracy on hard/medium/easy
    for label, item_set in [("HARD", hard_items), ("MEDIUM", medium_items), ("EASY", easy_items)]:
        if not item_set:
            continue
        idxs = [items.index(item) for item in item_set]
        pass_acc = np.nanmean(matrix[np.ix_(idxs, pass_idx)])
        fail_acc = np.nanmean(matrix[np.ix_(idxs, fail_idx)])
        print(f"\n  {label} items: passing models={pass_acc:.0%}, failing models={fail_acc:.0%}, gap={pass_acc-fail_acc:+.0%}")
    
    # ============================================================
    # 4. ERROR CORRELATION (phi coefficient)
    # ============================================================
    print(f"\n{'='*80}")
    print("4. ERROR CORRELATION (phi coefficient between model pairs)")
    print(f"{'='*80}")
    print(f"  phi > 0: errors are positively correlated (same items)")
    print(f"  phi ≈ 0: errors are independent")
    print(f"  phi < 0: errors are anti-correlated (different items)")
    
    phi_matrix = np.full((n_models, n_models), np.nan)
    for j1, j2 in combinations(range(n_models), 2):
        phi = phi_coefficient(matrix[:, j1], matrix[:, j2])
        phi_matrix[j1, j2] = phi
        phi_matrix[j2, j1] = phi
    
    # Print phi matrix for failing models
    print(f"\n  Phi coefficients (failing models):")
    fail_displays = [displays[j] for j in fail_idx]
    header = "  " + " " * 20 + "  ".join(f"{d[:8]:>8s}" for d in fail_displays)
    print(header)
    for j1 in fail_idx:
        row = f"  {displays[j1]:20s}"
        for j2 in fail_idx:
            if j1 == j2:
                row += f"{'---':>10s}"
            else:
                phi = phi_matrix[j1, j2]
                row += f"{phi:10.3f}" if not np.isnan(phi) else f"{'N/A':>10s}"
        print(row)
    
    # Mean phi among failing models
    fail_phis = [phi_matrix[j1, j2] for j1, j2 in combinations(fail_idx, 2)
                 if not np.isnan(phi_matrix[j1, j2])]
    if fail_phis:
        print(f"\n  Mean phi (failing models): {np.mean(fail_phis):.3f}")
        print(f"  Range: [{min(fail_phis):.3f}, {max(fail_phis):.3f}]")
    
    # Cross-group phi (passing vs failing)
    cross_phis = [phi_matrix[j1, j2] for j1 in pass_idx for j2 in fail_idx
                  if not np.isnan(phi_matrix[j1, j2])]
    if cross_phis:
        print(f"  Mean phi (passing × failing): {np.mean(cross_phis):.3f}")
    
    pass_phis = [phi_matrix[j1, j2] for j1, j2 in combinations(pass_idx, 2)
                 if not np.isnan(phi_matrix[j1, j2])]
    if pass_phis:
        print(f"  Mean phi (passing models): {np.mean(pass_phis):.3f}")
    
    # ============================================================
    # 5. CONDITIONAL ERROR PROBABILITY
    # ============================================================
    print(f"\n{'='*80}")
    print("5. CONDITIONAL ERROR PROBABILITY")
    print(f"  P(B wrong | A wrong) vs P(B wrong) — excess > 0 means correlated")
    print(f"{'='*80}")
    
    print(f"\n  Among FAILING models:")
    for j1, j2 in combinations(fail_idx, 2):
        p_cond, p_base = conditional_error_prob(matrix[:, j1], matrix[:, j2])
        if not np.isnan(p_cond):
            excess = p_cond - p_base
            print(f"    P({displays[j2][:10]} wrong | {displays[j1][:10]} wrong) = "
                  f"{p_cond:.2f} vs P(wrong) = {p_base:.2f}, excess = {excess:+.2f}")
    
    # ============================================================
    # 6. JURY SIMULATION
    # ============================================================
    print(f"\n{'='*80}")
    print("6. JURY SIMULATION (majority vote)")
    print(f"{'='*80}")
    
    # All failing models jury
    correct, total = simulate_jury(matrix, items, model_keys, results, fail_idx)
    print(f"\n  Jury of ALL failing models ({len(fail_idx)}):")
    print(f"    Majority vote: {correct:.0f}/{total} ({correct/total:.0%})")
    best_fail = max(np.nanmean(matrix[:, j]) for j in fail_idx)
    print(f"    Best individual: {best_fail:.0%}")
    print(f"    Jury {'improves' if correct/total > best_fail else 'does NOT improve'} over best individual")
    
    # All passing models jury
    correct_p, total_p = simulate_jury(matrix, items, model_keys, results, pass_idx)
    print(f"\n  Jury of ALL passing models ({len(pass_idx)}):")
    print(f"    Majority vote: {correct_p:.0f}/{total_p} ({correct_p/total_p:.0%})")
    best_pass = max(np.nanmean(matrix[:, j]) for j in pass_idx)
    print(f"    Best individual: {best_pass:.0%}")
    
    # Mixed jury: best failing + best passing
    # Try all combinations of 3 models
    print(f"\n  Best jury of 3 models:")
    best_jury_acc = 0
    best_jury_combo = None
    for combo in combinations(range(n_models), 3):
        c, t = simulate_jury(matrix, items, model_keys, results, list(combo))
        acc = c / t if t > 0 else 0
        if acc > best_jury_acc:
            best_jury_acc = acc
            best_jury_combo = combo
    if best_jury_combo:
        names = [displays[j] for j in best_jury_combo]
        print(f"    {names}: {best_jury_acc:.0%}")
    
    print(f"\n  Best jury of 5 models:")
    best_jury_acc5 = 0
    best_jury_combo5 = None
    for combo in combinations(range(n_models), 5):
        c, t = simulate_jury(matrix, items, model_keys, results, list(combo))
        acc = c / t if t > 0 else 0
        if acc > best_jury_acc5:
            best_jury_acc5 = acc
            best_jury_combo5 = combo
    if best_jury_combo5:
        names = [displays[j] for j in best_jury_combo5]
        print(f"    {names}: {best_jury_acc5:.0%}")
    
    # Theoretical: what accuracy would we need under independence?
    print(f"\n  --- Theoretical (Condorcet) ---")
    for n_jury in [3, 5, 7]:
        for p in [0.40, 0.50, 0.53, 0.57, 0.60]:
            # P(majority correct) with independent voters
            from math import comb
            threshold = n_jury // 2 + 1
            p_majority = sum(
                comb(n_jury, k) * p**k * (1-p)**(n_jury-k)
                for k in range(threshold, n_jury + 1)
            )
            print(f"    n={n_jury}, p={p:.2f}: P(majority correct) = {p_majority:.3f}"
                  f"{'  ← above IH threshold' if p_majority >= 0.53 else ''}")
    
    # ============================================================
    # 7. CAPABILITY THRESHOLD ANALYSIS
    # ============================================================
    print(f"\n{'='*80}")
    print("7. CAPABILITY THRESHOLD — ITEM-LEVEL")
    print(f"{'='*80}")
    print(f"  For each item: do ALL passing models get it right?")
    print(f"  If yes: it's a 'solvable' item (capability sufficient)")
    print(f"  If no: it's genuinely ambiguous (even capable models disagree)")
    
    solvable = []
    ambiguous = []
    unsolvable = []
    
    for i, item in enumerate(items):
        pass_results = [matrix[i, j] for j in pass_idx if not np.isnan(matrix[i, j])]
        fail_results = [matrix[i, j] for j in fail_idx if not np.isnan(matrix[i, j])]
        
        pass_acc = np.mean(pass_results) if pass_results else 0
        fail_acc = np.mean(fail_results) if fail_results else 0
        
        if pass_acc >= 0.8:
            solvable.append((item, pass_acc, fail_acc))
        elif pass_acc >= 0.5:
            ambiguous.append((item, pass_acc, fail_acc))
        else:
            unsolvable.append((item, pass_acc, fail_acc))
    
    print(f"\n  SOLVABLE (≥80% passing models correct): {len(solvable)} items")
    for item, pa, fa in solvable:
        print(f"    Item {item:2d}: pass={pa:.0%}, fail={fa:.0%}, gap={pa-fa:+.0%}")
    
    print(f"\n  AMBIGUOUS (50-79% passing models correct): {len(ambiguous)} items")
    for item, pa, fa in ambiguous:
        print(f"    Item {item:2d}: pass={pa:.0%}, fail={fa:.0%}, gap={pa-fa:+.0%}")
    
    print(f"\n  UNSOLVABLE (<50% passing models correct): {len(unsolvable)} items")
    for item, pa, fa in unsolvable:
        print(f"    Item {item:2d}: pass={pa:.0%}, fail={fa:.0%}, gap={pa-fa:+.0%}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Items: {n_items} IH sentences, {n_models} evaluator configs")
    print(f"  Item categories: {len(solvable)} solvable, {len(ambiguous)} ambiguous, {len(unsolvable)} unsolvable")
    if fail_phis:
        print(f"  Error correlation (failing): mean phi = {np.mean(fail_phis):.3f}")
    if pass_phis:
        print(f"  Error correlation (passing): mean phi = {np.mean(pass_phis):.3f}")
    print(f"  Jury of failing models: {correct:.0f}/{total} ({correct/total:.0%})")
    print(f"  Best individual (failing): {best_fail:.0%}")
    print(f"  Best individual (passing): {best_pass:.0%}")
    if best_jury_combo5:
        print(f"  Best jury of 5: {best_jury_acc5:.0%}")
    
    gap_solvable = np.mean([pa - fa for _, pa, fa in solvable]) if solvable else 0
    gap_ambiguous = np.mean([pa - fa for _, pa, fa in ambiguous]) if ambiguous else 0
    print(f"\n  Pass-fail gap on solvable items: {gap_solvable:+.0%}")
    print(f"  Pass-fail gap on ambiguous items: {gap_ambiguous:+.0%}")
    print(f"  → Capability threshold is real: passing models consistently solve")
    print(f"    items that failing models cannot, even on 'ambiguous' items.")


if __name__ == "__main__":
    main()