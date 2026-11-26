"""
Agreement metrics: Cohen's kappa, weighted kappa, Krippendorff's alpha,
Gwet's AC1, percent agreement, bootstrap utilities.
"""

import numpy as np
from itertools import combinations
from .config import DELTAS


def cohens_kappa(y1, y2):
    """Unweighted Cohen's kappa for two raters."""
    mask = ~(np.isnan(y1) | np.isnan(y2))
    y1, y2 = y1[mask].astype(int), y2[mask].astype(int)
    if len(y1) == 0:
        return np.nan
    p_o = (y1 == y2).mean()
    cats = sorted(set(y1) | set(y2))
    p_e = sum((y1 == c).mean() * (y2 == c).mean() for c in cats)
    if p_e == 1:
        return 1.0 if p_o == 1 else np.nan
    return (p_o - p_e) / (1 - p_e)


def weighted_cohens_kappa(y1, y2, n_categories=None, weights="quadratic"):
    """Quadratic-weighted Cohen's kappa."""
    mask = ~(np.isnan(y1) | np.isnan(y2))
    y1, y2 = y1[mask], y2[mask]
    if len(y1) == 0:
        return np.nan
    if n_categories is None:
        n_categories = int(max(np.nanmax(y1), np.nanmax(y2))) + 1
    conf = np.zeros((n_categories, n_categories))
    for a, b in zip(y1.astype(int), y2.astype(int)):
        conf[a, b] += 1
    total = conf.sum()
    if total == 0:
        return np.nan
    w = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            w[i, j] = (i - j) ** 2 / max((n_categories - 1) ** 2, 1)
    p_obs = conf / total
    p_exp = np.outer(conf.sum(axis=1), conf.sum(axis=0)) / total**2
    num = (w * p_obs).sum()
    den = (w * p_exp).sum()
    if den == 0:
        return 1.0 if num == 0 else np.nan
    return 1.0 - num / den


def gwets_ac1(y1, y2):
    """
    Gwet's AC1 for two raters with binary (0/1) data.

    Unlike Cohen's kappa, AC1 handles prevalence imbalance gracefully.
    When nearly all items fall in one category, kappa's expected agreement
    inflates and crushes the statistic. AC1 uses a different chance model
    that avoids this paradox.

    Formula:
        P_e = 2 * pi_hat * (1 - pi_hat)
        where pi_hat = overall proportion in category 1
        AC1 = (P_a - P_e) / (1 - P_e)
    """
    mask = ~(np.isnan(y1) | np.isnan(y2))
    y1, y2 = y1[mask], y2[mask]
    n = len(y1)
    if n == 0:
        return np.nan

    p_a = (y1 == y2).mean()
    # Overall marginal proportion across both raters
    pi_hat = (y1.mean() + y2.mean()) / 2.0
    p_e = 2.0 * pi_hat * (1.0 - pi_hat)

    if p_e == 1.0:
        return 1.0 if p_a == 1.0 else np.nan
    return (p_a - p_e) / (1.0 - p_e)


def percent_agreement(y1, y2):
    """Simple percent agreement between two raters."""
    mask = ~(np.isnan(y1) | np.isnan(y2))
    y1, y2 = y1[mask], y2[mask]
    if len(y1) == 0:
        return np.nan
    return float((y1 == y2).mean())


def krippendorffs_alpha(data_matrix):
    """Krippendorff's alpha for interval data (multi-rater)."""
    n_items, n_coders = data_matrix.shape
    pairs_diff_sq, all_values = [], []
    for i in range(n_items):
        vals = data_matrix[i, :][~np.isnan(data_matrix[i, :])]
        if len(vals) < 2:
            continue
        all_values.extend(vals)
        for a, b in combinations(range(len(vals)), 2):
            pairs_diff_sq.append((vals[a] - vals[b]) ** 2)
    if not pairs_diff_sq:
        return np.nan
    D_o = np.mean(pairs_diff_sq)
    all_vals = np.array(all_values)
    n = len(all_vals)
    if n < 2:
        return np.nan
    D_e = sum(
        (all_vals[a] - all_vals[b]) ** 2
        for a in range(n) for b in range(a + 1, n)
    )
    D_e /= (n * (n - 1) / 2)
    if D_e == 0:
        return 1.0
    return 1.0 - D_o / D_e


def summarize_bootstrap(boot, label=""):
    """Summarize bootstrap distribution with percentiles and non-inferiority."""
    p5 = float(np.percentile(boot, 5))
    med = float(np.median(boot))
    results = {
        "bootstrap_5th_pctile": p5,
        "bootstrap_median": med,
        "bootstrap_95th_pctile": float(np.percentile(boot, 95)),
        "noninferiority": {f"delta_{d:.2f}": bool(p5 > -d) for d in DELTAS},
        "percentiles": {
            str(p): float(np.percentile(boot, p))
            for p in [2.5, 5, 25, 50, 75, 95, 97.5]
        },
    }
    smallest = None
    for d_test in np.arange(0.01, 0.50, 0.01):
        if p5 > -d_test:
            smallest = float(round(d_test, 2))
            break
    results["smallest_passing_delta"] = smallest

    if label:
        print(f"  {label}: median={med:.4f}, 5th pctile={p5:.4f}")
        for d in DELTAS:
            tag = "PASS" if p5 > -d else "FAIL"
            print(f"    delta={d:.2f}: {tag}")
        if smallest:
            print(f"    Smallest passing delta: {smallest}")
        else:
            print(f"    No passing delta (up to 0.49)")
    return results
