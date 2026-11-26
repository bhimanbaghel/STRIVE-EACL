"""
Stratified agreement AAT and generator-defect detection.

The primary unweighted-kappa AAT pools all 120 sentences. Easy items
(plausible_easy, implausible_easy) can keep that pooled test above the
non-inferiority line even when Hard items (plausible_hard, implausible_hard)
are where the evaluator and the annotators come apart.

Three checks:
  1. Stratified multiclass kappa AAT: combined (N=120), easy (N=60), hard (N=60).
  2. Per-category raw-agreement AAT (N=30). Raw agreement avoids the
     zero-variance kappa paradox inside a single category.
  3. Generator-defect detection: of the items whose human majority rejects the
     generator's assigned category, how many does the evaluator also reject?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CATEGORY_TO_RATING


def cohen_kappa_score(y1, y2, labels=(1, 2, 3, 4)):
    """Unweighted Cohen's kappa with a fixed label set.

    Matches sklearn.metrics.cohen_kappa_score(..., labels=[1, 2, 3, 4]).
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    index = {label: i for i, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=float)
    for left, right in zip(y1, y2):
        confusion[index[int(left)], index[int(right)]] += 1
    total = confusion.sum()
    if total == 0:
        return np.nan
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / total
    weights = np.ones_like(confusion)
    np.fill_diagonal(weights, 0.0)
    denominator = np.sum(weights * expected)
    if denominator == 0:
        return np.nan
    return float(1.0 - np.sum(weights * confusion) / denominator)


def recall_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    positive = int(y_true.sum())
    if positive == 0:
        return 0.0
    return float(np.sum(y_true & y_pred) / positive)


def precision_score(y_true, y_pred, zero_division=0):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    predicted = int(y_pred.sum())
    if predicted == 0:
        return float(zero_division)
    return float(np.sum(y_true & y_pred) / predicted)


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_prompt_comprehensive(df, human_matrix, llm_vec, n_boot=10000, delta_thresh=0.05):
    """
    Runs 3 complementary checks to prevent Easy classes (PE/IE) from masking Hard class (PH/IH) failures:
      1. Stratified Multiclass Kappa AAT: Combined (N=120), Easy (PE+IE, N=60), Hard (PH+IH, N=60)
      2. Per-Category Raw Agreement AAT (po_delta, N=30 per category — avoids Zero-Variance Kappa Paradox)
      3. Generator Defect Detection Recall & F1 (how many human-rejected items the evaluator catches)
    """
    human_matrix = np.asarray(human_matrix)
    llm_vec = np.asarray(llm_vec)
    n_anns = human_matrix.shape[1]
    rng = np.random.default_rng(42)

    def boot_aat(sub_h, sub_l, use_kappa=True):
        n_items = len(sub_l)
        idx = rng.integers(0, n_items, size=(n_boot, n_items))
        if not use_kappa:
            # Linear per-sentence agreement delta
            lh_per_s = (sub_h == sub_l[:, None]).mean(axis=1)
            hh_per_s = np.mean(
                [sub_h[:, i] == sub_h[:, j] for i in range(n_anns) for j in range(i + 1, n_anns)],
                axis=0,
            )
            diff = lh_per_s - hh_per_s
            boots = diff[idx].mean(axis=1)
            return float(diff.mean()), float(np.percentile(boots, 5))

        # Vectorized unweighted Cohen's kappa across bootstrap resamples
        l_b = sub_l[idx]
        l_p = np.stack([(l_b == c).mean(axis=1) for c in (1, 2, 3, 4)], axis=1)
        h_b = [sub_h[:, i][idx] for i in range(n_anns)]
        h_p = [
            np.stack([(h_b[i] == c).mean(axis=1) for c in (1, 2, 3, 4)], axis=1)
            for i in range(n_anns)
        ]

        with np.errstate(divide="ignore", invalid="ignore"):
            lh_k = np.stack([
                np.where(
                    (l_p * h_p[i]).sum(1) < 1,
                    ((l_b == h_b[i]).mean(1) - (l_p * h_p[i]).sum(1)) / (1 - (l_p * h_p[i]).sum(1)),
                    0.0,
                )
                for i in range(n_anns)
            ], axis=1)
            hh_k = np.stack([
                np.where(
                    (h_p[i] * h_p[j]).sum(1) < 1,
                    (
                        (h_b[i] == h_b[j]).mean(1) - (h_p[i] * h_p[j]).sum(1)
                    ) / (1 - (h_p[i] * h_p[j]).sum(1)),
                    0.0,
                )
                for i in range(n_anns) for j in range(i + 1, n_anns)
            ], axis=1)
        boots = np.nanmean(lh_k, axis=1) - np.nanmean(hh_k, axis=1)

        lh_orig = np.nanmean([
            cohen_kappa_score(sub_l, sub_h[:, i], labels=(1, 2, 3, 4))
            for i in range(n_anns)
        ])
        hh_orig = np.nanmean([
            cohen_kappa_score(sub_h[:, i], sub_h[:, j], labels=(1, 2, 3, 4))
            for i in range(n_anns) for j in range(i + 1, n_anns)
        ])
        return float(lh_orig - hh_orig), float(np.percentile(boots, 5))

    results = {}
    # 1. Stratified Kappa AAT (Combined, Easy, Hard)
    for name, mask in [
        ("kappa_combined", np.ones(len(df), dtype=bool)),
        ("kappa_easy_PE_IE", df["condition_generator"].isin(["plausible_easy", "implausible_easy"]).values),
        ("kappa_hard_PH_IH", df["condition_generator"].isin(["plausible_hard", "implausible_hard"]).values),
    ]:
        delta, p5 = boot_aat(human_matrix[mask], llm_vec[mask], use_kappa=True)
        results[name] = {"delta": round(delta, 4), "p5": round(p5, 4), "pass": bool(p5 > -delta_thresh)}

    # 2. Per-Category Raw Agreement AAT (N=30 each)
    for cond in ["plausible_easy", "plausible_hard", "implausible_hard", "implausible_easy"]:
        mask = (df["condition_generator"] == cond).values
        delta, p5 = boot_aat(human_matrix[mask], llm_vec[mask], use_kappa=False)
        results[f"po_{cond}"] = {"delta": round(delta, 4), "p5": round(p5, 4), "pass": bool(p5 > -delta_thresh)}

    # 3. Defect Detection Recall & F1 (Critical for stopping rubber-stamping!)
    gen_map = {"plausible_easy": 1, "plausible_hard": 2, "implausible_hard": 3, "implausible_easy": 4}
    gen_vec = df["condition_generator"].map(gen_map).values
    hum_maj = df["human_majority_rating"].values
    human_defect = (hum_maj != gen_vec)
    llm_defect = (llm_vec != gen_vec)
    results["defect_detection"] = {
        "overall_recall": round(float(recall_score(human_defect, llm_defect)), 4),
        "overall_precision": round(float(precision_score(human_defect, llm_defect, zero_division=0)), 4),
        "overall_f1": round(float(f1_score(human_defect, llm_defect)), 4),
        "PH_defect_recall": round(float(recall_score(human_defect[gen_vec == 2], llm_defect[gen_vec == 2])), 4),
        "IH_defect_recall": round(float(recall_score(human_defect[gen_vec == 3], llm_defect[gen_vec == 3])), 4),
        "n_human_defects": int(human_defect.sum()),
        "n_PH_human_defects": int(human_defect[gen_vec == 2].sum()),
        "n_IH_human_defects": int(human_defect[gen_vec == 3].sum()),
        "n_PH": int((gen_vec == 2).sum()),
        "n_IH": int((gen_vec == 3).sum()),
    }
    return results


def _majority(ratings: np.ndarray) -> float:
    valid = ratings[np.isfinite(ratings)].astype(int)
    if len(valid) == 0:
        return np.nan
    values, counts = np.unique(valid, return_counts=True)
    return float(values[np.argmax(counts)])


def run(llm, annotator_ids, pivot_raw, **kwargs):
    print("\n" + "=" * 70)
    print("C3. STRATIFIED KAPPA AAT AND GENERATOR-DEFECT DETECTION")
    print("  Easy (PE+IE) can mask Hard (PH+IH) failures in the pooled AAT.")
    print("  Pass rule for agreement checks: bootstrap 5th percentile > -0.05")
    print("=" * 70)

    actual_anns = [a for a in annotator_ids if a in pivot_raw.columns]
    human_matrix = pivot_raw[actual_anns].to_numpy(dtype=float)
    llm_ratings = []
    conditions = []
    majorities = []
    for item, cond in pivot_raw.index:
        if item in llm and cond in llm[item]["judgments"]:
            rating = llm[item]["judgments"][cond]["llm_rating"]
            llm_ratings.append(float(rating) if rating is not None else np.nan)
        else:
            llm_ratings.append(np.nan)
        row = pivot_raw.loc[(item, cond), actual_anns].to_numpy(dtype=float)
        conditions.append(cond)
        majorities.append(_majority(row))

    llm_vec = np.asarray(llm_ratings, dtype=float)
    majority = np.asarray(majorities, dtype=float)
    complete = (
        np.isfinite(human_matrix).all(axis=1)
        & np.isfinite(llm_vec)
        & np.isfinite(majority)
    )
    n_dropped = int((~complete).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped} sentences with a missing human or LLM rating")
    human_matrix = human_matrix[complete]
    llm_vec = llm_vec[complete]
    df = pd.DataFrame({
        "condition_generator": np.asarray(conditions)[complete],
        "human_majority_rating": majority[complete].astype(int),
    })
    print(f"  Sentences scored: {len(df)}")
    for cond, rating in CATEGORY_TO_RATING.items():
        n_cond = int((df["condition_generator"] == cond).sum())
        print(f"    {cond:20s} n={n_cond}  generator label={rating}")

    results = evaluate_prompt_comprehensive(df, human_matrix, llm_vec)
    for name, row in results.items():
        if name == "defect_detection":
            continue
        tag = "PASS" if row["pass"] else "FAIL"
        print(f"  {name:28s} delta={row['delta']:+.4f}  p5={row['p5']:+.4f}  {tag}")
    defect = results["defect_detection"]
    print(
        f"  defect detection  recall={defect['overall_recall']:.4f}  "
        f"precision={defect['overall_precision']:.4f}  f1={defect['overall_f1']:.4f}"
    )
    print(
        f"    PH recall={defect['PH_defect_recall']:.4f} "
        f"({defect['n_PH_human_defects']}/{defect['n_PH']} human defects)  "
        f"IH recall={defect['IH_defect_recall']:.4f} "
        f"({defect['n_IH_human_defects']}/{defect['n_IH']} human defects)"
    )
    return {"stratified_defect_aat": results}
