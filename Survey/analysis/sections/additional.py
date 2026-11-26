"""Section G: Additional metrics (tier analysis, gradient, model comparison)."""

import numpy as np
from scipy import stats
from ..config import COND_ORDER, CATEGORY_TO_RATING
from ..metrics import krippendorffs_alpha


def run(plaus_valid, plaus_bin, llm_binary_map, spread_data, diff_by_cond,
        ident_dist, annotator_ids, **kwargs):
    print("\n" + "=" * 70)
    print("G. ADDITIONAL METRICS")
    print("=" * 70)

    # Tier-level alpha
    tier_alpha = {}
    for tier in ["GOLD", "SILVER", "FAIL"]:
        td = plaus_valid[plaus_valid["tier"] == tier]
        if len(td) == 0:
            continue
        pt = td.pivot_table(
            index=["item", "condition"], columns="annotator",
            values="rating", aggfunc="first",
        )
        a = krippendorffs_alpha(pt.values.astype(float))
        tier_alpha[tier] = {"alpha": float(a), "n_sentences": len(pt)}
        print(f"  {tier:6s}: alpha={a:.4f} (n={len(pt)})")

    # Gradient correlation
    meds = plaus_valid.groupby(["item", "condition"])["rating"].median().reset_index()
    meds["generator"] = meds["condition"].map(CATEGORY_TO_RATING)
    rho_s, p_s = stats.spearmanr(meds["generator"], meds["rating"])
    print(f"  Gradient: rho={rho_s:.4f}, p={p_s:.6f}")

    # Per-model agreement
    model_agree = {}
    for model in sorted(plaus_valid["model"].unique()):
        ms = plaus_bin[plaus_bin["model"] == model]
        if len(ms) > 0:
            agrees = 0
            total = 0
            for _, row in ms.iterrows():
                lb = llm_binary_map.get((row["item"], row["condition"]))
                if lb and row["binary_human"]:
                    if lb == row["binary_human"]:
                        agrees += 1
                    total += 1
            acc = agrees / total if total > 0 else 0
            model_agree[model] = {"lh_binary_agreement": float(acc), "n": total}
            print(f"  {model}: LLM-Human binary agree={acc:.1%} (n={total})")

    return {
        "additional": {
            "tier_alpha": tier_alpha,
            "plausibility_gradient": {
                "spearman_rho": float(rho_s), "spearman_p": float(p_s),
            },
            "identifiability": ident_dist,
            "model_agreement": model_agree,
            "human_spread": spread_data,
            "difficulty_by_condition": diff_by_cond,
        },
    }
