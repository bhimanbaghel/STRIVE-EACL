"""Section A: Survey validation and descriptive statistics."""

import pandas as pd
from ..config import COND_ORDER


def run(plaus_df, plaus_valid, diff_df, qualtrics_df, n_ann, **kwargs):
    print("\n" + "=" * 70)
    print("A. SURVEY VALIDATION")
    print("=" * 70)

    n_items = kwargs.get("n_items", 30)
    sv = {
        "n_annotators": n_ann,
        "n_items": n_items,
        "n_sentences": n_items * 4,
        "n_plausibility_ratings": len(plaus_valid),
        "cannot_decide_rate": float((plaus_df["rating"] == 5).sum() / len(plaus_df)),
        "cannot_decide_count": int((plaus_df["rating"] == 5).sum()),
        "rating_distribution": {
            str(k): v for k, v in
            plaus_df["rating"].value_counts(normalize=True).sort_index().items()
        },
        "difficulty_distribution": {
            str(k): v for k, v in
            diff_df["rating"].value_counts(normalize=True).sort_index().items()
        },
    }

    if "Duration (in seconds)" in qualtrics_df.columns:
        dur = pd.to_numeric(qualtrics_df["Duration (in seconds)"], errors="coerce") / 60
        sv["duration_minutes"] = dur.tolist()
        sv["duration_median"] = float(dur.median())
        sv["duration_per_set"] = float(dur.median() / n_items)

    print(f"  Cannot Decide: {sv['cannot_decide_rate']:.1%}")
    print(f"  Ratings: {sv['rating_distribution']}")
    print(f"  Difficulty: {sv['difficulty_distribution']}")
    if "duration_median" in sv:
        print(f"  Time: median {sv['duration_median']:.0f} min, {sv['duration_per_set']:.1f} min/set")

    return {"survey_validation": sv}
