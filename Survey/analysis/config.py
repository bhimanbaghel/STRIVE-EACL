"""
Configuration: paths, constants, mappings.
Edit this file to change file paths or analysis parameters.
"""

from pathlib import Path

# ---- Directory layout ----
# Survey/ lives inside Events/; experiment outputs live in Events/results/
SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent
OUTPUT_DIR = SURVEY_DIR / "outputs"
EXPORTS_DIR = OUTPUT_DIR / "exports"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
EVALUATIONS_DIR = PROJECT_ROOT / "results" / "evaluations"

# ---- Evaluator version ----
# Controls which eval_run rows are loaded from the merged CSV.
# Set via CLI: python -m analysis.main --eval-prefix evalv4
# The source_run in sampled_stimuli.csv uses "evalv3_" prefix;
# this prefix is swapped at load time to match the requested version.
EVAL_PREFIX = "evalv3"  # default; overridden by CLI

# ---- File paths ----
QUALTRICS_CSV = SURVEY_DIR / "annotation_responses.csv"
STIMULI_CSV = SURVEY_DIR / "sampled_stimuli.csv"
EVAL_CSV = PROJECT_ROOT / "results" / "aggregate" / "all_evaluations_merged_preview.csv"

# Output paths are versioned by EVAL_PREFIX (set at runtime by main.py)
RESULTS_JSON = OUTPUT_DIR / "analysis_results_v2.json"  # overridden at runtime


def ensure_output_dirs():
    """Create output directories if missing."""
    for d in (OUTPUT_DIR, EXPORTS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def results_json_path(eval_prefix: str) -> Path:
    return OUTPUT_DIR / f"analysis_results_{eval_prefix}.json"


def sentence_examples_path(eval_prefix: str) -> Path:
    return EXPORTS_DIR / f"sentence_examples_{eval_prefix}.csv"


def sentence_comparison_path(base_prefix: str, new_prefix: str) -> Path:
    return EXPORTS_DIR / f"sentence_comparison_{base_prefix}_vs_{new_prefix}.csv"


def visual_identifiability_examples_path(eval_prefix: str) -> Path:
    return EXPORTS_DIR / f"visual_identifiability_examples_{eval_prefix}.csv"


def visual_pairwise_examples_path(eval_prefix: str) -> Path:
    return EXPORTS_DIR / f"visual_pairwise_examples_{eval_prefix}.csv"


def figures_dir(eval_prefix: str) -> Path:
    return FIGURES_DIR / f"figures_{eval_prefix}"


def tables_dir(eval_prefix: str) -> Path:
    return TABLES_DIR / f"tables_{eval_prefix}"


def evaluation_json_path(eval_run: str, verb: str) -> Path:
    return EVALUATIONS_DIR / eval_run / f"{verb}.json"


# ---- Survey structure ----
QS_PER_ITEM = 13
N_ITEMS = 30
Q_START = 2

# ---- Bootstrap ----
BOOTSTRAP_N = 10000
RANDOM_SEED = 42
DELTAS = [0.05, 0.10, 0.15]

# ---- Annotator exclusions (empty = use all) ----
EXCLUDE_ANNOTATORS = []

# ---- Category mappings ----
CATEGORY_TO_RATING = {
    "plausible_easy": 1, "plausible_hard": 2,
    "implausible_hard": 3, "implausible_easy": 4,
}
RATING_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_RATING.items()}

CATEGORY_TO_BINARY = {
    "plausible_easy": "PLAUSIBLE", "plausible_hard": "PLAUSIBLE",
    "implausible_hard": "IMPLAUSIBLE", "implausible_easy": "IMPLAUSIBLE",
}

COND_ORDER = ["plausible_easy", "plausible_hard", "implausible_hard", "implausible_easy"]

# ---- Qualtrics text -> numeric mappings ----
PLAUS_TEXT_MAP = {}
for _dash in [" — ", " \u2014 "]:
    PLAUS_TEXT_MAP[f"Clearly Plausible{_dash}this is a normal, expected event"] = 1
    PLAUS_TEXT_MAP[f"Somewhat Plausible{_dash}unusual but I can see it happening"] = 2
    PLAUS_TEXT_MAP[f"Somewhat Implausible{_dash}feels wrong but I can see a remote connection"] = 3
    PLAUS_TEXT_MAP[f"Clearly Implausible{_dash}this makes no sense at all"] = 4
    PLAUS_TEXT_MAP[f"Cannot Decide{_dash}I genuinely cannot judge this"] = 5

DIFF_TEXT_MAP = {}
for _dash in [" — ", " \u2014 "]:
    DIFF_TEXT_MAP[f"Easy{_dash}I decided immediately"] = 1
    DIFF_TEXT_MAP[f"Moderate{_dash}I had to think about it"] = 2
    DIFF_TEXT_MAP[f"Hard{_dash}I was very unsure"] = 3

IDENT_TEXT_MAP = {
    "Yes -- I could clearly identify them from their clothing, gear, or appearance": "Yes",
    "Maybe -- I might guess correctly, but they could be confused with other roles": "Maybe",
    "No - They would look like an ordinary person in everyday clothing": "No",
}


def rating_to_binary(rating):
    if rating in (1, 2):
        return "PLAUSIBLE"
    elif rating in (3, 4):
        return "IMPLAUSIBLE"
    return None
