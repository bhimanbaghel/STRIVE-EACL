#!/bin/bash
# Merge and score one completed text-only GPT AAT replicate.
# Usage: bash slurm_jobs/score_annotation_textonly_aat_replicate.sh 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN="${1:?expected replicate number}"
[[ "${RUN}" =~ ^[1-9][0-9]*$ ]] || {
  echo "replicate number must be a positive integer" >&2
  exit 2
}

PREFIX_ROOT="${AAT_PREFIX_ROOT:-evalv31agent_textonly_gpt51}"
PREFIX="${PREFIX_ROOT}-r${RUN}"
CSV_TAG="${PREFIX_ROOT//[^a-zA-Z0-9_.-]/_}"
RUN_CSV="${REPO_ROOT}/results/aggregate/aat_${CSV_TAG}_r${RUN}.csv"
PYTHON="${AAT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[ -x "${PYTHON}" ] || PYTHON="python"

"${PYTHON}" scripts/merge_iterative_to_evals.py \
  --eval-prefix "${PREFIX}_" \
  --criterion text \
  --output-csv "${RUN_CSV}"

"${PYTHON}" - "${RUN_CSV}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if len(rows) != 30:
    raise SystemExit(f"Expected exactly 30 AAT rows in {path}, found {len(rows)}")

failed = [
    f"{row.get('eval_run', '?')}/{row.get('verb', '?')}"
    for row in rows
    if row.get("eval_success", "").strip().lower() != "true"
]
if failed:
    raise SystemExit(
        f"Refusing AAT analysis: {len(failed)}/30 evaluator calls failed "
        f"({', '.join(failed[:5])})"
    )

missing_actual = []
allowed = {
    "plausible_easy",
    "plausible_hard",
    "implausible_hard",
    "implausible_easy",
}
for row in rows:
    for condition in (
        "plausible_easy",
        "plausible_hard",
        "implausible_hard",
        "implausible_easy",
    ):
        value = row.get(f"actual_{condition}", "").strip()
        if value not in allowed:
            missing_actual.append(
                f"{row.get('eval_run', '?')}/{row.get('verb', '?')}:"
                f"{condition}={value or '<empty>'}"
            )
if missing_actual:
    raise SystemExit(
        "Refusing AAT analysis: missing/invalid four-point classifications for "
        + ", ".join(missing_actual[:5])
    )

print("All 30 evaluator calls succeeded and contain four-point classifications.")
PY

cd "${REPO_ROOT}/Survey"
"${PYTHON}" -m analysis.main \
  --eval-prefix "${PREFIX}" \
  --eval-csv "${RUN_CSV}" \
  --plausibility-only

echo "Primary AAT result: Survey/outputs/analysis_results_${PREFIX}.json"
