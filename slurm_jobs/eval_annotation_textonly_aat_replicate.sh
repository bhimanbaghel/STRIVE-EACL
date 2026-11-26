#!/bin/bash
# Validate or run the text-only GPT evaluator on the fixed 30-set AAT sample.
#
# No-call structural check:
#   bash slurm_jobs/eval_annotation_textonly_aat_replicate.sh plan
# One-item API smoke (excluded from the five-run AAT decision):
#   bash slurm_jobs/eval_annotation_textonly_aat_replicate.sh smoke
# One complete independent replicate:
#   bash slurm_jobs/eval_annotation_textonly_aat_replicate.sh replicate 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:?expected plan, smoke, or replicate}"
RUN="${2:-}"
PROMPT="${AAT_EVAL_PROMPT:-prompts/evaluator_v3.1_agent_textonly.json}"
PREFIX_ROOT="${AAT_PREFIX_ROOT:-evalv31agent_textonly_gpt51}"
MODEL="${AAT_MODEL:-gpt-5.1-2025-11-13}"
PYTHON="${AAT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[ -x "${PYTHON}" ] || PYTHON="python"

case "${MODE}" in
  plan)
    PREFIX="${PREFIX_ROOT}-plan"
    ;;
  smoke)
    PREFIX="${PREFIX_ROOT}-smoke"
    ;;
  replicate)
    [[ "${RUN}" =~ ^[1-9][0-9]*$ ]] || {
      echo "replicate mode requires a positive integer run number" >&2
      exit 2
    }
    PREFIX="${PREFIX_ROOT}-r${RUN}"
    ;;
  *)
    echo "Unknown mode: ${MODE}; expected plan, smoke, or replicate" >&2
    exit 2
    ;;
esac

[ -f "${PROMPT}" ] || {
  echo "Missing evaluator prompt: ${PROMPT}" >&2
  exit 2
}

echo "Mode: ${MODE}"
echo "Prompt: ${PROMPT}"
echo "Prompt SHA-256: $(sha256sum "${PROMPT}" | awk '{print $1}')"
echo "Output prefix: ${PREFIX}"

PLANNED_COUNT=0

validate_generation() {
  local path="$1"
  "${PYTHON}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("success") and d.get("parsed") else 1)' \
    "${path}"
}

run_eval() {
  local suffix="$1" input="$2"
  shift 2
  local name="${PREFIX}_${suffix}"
  local missing=() verb source_file output_file stimulus_rows

  [ -d "${input}" ] || {
    echo "Missing input directory: ${input}" >&2
    exit 2
  }

  for verb in "$@"; do
    stimulus_rows="$(awk -F, -v run="evalv3_${suffix}" -v verb="${verb}" \
      'NR > 1 && $4 == run && $6 == verb {count++} END {print count + 0}' \
      Survey/sampled_stimuli.csv)"
    [ "${stimulus_rows}" -eq 4 ] || {
      echo "Frozen AAT manifest mismatch for evalv3_${suffix}/${verb}: " \
           "expected four condition rows, found ${stimulus_rows}" >&2
      exit 2
    }

    source_file="${input}/${verb}.json"
    [ -s "${source_file}" ] || {
      echo "Missing AAT generation input: ${source_file}" >&2
      exit 2
    }
    validate_generation "${source_file}" || {
      echo "Invalid AAT generation input (requires success + parsed): ${source_file}" >&2
      exit 2
    }
    PLANNED_COUNT=$((PLANNED_COUNT + 1))

    output_file="results/evaluations/${name}/${verb}.json"
    if [ -s "${output_file}" ] && "${PYTHON}" -c \
      'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("success") else 1)' \
      "${output_file}"; then
      echo "KEEP existing successful output: ${output_file}"
    else
      missing+=("${verb}")
    fi
  done

  if [ "${MODE}" = "plan" ]; then
    echo "PLAN ${name}: $*"
    return 0
  fi
  if [ "${#missing[@]}" -eq 0 ]; then
    return 0
  fi

  "${PYTHON}" scripts/run_evaluator.py \
    --input-dir "${input}" \
    --provider openai \
    --model "${MODEL}" \
    --reasoning-effort high \
    --temperature 0.0 \
    --max-tokens 24576 \
    --eval-prompt "${PROMPT}" \
    --criterion text \
    --run-name "${name}" \
    --verbs "${missing[@]}" \
    --no-aggregate
}

if [ "${MODE}" = "smoke" ]; then
  run_eval gpt-5.1-2025-11-13_v4 \
    results/intermediate/gpt-5.1-2025-11-13_v4 paint
  echo "Smoke output is isolated under ${PREFIX}_* and is not an AAT replicate."
  exit 0
fi

# The visual, surprisal, and text-only workflows use the same frozen 30 sets.
while IFS=$'\t' read -r suffix directory verb_list; do
  read -r -a verbs <<< "${verb_list}"
  run_eval "${suffix}" "${directory}" "${verbs[@]}"
done < "${SCRIPT_DIR}/survey_items.tsv"

if [ "${PLANNED_COUNT}" -ne 30 ]; then
  echo "Expected exactly 30 AAT sets, found ${PLANNED_COUNT}" >&2
  exit 2
fi

if [ "${MODE}" = "plan" ]; then
  echo "Structural plan passed: 30/30 fixed inputs are valid; no API calls were made."
else
  echo "Replicate ${RUN} complete or resumed: outputs are under results/evaluations/${PREFIX}_*."
fi
