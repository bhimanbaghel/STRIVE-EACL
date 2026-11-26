# Human Annotation Survey Analysis

The paper's study contains 30 stimulus sets rated by eight annotators. The
agreement analysis joins their de-identified responses with evaluator judgments
in the repository's `results/` directory.

## Setup

From the repository root, install the experiment dependencies.

```bash
uv sync --locked
cd Survey
```

Run the commands below from `Survey/`, except where specified.

## Reproduce the primary evaluator analysis

Run the paper's GPT-5.1 high-reasoning analysis and generate its survey plots.
The fixed Python hash seed makes the per-condition binary bootstrap reproducible.

```bash
PYTHONHASHSEED=0 uv run --locked python -m analysis.main --eval-prefix evalv31_gpt51_highreasoning
uv run --locked python generate_figures.py --eval-prefix evalv31_gpt51_highreasoning
```

The analysis writes `outputs/analysis_results_evalv31_gpt51_highreasoning.json`.
The rating-distribution plot is
`outputs/figures/figures_evalv31_gpt51_highreasoning/fig_section_AB_rating_distributions.pdf`.

## Rerun the visual evaluator

For fresh visual judgments, run the shared launcher from the repository root.

```bash
./slurm_jobs/run.sh survey --judge gpt51 --mode thinking
./slurm_jobs/run.sh survey --judge qwen4b --mode thinking --submit
```

The launcher uses all 30 historical generation JSONs. Four are bundled under
`results/aat_inputs/`. Supply the other 26 original files at the paths in
`slurm_jobs/survey_items.tsv`. That file also defines the inputs for the
sentence-only AAT and the three surprisal variants.

The scaling table uses evalv3.1 for GPT-5.1 and Sonnet 4.6 and evalv4 for
GPT-5.4-mini and the Qwen judges. Select `--mode greedy` or `--mode thinking`
for each reported configuration. Qwen3-Next-80B uses greedy mode.

## Reproduce the human plausibility tests

Average the eight ratings for each sentence, then compare the four conditions
within each of the 30 matched sets. The Wilcoxon tests are two-sided and use
Holm correction across the three adjacent pairs.

```bash
uv run --locked python - <<'PY'
from scipy.stats import friedmanchisquare, wilcoxon

from analysis.config import COND_ORDER, QUALTRICS_CSV, STIMULI_CSV
from analysis.data_loading import build_question_map, extract_responses, load_qualtrics, load_stimuli

responses, question_labels = load_qualtrics(QUALTRICS_CSV)
items = load_stimuli(STIMULI_CSV)
question_map = build_question_map(items, question_labels)
ratings, *_ = extract_responses(responses, question_map, items)
means = ratings.groupby(["item", "condition"])["rating"].mean().unstack()
scores = [means[condition].to_numpy() for condition in COND_ORDER]
labels = ("PE", "PH", "IH", "IE")

print("Mean ratings", ", ".join(f"{label}={score.mean():.2f}" for label, score in zip(labels, scores)))
friedman = friedmanchisquare(*scores)
print(f"Friedman chi-square(3)={friedman.statistic:.1f}, p={friedman.pvalue:.3g}")

p_values = [
    wilcoxon(left, right, alternative="two-sided", method="asymptotic").pvalue
    for left, right in zip(scores, scores[1:])
]
adjusted = [0.0] * len(p_values)
largest_adjusted = 0.0
for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
    largest_adjusted = max(largest_adjusted, (len(p_values) - rank) * p_values[index])
    adjusted[index] = min(1.0, largest_adjusted)
for left, right, p_value in zip(labels, labels[1:], adjusted):
    print(f"{left}-{right} Holm-adjusted p={p_value:.3g}")
PY
```

## Compare earlier evaluator runs

```bash
PYTHONHASHSEED=0 uv run --locked python -m analysis.main --eval-prefix evalv3
PYTHONHASHSEED=0 uv run --locked python -m analysis.main --eval-prefix evalv4
uv run --locked python -m analysis.comparison --prefixes evalv3 evalv4
```

## Reproduce the revised sentence-only evaluator AAT

Score the paper's three saved GPT-5.1 runs using the bundled evaluation CSVs.

```bash
for run in 1 2 3; do
  prefix="evalv31agent_textonly_rolev2_gpt51-r${run}"
  uv run --locked python -m analysis.main --eval-prefix "$prefix" \
    --eval-csv "../results/aggregate/aat_evalv31agent_textonly_rolev2_gpt51_r${run}.csv" \
    --plausibility-only
done
uv run --locked python ../scripts/summarize_aat_replicates.py \
  --results-dir outputs \
  --template 'analysis_results_evalv31agent_textonly_rolev2_gpt51-r{run}.json' \
  --runs 1 2 3
```

## Run fresh text-only evaluator jobs

Use the fixed 30 human-rated sets (120 sentences) to validate an
evaluator that intentionally omits visual judgments. The default analysis
mode above is unchanged; `--plausibility-only` skips visual AAT and visual
exports while retaining the primary strict four-point AAT.

From the project root, first run the structural check. It validates the prompt
and all 30 frozen generation inputs without making an API call or writing an
evaluation:

```bash
bash slurm_jobs/eval_annotation_textonly_aat_replicate.sh plan
```

Then run one isolated one-item smoke. Its `-smoke_` output is not included in
the AAT decision:

```bash
bash slurm_jobs/eval_annotation_textonly_aat_replicate.sh smoke
```

After inspecting the smoke result, freeze the prompt and run five independent
replicates. Every run name and derived CSV is text-only-specific; the scripts
do not modify historical evaluator JSONs or the shared merged preview CSV.

```bash
for run in 1 2 3 4 5; do
  bash slurm_jobs/eval_annotation_textonly_aat_replicate.sh replicate "$run"
  bash slurm_jobs/score_annotation_textonly_aat_replicate.sh "$run"
done
```

Apply the predeclared acceptance rule:

```bash
.venv/bin/python scripts/summarize_aat_replicates.py
```

The primary metric is strict four-point unweighted Cohen's kappa. A replicate
passes when its 10,000-resample bootstrap fifth percentile of
`mean(L-H kappa) - mean(H-H kappa)` is greater than `-0.05`; the evaluator is
accepted when at least 3 of 5 independent replicates pass. Missing or malformed
replicate files produce an `INCOMPLETE` decision rather than being silently
excluded.

To test a prompt at another path without editing the launcher, set
`AAT_EVAL_PROMPT`. To keep experimental families separate, set
`AAT_PREFIX_ROOT`; use the same value for evaluation and scoring commands.

## Inputs

| File | Purpose |
|---|---|
| `sampled_stimuli.csv` | 30 survey items mapped to eval runs |
| `annotation_responses.csv` | De-identified responses from the eight-annotator study, with question labels and completion durations |
| `../results/aggregate/all_evaluations_merged_preview.csv` | LLM evaluator judgments |

## Outputs (gitignored under `outputs/`)

| Path | Contents |
|---|---|
| `outputs/analysis_results_{prefix}.json` | Full analysis results |
| `outputs/exports/sentence_examples_{prefix}.csv` | Per-sentence export |
| `outputs/exports/sentence_comparison_*.csv` | Evaluator version comparisons |
| `outputs/figures/figures_{prefix}/` | Generated plots |
