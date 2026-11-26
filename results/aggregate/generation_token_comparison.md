# Generation output token comparison

Compares **(w/oRe+ST)** baselines to **(wRes+ST)** single-shot reasoning and **(wRe+ST)** iterative runs. Per-verb counts in `results/aggregate/generation_token_counts.csv` come from the saved calls' `token_info` fields.

**Iterative runs:** generator totals sum `iter_*_generation.json` and `iter_*_refinement.json` only (GPT evaluator `iter_*_evaluation.json` excluded). For R2R/R2VR, cross-strategy comparisons use **output ÷ n_iterations** (mean per round from `history.json`), so multi-round runs are not inflated vs single-shot.

---

## 1. Output tokens by model × strategy (totals per verb)

Per verb: **sum** of all generator calls in that experiment. Iterative rows also show mean rounds and per-iteration average.

| Model | Strategy | Label | n | Mean total out | Median total | Mean iters | Mean out/iter | Mean input |
|-------|----------|-------|---|----------------|--------------|------------|-------------|------------|
| gpt-5.1 | v7_60 | RB (w/oRe+ST) | 60 | 610 | 611 | 1.0 | 610 | 5,404 |
| gpt-5.1 | v7_agent_60 | RB (wRes+ST) | 60 | 2,633 | 2,576 | 1.0 | 2,633 | 6,181 |
| gpt-5.1 | v7_verbalized_60 | R2V (w/oRe+ST) | 60 | 2,147 | 2,126 | 1.0 | 2,147 | 5,579 |
| gpt-5.1 | v7_agent_verbalized_60 | R2V (wRes+ST) | 60 | 4,819 | 4,734 | 1.0 | 4,819 | 8,374 |
| gpt-5.1 | v7_agent_iterative_60 | R2R (wRe+ST) | 60 | 7,161 | 5,548 | 2.6 | 2,634 | 33,809 |
| gpt-5.1 | v7_agent_verbalized_iterative_60 | R2VR (wRe+ST) | 60 | 9,759 | 9,992 | 2.8 | 3,702 | 41,947 |
| sonnet-4.6 | v7_60 | RB (w/oRe+ST) | 60 | 685 | 688 | 1.0 | 685 | 35 |
| sonnet-4.6 | v7_agent_60 | RB (wRes+ST) | 60 | 4,421 | 3,884 | 1.0 | 4,421 | 33 |
| sonnet-4.6 | v7_verbalized_60 | R2V (w/oRe+ST) | 60 | 2,424 | 2,403 | 1.0 | 2,424 | 35 |
| sonnet-4.6 | v7_agent_verbalized_60 | R2V (wRes+ST) | 60 | 6,795 | 6,309 | 1.0 | 6,795 | 34 |
| sonnet-4.6 | v7_agent_iterative_60 | R2R (wRe+ST) | 55 | 15,531 | 11,570 | 2.6 | 5,958 | 37,882 |
| sonnet-4.6 | v7_agent_verbalized_iterative_60 | R2VR (wRe+ST) | 60 | 15,276 | 13,280 | 2.4 | 6,525 | 35,832 |
| Qwen3-Next-80B | v7_60 | RB (w/oRe+ST) | 60 | 557 | 556 | 1.0 | 557 | 5,330 |
| Qwen3-Next-80B | v7_agent_60 | RB (wRes+ST) | 60 | 1,645 | 1,640 | 1.0 | 1,645 | 6,074 |
| Qwen3-Next-80B | v7_verbalized_60 | R2V (w/oRe+ST) | 60 | 2,046 | 2,043 | 1.0 | 2,046 | 5,566 |
| Qwen3-Next-80B | v7_agent_verbalized_60 | R2V (wRes+ST) | 60 | 3,083 | 3,071 | 1.0 | 3,083 | 8,392 |
| Qwen3-30B | v7_60 | RB (w/oRe+ST) | 60 | 597 | 598 | 1.0 | 597 | 5,330 |
| Qwen3-30B | v7_agent_60 | RB (wRes+ST) | 60 | 1,659 | 1,606 | 1.0 | 1,659 | 6,074 |
| Qwen3-30B | v7_verbalized_60 | R2V (w/oRe+ST) | 60 | 1,921 | 1,906 | 1.0 | 1,921 | 5,566 |
| Qwen3-30B | v7_agent_verbalized_60 | R2V (wRes+ST) | 60 | 2,982 | 2,974 | 1.0 | 2,982 | 8,392 |
| Ministral-14B | v7_60 | RB (w/oRe+ST) | 60 | 618 | 620 | 1.0 | 618 | 5,599 |
| Ministral-14B | v7_agent_60 | RB (wRes+ST) | 60 | 1,663 | 1,638 | 1.0 | 1,663 | 6,388 |
| Ministral-14B | v7_verbalized_60 | R2V (w/oRe+ST) | 60 | 1,921 | 1,898 | 1.0 | 1,921 | 5,824 |
| Ministral-14B | v7_agent_verbalized_60 | R2V (wRes+ST) | 60 | 2,900 | 2,911 | 1.0 | 2,900 | 8,764 |
| Qwen3-4B | v7_60 | RB (w/oRe+ST) | 60 | 636 | 636 | 1.0 | 636 | 5,330 |
| Qwen3-4B | v7_agent_60 | RB (wRes+ST) | 60 | 1,821 | 1,672 | 1.0 | 1,821 | 6,074 |
| Qwen3-4B | v7_verbalized_60 | R2V (w/oRe+ST) | 60 | 2,142 | 2,134 | 1.0 | 2,142 | 5,566 |
| Qwen3-4B | v7_agent_verbalized_60 | R2V (wRes+ST) | 60 | 3,159 | 3,068 | 1.0 | 3,159 | 8,392 |

---

## 2. Extra output tokens vs (w/oRe+ST) baseline (fair comparison)

Paired on the same **(model, verb)**.

- **RB / R2V:** single-shot total output (one API call).
- **R2R / R2VR:** **per-iteration** generator output (total ÷ `total_iterations`), compared to single-shot baseline.

| Modality | Baseline | With reasoning / iterative | n pairs | Mean Δ | Median Δ | Mean ratio | % increase |
|----------|----------|---------------------------|---------|--------|----------||------------|------------|
| RB | RB (w/oRe+ST) | RB (wRes+ST) | 360 | +1,690 | +1,136 | 3.74× | +273.9% |
| R2V | R2V (w/oRe+ST) | R2V (wRes+ST) | 360 | +1,856 | +1,136 | 1.88× | +88.4% |
| R2R | RB (w/oRe+ST) | R2R (wRe+ST) | 115 | +3,578 | +2,416 | 6.54× | +553.5% |
| R2VR | R2V (w/oRe+ST) | R2VR (wRe+ST) | 120 | +2,828 | +2,210 | 2.24× | +123.8% |

### 2b. Mean Δ by model (normalized where applicable)

| Modality | Model | n pairs | Mean Δ | Median Δ | Mean iters (with side) |
|----------|-------|---------|--------|----------|------------------------|
| RB | gpt-5.1 | 60 | +2,023 | +1,990 | 1.0 |
| RB | sonnet-4.6 | 60 | +3,736 | +3,186 | 1.0 |
| RB | Qwen3-Next-80B | 60 | +1,089 | +1,089 | 1.0 |
| RB | Qwen3-30B | 60 | +1,063 | +1,022 | 1.0 |
| RB | Ministral-14B | 60 | +1,045 | +1,028 | 1.0 |
| RB | Qwen3-4B | 60 | +1,185 | +1,033 | 1.0 |
| R2V | gpt-5.1 | 60 | +2,672 | +2,624 | 1.0 |
| R2V | sonnet-4.6 | 60 | +4,371 | +3,853 | 1.0 |
| R2V | Qwen3-Next-80B | 60 | +1,037 | +1,007 | 1.0 |
| R2V | Qwen3-30B | 60 | +1,061 | +1,066 | 1.0 |
| R2V | Ministral-14B | 60 | +978 | +930 | 1.0 |
| R2V | Qwen3-4B | 60 | +1,017 | +914 | 1.0 |
| R2R | gpt-5.1 | 60 | +2,024 | +1,985 | 2.6 |
| R2R | sonnet-4.6 | 55 | +5,272 | +4,616 | 2.6 |
| R2VR | gpt-5.1 | 60 | +1,556 | +1,403 | 2.8 |
| R2VR | sonnet-4.6 | 60 | +4,101 | +3,398 | 2.4 |

### 2c. Unnormalized iterative totals (reference only)

Same pairs as §2 but R2R/R2VR use **summed** tokens across all rounds (inflated vs single-shot).

| Modality | Mean Δ (total sum) | Median Δ |
|----------|-------------------|----------|
| R2R | +10,518 | +8,184 |
| R2VR | +10,233 | +8,186 |

---

## 3. Pooled summary (normalized comparison)

- **Average additional output tokens (fair comparison):** **+2,123** per paired cell (median +1,411, n=955). R2R/R2VR use per-iteration generator output.

---

## 4. Iterative pipeline totals (generator + GPT evaluator)

For R2R / R2VR only: mean output tokens when **all** iterative API calls are included.

| Model | Strategy | Mean gen-only out | Mean full pipeline out | Eval share |
|-------|----------|-------------------|------------------------|------------|
| gpt-5.1 | R2R (wRe+ST) | 7,161 | 32,505 | 78.0% of pipeline |
| gpt-5.1 | R2VR (wRe+ST) | 9,759 | 36,908 | 73.6% of pipeline |
| sonnet-4.6 | R2R (wRe+ST) | 15,531 | 45,592 | 65.9% of pipeline |
| sonnet-4.6 | R2VR (wRe+ST) | 15,276 | 42,196 | 63.8% of pipeline |
