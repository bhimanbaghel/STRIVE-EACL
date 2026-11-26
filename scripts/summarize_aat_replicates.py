#!/usr/bin/env python3
"""Summarize the primary AAT decision across independent evaluator runs.

The acceptance rule is intentionally narrow and paper-facing: a replicate
passes when the strict four-point *unweighted* Cohen's-kappa bootstrap fifth
percentile is greater than ``-margin``.  The evaluator is accepted when at
least ``min_passes`` of the expected replicates pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TEMPLATE = "analysis_results_evalv31agent_textonly_gpt51-r{run}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the primary strict four-point unweighted-kappa AAT rule."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Survey/outputs"),
        help="Directory containing analysis_results_*.json files",
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help="Filename template containing {run}",
    )
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-passes", type=int, default=3)
    parser.add_argument(
        "--expected-sentences",
        type=int,
        default=120,
        help="Require this many valid sentence judgments per replicate",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable summary (never overwrites unless --force)",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing --output-json")
    return parser.parse_args()


def load_primary(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    primary = data["strict_4point_aat"]["unweighted_kappa"]
    bootstrap = primary["bootstrap"]
    return {
        "path": str(path),
        "n_sentences": int(data["strict_4point_aat"]["n_sentences"]),
        "n_valid_llm": int(data["strict_4point_aat"]["n_valid_llm"]),
        "hh_mean": float(primary["hh_mean"]),
        "lh_mean": float(primary["lh_mean"]),
        "delta": float(primary["delta"]),
        "bootstrap_5th_pctile": float(bootstrap["bootstrap_5th_pctile"]),
    }


def main() -> int:
    args = parse_args()
    if "{run}" not in args.template:
        raise SystemExit("--template must contain {run}")
    if args.margin <= 0:
        raise SystemExit("--margin must be positive")
    if args.expected_sentences <= 0:
        raise SystemExit("--expected-sentences must be positive")
    if not 1 <= args.min_passes <= len(args.runs):
        raise SystemExit("--min-passes must be between 1 and the number of runs")
    if len(set(args.runs)) != len(args.runs):
        raise SystemExit("--runs contains duplicates")

    missing = []
    malformed = []
    records = []
    for run in args.runs:
        path = args.results_dir / args.template.format(run=run)
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            record = load_primary(path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            malformed.append(f"{path}: {exc}")
            continue
        if (
            record["n_sentences"] != args.expected_sentences
            or record["n_valid_llm"] != args.expected_sentences
        ):
            malformed.append(
                f"{path}: expected {args.expected_sentences}/{args.expected_sentences} "
                f"valid judgments, found {record['n_valid_llm']}/{record['n_sentences']}"
            )
            continue
        record["run"] = run
        record["passed"] = record["bootstrap_5th_pctile"] > -args.margin
        records.append(record)

    if missing or malformed:
        for path in missing:
            print(f"MISSING: {path}")
        for detail in malformed:
            print(f"MALFORMED: {detail}")
        print("AAT decision: INCOMPLETE (all requested replicates are required)")
        return 2

    print("Primary AAT: strict four-point unweighted Cohen's kappa")
    print(f"Rule: bootstrap 5th percentile > {-args.margin:.4f}")
    print(f"{'run':>4} {'valid':>9} {'H-H':>9} {'L-H':>9} {'delta':>9} {'boot5':>9} result")
    for record in records:
        result = "PASS" if record["passed"] else "FAIL"
        print(
            f"{record['run']:>4} "
            f"{record['n_valid_llm']:>3}/{record['n_sentences']:<5} "
            f"{record['hh_mean']:>9.4f} {record['lh_mean']:>9.4f} "
            f"{record['delta']:>+9.4f} {record['bootstrap_5th_pctile']:>9.4f} {result}"
        )

    passes = sum(record["passed"] for record in records)
    accepted = passes >= args.min_passes
    decision = "ACCEPT" if accepted else "REJECT"
    print(f"AAT decision: {decision} ({passes}/{len(records)} pass; need {args.min_passes})")

    summary = {
        "metric": "strict_4point_unweighted_cohens_kappa",
        "bootstrap_percentile": 5,
        "noninferiority_margin": args.margin,
        "pass_rule": f"bootstrap_5th_pctile > {-args.margin}",
        "minimum_passes": args.min_passes,
        "passes": passes,
        "n_runs": len(records),
        "accepted": accepted,
        "runs": records,
    }
    if args.output_json:
        if args.output_json.exists() and not args.force:
            raise SystemExit(
                f"Refusing to overwrite {args.output_json}; use --force or a new path"
            )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {args.output_json}")

    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
