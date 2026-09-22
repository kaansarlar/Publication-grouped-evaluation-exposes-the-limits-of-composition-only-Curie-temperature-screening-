#!/usr/bin/env python3
"""Convenience runner for verification, figure regeneration, or the full analysis."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PYTHON = sys.executable


def run(*parts: str | Path) -> None:
    command = [str(part) for part in parts]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify() -> None:
    run(PYTHON, SCRIPTS / "verify_package.py")


def figures() -> None:
    run(PYTHON, SCRIPTS / "development_window_rules.py")
    run(PYTHON, SCRIPTS / "make_rule_figures.py")


def full(n_jobs: int) -> None:
    grouped = ROOT / "reproduced/grouped_cv"
    interpretation = ROOT / "reproduced/interpretation"
    nested = ROOT / "reproduced/rule_reassessment"
    historical = ROOT / "reproduced/historical_audit"
    external = ROOT / "reproduced/external_rules"

    run(PYTHON, SCRIPTS / "publication_grouped_analysis.py",
        "--data", ROOT / "data/processed/development_unique_141.csv",
        "--full-data", ROOT / "data/processed/development_full_199.csv",
        "--output-dir", grouped, "--candidate-budget", "60", "--n-jobs", str(n_jobs))
    run(PYTHON, SCRIPTS / "final_interpretation_and_figures.py",
        "--results-dir", grouped,
        "--old-ablation", ROOT / "data/raw/record_random_ablation_reference.csv",
        "--output-dir", interpretation, "--candidate-budget", "60", "--n-jobs", str(n_jobs))
    run(PYTHON, SCRIPTS / "development_window_rules.py")
    run(PYTHON, SCRIPTS / "historical_audit_metrics.py",
        "--data", ROOT / "data/processed/historical_audit_32.csv",
        "--output-dir", historical)
    run(PYTHON, SCRIPTS / "nested_rule_selection.py",
        "--data", grouped / "TC_unique_141_with_composition_and_reference.csv",
        "--output-dir", nested, "--n-jobs", str(n_jobs))
    run(PYTHON, SCRIPTS / "evaluate_frozen_rules.py", "--rule-file", nested / "final_v2_rule.json",
        "--output-dir", external)

    nested.mkdir(parents=True, exist_ok=True)
    for name in ["historical_32_v1_v2_predictions.csv", "targeted_challenge_v1_v2_predictions.csv",
                 "external_v1_v2_metrics.csv"]:
        shutil.copy2(external / name, nested / name)
    run(PYTHON, SCRIPTS / "make_rule_figures.py", "--input-dir", nested,
        "--output-dir", ROOT / "reproduced/figures")
    verify()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify", "figures", "full"], default="verify")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "verify":
        verify()
    elif args.mode == "figures":
        figures()
    else:
        full(args.n_jobs)


if __name__ == "__main__":
    main()
