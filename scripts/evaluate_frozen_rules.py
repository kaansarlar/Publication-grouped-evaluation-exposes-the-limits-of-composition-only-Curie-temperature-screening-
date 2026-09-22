#!/usr/bin/env python3
"""Recalculate Table 6 and supporting external-rule metrics from frozen descriptors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


ROOT = Path(__file__).resolve().parents[1]


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.casefold().map({"true": True, "false": False, "1": True, "0": False})


def metric_row(name: str, frame: pd.DataFrame, pred_col: str, label_col: str) -> dict:
    d = frame.dropna(subset=[label_col]).copy()
    y = as_bool(d[label_col]).astype(int).to_numpy()
    pred = as_bool(d[pred_col]).astype(int).to_numpy()
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mcc = matthews_corrcoef(y, pred) if len(np.unique(y)) == 2 and len(np.unique(pred)) == 2 else 0.0
    prevalence = float(y.mean())
    return {
        "set": name, "n": len(y), "positives": int(y.sum()), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "F1": f1, "MCC": float(mcc), "base_rate": prevalence,
        "enrichment": precision / prevalence if prevalence else math.nan,
    }


def apply_bounds(frame: pd.DataFrame, bounds: dict[str, list[float]]) -> pd.Series:
    selected = np.ones(len(frame), dtype=bool)
    for feature, (low, high) in bounds.items():
        if feature not in frame:
            raise KeyError(f"Frozen feature {feature!r} is absent from the evaluation table")
        selected &= frame[feature].between(low, high, inclusive="both").to_numpy()
    return pd.Series(selected, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, default=ROOT / "data/processed/historical_audit_32.csv")
    parser.add_argument("--challenge", type=Path, default=ROOT / "data/processed/targeted_challenge_26.csv")
    parser.add_argument("--rule-file", type=Path, default=ROOT / "results/rule_reassessment/final_v2_rule.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced/external_rules")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    historical = pd.read_csv(args.historical)
    challenge = pd.read_csv(args.challenge)
    bounds = json.loads(args.rule_file.read_text(encoding="utf-8"))["bounds"]
    historical["pass::v2_250_350"] = apply_bounds(historical, bounds)
    challenge["pass::v2_250_350"] = apply_bounds(challenge, bounds)
    rows = [
        metric_row("Historical 32 records, v1 locked", historical, "pass::250–350 K", "positive_v2"),
        metric_row("Historical 32 records, v2 frozen", historical, "pass::v2_250_350", "positive_v2"),
    ]
    experimental = historical.loc[~as_bool(historical["is_simulation"])]
    rows.append(metric_row("Historical experimental records, v2", experimental, "pass::v2_250_350", "positive_v2"))
    for stratum, subset in historical.groupby("chemistry_stratum", sort=False):
        rows.append(metric_row(f"Historical v2: {stratum}", subset, "pass::v2_250_350", "positive_v2"))

    primary = challenge.loc[challenge["cohort"].eq("primary")]
    rows.extend([
        metric_row("Targeted five-paper primary challenge, v1", primary, "rule_pass_direct", "target_250_350"),
        metric_row("Targeted five-paper primary challenge, v2", primary, "pass::v2_250_350", "target_250_350"),
        metric_row("Targeted challenge clean subset, v2", primary.loc[primary["quality_flag"].eq("clean")],
                   "pass::v2_250_350", "target_250_350"),
        metric_row("Multi-transition ambiguity stress, v2", challenge.loc[challenge["cohort"].eq("ambiguity_stress")],
                   "pass::v2_250_350", "target_250_350"),
    ])
    result = pd.DataFrame(rows)
    historical.to_csv(args.output_dir / "historical_32_v1_v2_predictions.csv", index=False)
    challenge.to_csv(args.output_dir / "targeted_challenge_v1_v2_predictions.csv", index=False)
    result.to_csv(args.output_dir / "external_v1_v2_metrics.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
