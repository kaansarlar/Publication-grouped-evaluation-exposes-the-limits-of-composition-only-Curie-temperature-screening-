#!/usr/bin/env python3
"""Recalculate Table 2 and Figure 5 development-set descriptor windows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "250–350 K": {
        "window": (250.0, 350.0),
        "features": ["VEC", "avg_d_valence_electrons", "mag_moment_mean"],
        "percentiles": (5.0, 95.0),
        "color": "#2AA198",
    },
    "280–320 K": {
        "window": (280.0, 320.0),
        "features": ["VEC", "sigma", "dHmix", "dSmix"],
        "percentiles": (10.0, 90.0),
        "color": "#F28E2B",
    },
    "300–400 K": {
        "window": (300.0, 400.0),
        "features": ["VEC", "avg_d_valence_electrons", "mag_moment_mean"],
        "percentiles": (5.0, 95.0),
        "color": "#7B6ED6",
    },
}


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    mcc = matthews_corrcoef(y, pred) if len(np.unique(y)) == 2 and len(np.unique(pred)) == 2 else math.nan
    prevalence = float(y.mean())
    return {
        "n": len(y), "positives": int(y.sum()), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "F1": f1, "MCC": float(mcc), "prevalence": prevalence,
        "enrichment": precision / prevalence if prevalence else math.nan,
    }


def derive(data: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows: list[dict] = []
    bounds_all: dict[str, dict] = {}
    predictions = data[["source_row", "composition", "reference_id", "TC"]].copy()
    for label, rule in RULES.items():
        low, high = rule["window"]
        q_low, q_high = rule["percentiles"]
        y = data["TC"].between(low, high, inclusive="both").astype(int).to_numpy()
        positive = data.loc[y.astype(bool), rule["features"]]
        bounds = {
            feature: [float(np.percentile(positive[feature], q_low)), float(np.percentile(positive[feature], q_high))]
            for feature in rule["features"]
        }
        selected = np.ones(len(data), dtype=bool)
        for feature, (lo, hi) in bounds.items():
            selected &= data[feature].between(lo, hi, inclusive="both").to_numpy()
        row = {
            "target_window": label,
            "window_low_K": low,
            "window_high_K": high,
            "descriptor_family": ";".join(rule["features"]),
            "positive_percentile_low": q_low,
            "positive_percentile_high": q_high,
            **metrics(y, selected.astype(int)),
        }
        rows.append(row)
        bounds_all[label] = {
            "window_K": [low, high],
            "positive_percentile_band": [q_low, q_high],
            "features": rule["features"],
            "bounds_full_precision": bounds,
        }
        predictions[f"target::{label}"] = y.astype(bool)
        predictions[f"selected::{label}"] = selected
    return pd.DataFrame(rows), bounds_all, predictions


def make_figure(data: pd.DataFrame, summary: pd.DataFrame, bounds: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))

    # (a) target windows and rounded descriptor intervals
    ax = axes[0, 0]
    y_positions = {"250–350 K": 2, "280–320 K": 1, "300–400 K": 0}
    for label, rule in RULES.items():
        y = y_positions[label]
        lo, hi = rule["window"]
        ax.barh(y, hi - lo, left=lo, height=0.18, color=rule["color"], alpha=0.9)
        ax.text(lo + 10, y + 0.22, label, color=rule["color"], weight="bold", fontsize=11)
        lines = []
        for feature, (a, b) in bounds[label]["bounds_full_precision"].items():
            shown = "δ" if feature == "sigma" else feature
            lines.append(f"{shown}: {a:.2f}–{b:.2f}")
        ax.annotate("\n".join(lines), xy=(hi, y), xytext=(hi + 80, y), va="center", fontsize=8.5,
                    arrowprops={"arrowstyle": "-", "color": rule["color"]},
                    bbox={"boxstyle": "round,pad=.25", "fc": "white", "ec": rule["color"]})
    ax.set_xlim(0, 620); ax.set_ylim(-0.7, 2.7); ax.set_yticks([])
    ax.set_xlabel(r"Target $T_c$ window (K)")
    ax.set_title(r"Target $T_c$ windows with descriptor rules")
    ax.grid(axis="x", color="0.92")

    # (b) development-set classification metrics
    ax = axes[0, 1]
    order = ["250–350 K", "300–400 K", "280–320 K"]
    s = summary.set_index("target_window").loc[order]
    x = np.arange(len(order)); width = 0.18
    for offset, (col, display, color) in enumerate([
        ("precision", "Precision", "#1f77b4"), ("recall", "Recall", "#ff7f0e"),
        ("F1", "F1", "#2ca02c"), ("MCC", "MCC", "#d62728")
    ]):
        ax.bar(x + (offset - 1.5) * width, s[col], width, label=display, color=color)
    ax.set_xticks(x, order); ax.set_ylim(0, 1); ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title("Rule classification metrics")
    ax.grid(axis="y", color="0.92")

    # (c) the broad 250–350 K descriptor intervals
    ax = axes[1, 0]
    label = "250–350 K"
    feature_order = ["VEC", "avg_d_valence_electrons", "mag_moment_mean"]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    for i, (feature, color) in enumerate(zip(feature_order, colors)):
        lo, hi = bounds[label]["bounds_full_precision"][feature]
        ax.plot([lo, hi], [2 - i, 2 - i], color=color, lw=10, solid_capstyle="round")
        ax.text((lo + hi) / 2, 1.78 - i, f"{lo:.2f}–{hi:.2f}", ha="center", fontsize=9)
    ax.set_yticks([2, 1, 0], feature_order)
    ax.set_xlabel("Descriptor value"); ax.set_title("250–350 K descriptor intervals")
    ax.grid(axis="x", color="0.92")

    # (d) selected versus non-selected records across Tc
    ax = axes[1, 1]
    broad = bounds[label]["bounds_full_precision"]
    selected = np.ones(len(data), dtype=bool)
    for feature, (lo, hi) in broad.items():
        selected &= data[feature].between(lo, hi, inclusive="both").to_numpy()
    rng = np.random.default_rng(42)
    y = selected.astype(float) + rng.normal(0, 0.045, len(data))
    ax.scatter(data.loc[~selected, "TC"], y[~selected], color="0.55", s=22, alpha=0.75, label="Not selected")
    ax.scatter(data.loc[selected, "TC"], y[selected], color="#2AA198", s=28, alpha=0.85, label="Selected by rule")
    ax.axvspan(250, 350, color="#E8C66A", alpha=0.18, label="Target window")
    ax.set_yticks([0, 1], ["Not selected", "Selected"])
    ax.set_xlabel(r"Experimental $T_c$ (K)")
    ax.set_title(r"250–350 K rule selection across the $T_c$ range")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="x", color="0.92")

    for letter, ax in zip("abcd", axes.ravel()):
        ax.text(-0.03, 1.08, f"({letter})", transform=ax.transAxes, fontsize=14, weight="bold", ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/development_unique_141.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced/development_windows")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data)
    summary, bounds, predictions = derive(data)
    summary.to_csv(args.output_dir / "development_window_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "development_window_predictions.csv", index=False)
    (args.output_dir / "development_window_bounds.json").write_text(
        json.dumps(bounds, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figure(data, summary, bounds, args.output_dir / "Fig5_development_window_rules")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
