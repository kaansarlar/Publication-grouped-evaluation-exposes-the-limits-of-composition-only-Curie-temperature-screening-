#!/usr/bin/env python3
"""Recalculate external rule validation overall and by chemistry/experiment strata."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from sklearn.metrics import confusion_matrix, matthews_corrcoef


RULES = {
    "250–350 K": ("250–350", 250.0, 350.0),
    "280–320 K strict": ("Strict 280–320", 280.0, 320.0),
    "280–320 K VEC–δ core": ("VEC–δ core", 280.0, 320.0),
    "300–400 K": ("300–400", 300.0, 400.0),
}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    den = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def parse_external_table(path: Path) -> pd.DataFrame:
    doc = Document(path)
    target = None
    for table in doc.tables:
        headers = [c.text.strip() for c in table.rows[0].cells]
        if headers and headers[0] == "Composition" and "Reported Tc / transition (K)" in headers:
            target = table
            break
    if target is None:
        raise RuntimeError("External validation table not found")
    headers = [c.text.strip() for c in target.rows[0].cells]
    records = []
    for row in target.rows[1:]:
        vals = [c.text.strip().replace("\n", " ") for c in row.cells]
        records.append(dict(zip(headers, vals)))
    df = pd.DataFrame(records)
    df["TC_K"] = pd.to_numeric(df["Reported Tc / transition (K)"], errors="raise")
    df["is_simulation"] = df["Transition basis"].str.contains("simulation", case=False, na=False)
    df["chemistry_stratum"] = np.where(
        df["is_simulation"],
        "Simulation sensitivity records",
        np.where(
            df["Composition"].str.startswith("Mn"),
            "Mn–Ni–Si intermetallic OOD stress test",
            "Experimental HEA/MEA/CCA domain",
        ),
    )
    df["validation_role"] = np.where(
        df["is_simulation"],
        "Simulation sensitivity",
        np.where(
            df["Composition"].str.startswith("Mn"),
            "Out-of-domain intermetallic stress test",
            "Primary in-domain HEA/MEA/CCA control",
        ),
    )
    df["experiment_stratum"] = np.where(df["is_simulation"], "Simulation", "Experimental")
    for label, (column, low, high) in RULES.items():
        df[f"pass::{label}"] = df[column].str.casefold().eq("pass")
        df[f"positive::{label}"] = df["TC_K"].between(low, high, inclusive="both")
    return df


def load_external_csv(path: Path) -> pd.DataFrame:
    """Load the public 32-row audit table distributed with the repository."""
    df = pd.read_csv(path)
    required = {"TC_K", "chemistry_stratum"}
    for label, (column, low, high) in RULES.items():
        required.update({f"pass::{label}", f"positive::{label}"})
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Historical audit table is missing required columns: {missing}")
    for label in RULES:
        for prefix in ("pass", "positive"):
            name = f"{prefix}::{label}"
            if df[name].dtype != bool:
                df[name] = df[name].astype(str).str.strip().str.casefold().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
    return df


def metric_row(frame: pd.DataFrame, rule: str, stratum_type: str, stratum: str) -> dict:
    y = frame[f"positive::{rule}"].to_numpy(dtype=bool)
    pred = frame[f"pass::{rule}"].to_numpy(dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False, True]).ravel()
    n = len(frame)
    positive_n = int(y.sum())
    selected_n = int(pred.sum())
    precision = tp / selected_n if selected_n else math.nan
    recall = tp / positive_n if positive_n else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    ba_parts = [x for x in (recall, specificity) if not math.isnan(x)]
    balanced_accuracy = sum(ba_parts) / len(ba_parts) if len(ba_parts) == 2 else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else math.nan
    mcc = matthews_corrcoef(y, pred) if len(np.unique(y)) == 2 and len(np.unique(pred)) == 2 else math.nan
    prevalence = positive_n / n if n else math.nan
    enrichment = precision / prevalence if selected_n and prevalence > 0 else math.nan
    p_lo, p_hi = wilson(int(tp), selected_n)
    r_lo, r_hi = wilson(int(tp), positive_n)
    s_lo, s_hi = wilson(int(tn), int(tn + fp))
    return {
        "stratum_type": stratum_type,
        "stratum": stratum,
        "rule": rule,
        "n": n,
        "n_positive": positive_n,
        "n_selected": selected_n,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "precision": precision,
        "precision_ci_low": p_lo,
        "precision_ci_high": p_hi,
        "recall": recall,
        "recall_ci_low": r_lo,
        "recall_ci_high": r_hi,
        "specificity": specificity,
        "specificity_ci_low": s_lo,
        "specificity_ci_high": s_hi,
        "balanced_accuracy": balanced_accuracy,
        "F1": f1,
        "MCC": mcc,
        "prevalence": prevalence,
        "enrichment": enrichment,
    }


def calculate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets: list[tuple[str, str, pd.DataFrame]] = [
        ("overall", "All external records", df),
    ]
    role_order = [
        "Experimental HEA/MEA/CCA domain",
        "Mn–Ni–Si intermetallic OOD stress test",
        "Simulation sensitivity records",
    ]
    for name in role_order:
        subsets.append(("validation_role", name, df.loc[df["chemistry_stratum"].eq(name)]))
    for stype, name, part in subsets:
        for rule in RULES:
            rows.append(metric_row(part, rule, stype, name))
    return pd.DataFrame(rows)


def plot_primary(metrics: pd.DataFrame, out: Path) -> None:
    rule = "250–350 K"
    show = metrics.loc[(metrics["rule"] == rule) & metrics["stratum"].isin([
        "All external records", "Experimental HEA/MEA/CCA domain",
        "Mn–Ni–Si intermetallic OOD stress test", "Simulation sensitivity records"
    ])].copy()
    order = ["All external records", "Experimental HEA/MEA/CCA domain",
             "Mn–Ni–Si intermetallic OOD stress test", "Simulation sensitivity records"]
    show["stratum"] = pd.Categorical(show["stratum"], order, ordered=True)
    show = show.sort_values("stratum")
    labels = ["Pooled\n(n=32)", "HEA/MEA/CCA\n(n=14)", "Mn intermetallic OOD\n(n=15)", "Simulation\n(n=3)"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    x = np.arange(len(show))
    selected_rate = show["n_selected"] / show["n"]
    prevalence = show["n_positive"] / show["n"]
    width = 0.36
    axes[0].bar(x - width / 2, selected_rate, width, label="Rule-selected fraction", color="#2E6F9E")
    axes[0].bar(x + width / 2, prevalence, width, label="250–350 K prevalence", color="#C47A2C")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Fraction of stratum")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_title("Selection rate versus target prevalence")
    axes[0].legend(frameon=False, fontsize=8)
    specificity = show["specificity"]
    axes[1].bar(x, specificity, color=["#8A8A8A", "#4C8C6B", "#C47A2C", "#7B6EA8"])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Specificity")
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_title("Rejection of outside-window records")
    axes[1].text(1, 0.52, "No target-positive\nHEA/MEA/CCA cases", ha="center", va="center", fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
    fig.suptitle("External 250–350 K rule by validation role", y=1.02)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/historical_audit_32.csv",
    )
    parser.add_argument(
        "--manuscript",
        type=Path,
        help="Optional legacy path: parse the historical table directly from a manuscript DOCX.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = parse_external_table(args.manuscript) if args.manuscript else load_external_csv(args.data)
    metrics = calculate(df)
    df.to_csv(args.output_dir / "external_records_with_strata.csv", index=False)
    metrics.to_csv(args.output_dir / "external_stratified_metrics.csv", index=False)
    plot_primary(metrics, args.output_dir / "external_250_350_stratified")
    primary = metrics.loc[metrics["rule"].eq("250–350 K")]
    print(primary[["stratum", "n", "TP", "TN", "FP", "FN", "balanced_accuracy", "enrichment"]].to_string(index=False))


if __name__ == "__main__":
    main()
