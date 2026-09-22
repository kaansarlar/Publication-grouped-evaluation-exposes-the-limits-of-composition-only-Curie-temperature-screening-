#!/usr/bin/env python3
"""Create source-grouped manuscript figures and a full-data LightGBM interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.model_selection import GridSearchCV

from publication_grouped_analysis import (
    HEA_FEATURES,
    TARGET,
    grouped_splits,
    model_specs,
    sample_grid,
)


FEATURE_LABELS = {
    "dX": r"$\Delta\chi$",
    "VEC": "VEC",
    "sigma": r"$\delta$",
    "dHmix": r"$\Delta H_{mix}$",
    "dSmix": r"$\Delta S_{mix}$",
}
COLORS = {"hea": "#2E6F9E", "matminer": "#C47A2C", "combined": "#4C8C6B"}


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.7, zorder=0)


def final_fit(data: pd.DataFrame, candidate_budget: int, n_jobs: int):
    X = data[HEA_FEATURES]
    y = data[TARGET].to_numpy()
    groups = data["reference_id"].to_numpy()
    spec = model_specs()["LightGBM"]
    grid = sample_grid(spec["grid"], candidate_budget, 42001)
    cv = grouped_splits(y, groups, 5, 42001)
    search = GridSearchCV(
        spec["pipeline"], grid, scoring="r2", cv=cv, n_jobs=n_jobs, refit=True, error_score="raise"
    )
    search.fit(X, y)
    model = search.best_estimator_.named_steps["regressor"]
    contrib = model.predict(X, pred_contrib=True)
    shap_values = contrib[:, :-1]
    base = contrib[:, -1]
    reconstructed = base + shap_values.sum(axis=1)
    fitted = model.predict(X)
    if not np.allclose(reconstructed, fitted, rtol=1e-5, atol=1e-5):
        raise RuntimeError("LightGBM contribution values do not reconstruct fitted predictions")
    return search, shap_values, base, fitted


def figure_validation(summary: pd.DataFrame, old_ablation: pd.DataFrame, folds: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))

    # a: impact of source grouping on the best model in each descriptor family
    old = old_ablation.loc[old_ablation["mode"].eq("unique")].set_index("feature_set")
    grouped_best = summary.sort_values("oof_r2", ascending=False).groupby("feature_set", as_index=True).first()
    families = ["hea", "matminer", "combined"]
    x = np.arange(3)
    w = 0.36
    axes[0, 0].bar(x - w / 2, [old.loc[f, "oof_r2"] for f in families], w, color="0.75", label="Record-random nested CV")
    axes[0, 0].bar(x + w / 2, [grouped_best.loc[f, "oof_r2"] for f in families], w,
                   color=[COLORS[f] for f in families], label="Publication-grouped nested CV")
    axes[0, 0].set_xticks(x, ["HEA", "Matminer", "Combined"])
    axes[0, 0].set_ylabel(r"Best OOF $R^2$")
    axes[0, 0].set_ylim(0, 0.92)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0, 0].set_title("a  Source grouping reduces optimistic transfer estimates", loc="left", fontsize=10)
    style_axis(axes[0, 0])

    # b: fair model comparison in the primary descriptor family
    hea = summary.loc[summary["feature_set"].eq("hea")].copy()
    order = ["LightGBM", "XGBoost", "GradientBoosting", "RandomForest", "SVR", "Ridge", "DummyMean"]
    hea["model"] = pd.Categorical(hea["model"], order, ordered=True)
    hea = hea.sort_values("model")
    axes[0, 1].bar(np.arange(len(hea)), hea["oof_r2"], color="#2E6F9E", zorder=3)
    axes[0, 1].axhline(0, color="0.25", linewidth=0.8)
    axes[0, 1].set_xticks(np.arange(len(hea)), [str(x) for x in hea["model"]], rotation=35, ha="right")
    axes[0, 1].set_ylabel(r"Publication-grouped OOF $R^2$")
    axes[0, 1].set_ylim(-0.05, 0.68)
    axes[0, 1].set_title("b  Equal-budget model comparison (HEA descriptors)", loc="left", fontsize=10)
    style_axis(axes[0, 1])

    # c: outer-fold variability for leading models
    top_models = ["LightGBM", "XGBoost", "GradientBoosting", "RandomForest"]
    values = [folds.loc[(folds["feature_set"] == "hea") & (folds["model"] == m), "r2"].to_numpy() for m in top_models]
    bp = axes[1, 0].boxplot(values, tick_labels=["LGBM", "XGB", "GBR", "RF"], patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8DB7D5")
    axes[1, 0].axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    axes[1, 0].set_ylabel(r"Outer-fold $R^2$")
    axes[1, 0].set_title("c  Between-publication variability", loc="left", fontsize=10)
    style_axis(axes[1, 0])

    # d: error metrics for the best feature-family models
    gb = grouped_best.loc[families]
    x = np.arange(3)
    axes[1, 1].bar(x - w / 2, gb["oof_mae"], w, color=[COLORS[f] for f in families], label="MAE")
    axes[1, 1].bar(x + w / 2, gb["oof_rmse"], w, color=[COLORS[f] for f in families], alpha=0.45, label="RMSE")
    axes[1, 1].set_xticks(x, ["HEA", "Matminer", "Combined"])
    axes[1, 1].set_ylabel("Error (K)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].set_title("d  Best-model absolute error by descriptor family", loc="left", fontsize=10)
    style_axis(axes[1, 1])

    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_prediction(oof: pd.DataFrame, out: Path):
    y = oof["y_true_TC"].to_numpy()
    pred = oof["y_pred_TC"].to_numpy()
    resid = pred - y
    fig = plt.figure(figsize=(8.2, 7.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], hspace=0.42, wspace=0.36)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    lim_lo = min(0, y.min(), pred.min()) - 10
    lim_hi = max(y.max(), pred.max()) + 20
    axes[0].scatter(y, pred, c=oof["fold"], cmap="viridis", s=25, alpha=0.8, edgecolors="white", linewidths=0.3)
    axes[0].plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="0.2", linewidth=1)
    axes[0].set(xlim=(lim_lo, lim_hi), ylim=(lim_lo, lim_hi), xlabel=r"Experimental $T_C$ (K)", ylabel=r"OOF prediction (K)")
    axes[0].text(0.04, 0.94, "$R^2$ = 0.607\nMAE = 57.7 K\nRMSE = 93.6 K", transform=axes[0].transAxes, va="top")
    axes[0].set_title("a  Publication-grouped parity", loc="left", fontsize=10)

    axes[1].scatter(y, resid, c=np.abs(resid), cmap="magma_r", s=25, alpha=0.85, edgecolors="white", linewidths=0.3)
    axes[1].axhline(0, color="0.25", linewidth=1)
    axes[1].set(xlabel=r"Experimental $T_C$ (K)", ylabel="Prediction residual (K)")
    axes[1].set_title("b  Residual structure", loc="left", fontsize=10)

    bins = [-np.inf, 100, 250, 400, np.inf]
    labels = ["<100", "100–250", "250–400", ">400"]
    bucket = pd.cut(y, bins=bins, labels=labels)
    abs_err = np.abs(resid)
    means = [abs_err[bucket == label].mean() for label in labels]
    counts = [int((bucket == label).sum()) for label in labels]
    axes[2].bar(np.arange(4), means, color="#4C8C6B", width=0.68)
    axes[2].set_xticks(np.arange(4), [f"{label}\n(n={n})" for label, n in zip(labels, counts)])
    axes[2].set(xlabel=r"Experimental $T_C$ range (K)", ylabel="Mean absolute error (K)")
    axes[2].set_title("c  Error by temperature range", loc="left", fontsize=10)
    for ax in axes:
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_shap(data: pd.DataFrame, shap_values: np.ndarray, out: Path):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    # deterministic compact beeswarm-style view
    rng = np.random.default_rng(42)
    for rank, idx in enumerate(order):
        vals = data[HEA_FEATURES[idx]].to_numpy(dtype=float)
        norm = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals) + 1e-12)
        jitter = rng.normal(0, 0.075, len(vals))
        axes[0, 0].scatter(shap_values[:, idx], rank + jitter, c=norm, cmap="coolwarm", s=15, alpha=0.75, linewidths=0)
    axes[0, 0].axvline(0, color="0.4", linewidth=0.8)
    axes[0, 0].set_yticks(np.arange(len(order)), [FEATURE_LABELS[HEA_FEATURES[i]] for i in order])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel(r"TreeSHAP contribution to $T_C$ (K)")
    axes[0, 0].set_title("a  Full-data contribution distribution", loc="left", fontsize=10)
    axes[0, 0].grid(axis="x", color="0.9", linewidth=0.7)
    axes[0, 0].spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(handles=[Line2D([], [], marker="o", linestyle="", color="#3B4CC0", label="low value"),
                               Line2D([], [], marker="o", linestyle="", color="#B40426", label="high value")],
                      frameon=False, fontsize=8, loc="lower right")

    axes[0, 1].barh(np.arange(len(order)), mean_abs[order], color="#2E6F9E")
    axes[0, 1].set_yticks(np.arange(len(order)), [FEATURE_LABELS[HEA_FEATURES[i]] for i in order])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Mean |TreeSHAP| (K)")
    axes[0, 1].set_title("b  Global contribution ranking", loc="left", fontsize=10)
    style_axis(axes[0, 1])

    for ax, idx, letter in zip(axes[1], order[:2], ["c", "d"]):
        values = data[HEA_FEATURES[idx]].to_numpy(dtype=float)
        ax.scatter(values, shap_values[:, idx], c=data[TARGET], cmap="viridis", s=24, alpha=0.8, edgecolors="white", linewidths=0.25)
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_xlabel(FEATURE_LABELS[HEA_FEATURES[idx]])
        ax.set_ylabel(r"TreeSHAP contribution to $T_C$ (K)")
        ax.set_title(f"{letter}  {FEATURE_LABELS[HEA_FEATURES[idx]]} dependence", loc="left", fontsize=10)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return mean_abs, order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--old-ablation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.results_dir / "publication_grouped_model_summary.csv")
    folds = pd.read_csv(args.results_dir / "publication_grouped_fold_metrics.csv")
    old_ablation = pd.read_csv(args.old_ablation)
    data = pd.read_csv(args.results_dir / "TC_unique_141_with_composition_and_reference.csv")
    oof_all = pd.read_csv(args.results_dir / "publication_grouped_oof_predictions.csv")
    oof = oof_all.loc[(oof_all["feature_set"] == "hea") & (oof_all["model"] == "LightGBM")].copy()
    split_audit = pd.read_csv(args.results_dir / "splits__hea__LightGBM.csv")
    fold_by_reference = {}
    for _, row in split_audit.iterrows():
        for reference_id in str(row["test_references"]).split(";"):
            fold_by_reference[reference_id] = int(row["fold"])
    oof["fold"] = oof["reference_id"].map(fold_by_reference)
    if oof["fold"].isna().any():
        raise RuntimeError("Could not map every OOF record to an outer fold")

    search, shap_values, base, fitted = final_fit(data, args.candidate_budget, args.n_jobs)
    pd.DataFrame(shap_values, columns=[f"TreeSHAP::{x}" for x in HEA_FEATURES]).assign(
        base_value_K=base, fitted_TC_K=fitted, observed_TC_K=data[TARGET].to_numpy(), reference_id=data["reference_id"].to_numpy()
    ).to_csv(args.output_dir / "full_data_lightgbm_treeshap_values.csv", index=False)
    (args.output_dir / "full_data_lightgbm_best_params.json").write_text(
        json.dumps({"best_inner_grouped_cv_r2": search.best_score_, "best_params": search.best_params_}, indent=2), encoding="utf-8"
    )

    figure_validation(summary, old_ablation, folds, args.output_dir / "Fig2_source_grouped_validation")
    figure_prediction(oof, args.output_dir / "Fig3_grouped_prediction")
    mean_abs, order = figure_shap(data, shap_values, args.output_dir / "Fig4_HEA_TreeSHAP")
    pd.DataFrame({
        "descriptor": HEA_FEATURES,
        "display_label": [FEATURE_LABELS[x] for x in HEA_FEATURES],
        "mean_abs_TreeSHAP_K": mean_abs,
    }).sort_values("mean_abs_TreeSHAP_K", ascending=False).to_csv(args.output_dir / "HEA_TreeSHAP_importance.csv", index=False)
    print("Best parameters:", search.best_params_)
    print("Inner grouped CV R2:", search.best_score_)
    print("TreeSHAP ranking:", [HEA_FEATURES[i] for i in order])


if __name__ == "__main__":
    main()
