#!/usr/bin/env python3
"""Publication-grouped nested SHAP-guided rule selection for the 250–350 K target."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/development_unique_141.csv"
OUT = ROOT / "reproduced/rule_reassessment"
FEATURES = [
    "dX", "VEC", "sigma", "dHmix", "dSmix", "nunfilled_mean",
    "mag_moment_mean", "avg_d_valence_electrons", "bandgap_mean",
    "melting_point_mean", "electronegativity_range",
]
ORIGINAL = ["VEC", "avg_d_valence_electrons", "mag_moment_mean"]
PERCENTILE_BANDS = [(0, 100), (2.5, 97.5), (5, 95), (10, 90), (15, 85), (20, 80)]
N_OUTER = 10
N_INNER = 5
N_REPEATS = 10
BASE_SEED = 42017
N_JOBS = max(1, min(8, os.cpu_count() or 1))


def classifier(seed):
    return LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.03,
        max_depth=-1, num_leaves=15, min_child_samples=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=0.1,
        random_state=seed, n_jobs=N_JOBS, verbosity=-1,
    )


def shap_ranking(frame, y, seed):
    model = classifier(seed)
    model.fit(frame[FEATURES], y)
    contributions = model.booster_.predict(frame[FEATURES], pred_contrib=True)
    values = np.abs(contributions[:, :-1]).mean(axis=0)
    ranking = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": values})
    ranking = ranking.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def candidate_sets(ranking):
    ranked = ranking["feature"].tolist()
    candidates = {}
    for k in (2, 3, 4):
        candidates[f"SHAP_top_{k}"] = ranked[:k]
        anchored = ["VEC"] + [f for f in ranked if f != "VEC"][: k - 1]
        candidates[f"SHAP_VEC_anchor_{k}"] = anchored
    candidates["original_electronic_3"] = ORIGINAL
    candidates["HEA_VEC_sigma"] = ["VEC", "sigma"]
    candidates["HEA_VEC_sigma_dHmix_dSmix"] = ["VEC", "sigma", "dHmix", "dSmix"]
    unique = {}
    seen = set()
    for name, feats in candidates.items():
        key = tuple(feats)
        if key not in seen:
            unique[name] = feats
            seen.add(key)
    return unique


def intervals(frame, y, feats, band):
    pos = frame.loc[np.asarray(y).astype(bool), feats]
    lo, hi = band
    return {f: (float(np.percentile(pos[f], lo)), float(np.percentile(pos[f], hi))) for f in feats}


def apply_rule(frame, bounds):
    pred = np.ones(len(frame), dtype=bool)
    for feature, (low, high) in bounds.items():
        x = frame[feature].to_numpy(float)
        pred &= (x >= low) & (x <= high)
    return pred.astype(int)


def metrics(y, pred):
    y = np.asarray(y, int); pred = np.asarray(pred, int)
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    ba = (recall + specificity) / 2
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mcc = matthews_corrcoef(y, pred) if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1 else 0.0
    base = y.mean()
    enrichment = precision / base if base else np.nan
    return {"n":len(y), "positives":int(y.sum()), "TP":tp, "TN":tn, "FP":fp, "FN":fn,
            "precision":precision, "recall":recall, "specificity":specificity,
            "balanced_accuracy":ba, "F1":f1, "MCC":float(mcc),
            "base_rate":float(base), "enrichment":float(enrichment)}


def grouped_splits(y, groups, n_splits, seed):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for train, test in splits:
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            raise RuntimeError(f"Single-class grouped fold at seed {seed}")
    return splits


def choose_candidate(result, objective):
    if objective == "legacy_f1":
        return result.sort_values(
            ["F1", "enrichment", "recall", "precision", "MCC", "n_features", "q_low"],
            ascending=[False, False, False, False, False, True, True],
        ).iloc[0]
    eligible = result[result["recall"] >= 0.60].copy()
    if eligible.empty:
        eligible = result.copy()
    return eligible.sort_values(
        ["MCC", "balanced_accuracy", "enrichment", "recall", "precision", "n_features", "q_low"],
        ascending=[False, False, False, False, False, True, True],
    ).iloc[0]


def select_candidate(frame, y, groups, candidates, seed, n_splits=N_INNER, objective="robust_mcc"):
    rows = []
    splits = grouped_splits(y, groups, n_splits, seed)
    for name, feats in candidates.items():
        for band in PERCENTILE_BANDS:
            oof = np.full(len(frame), -1, int)
            for train, valid in splits:
                bounds = intervals(frame.iloc[train], y[train], feats, band)
                oof[valid] = apply_rule(frame.iloc[valid], bounds)
            m = metrics(y, oof)
            rows.append({"candidate":name, "features":";".join(feats), "n_features":len(feats),
                         "q_low":band[0], "q_high":band[1], **m})
    result = pd.DataFrame(rows)
    chosen = choose_candidate(result, objective)
    return chosen, result


def evaluate_repeat(data, repeat):
    y = data["target"].to_numpy(int)
    groups = data["reference_id"].astype(str).to_numpy()
    outer = grouped_splits(y, groups, N_OUTER, BASE_SEED + 1000 * repeat)
    pred_v2 = np.full(len(data), -1, int)
    pred_v2_f1 = np.full(len(data), -1, int)
    pred_v1 = np.full(len(data), -1, int)
    fold_rows, selection_rows, shap_rows = [], [], []
    for fold, (train, test) in enumerate(outer, 1):
        train_df, test_df = data.iloc[train], data.iloc[test]
        y_train, y_test = y[train], y[test]
        ranking = shap_ranking(train_df, y_train, BASE_SEED + repeat * 100 + fold)
        ranking["repeat"] = repeat; ranking["outer_fold"] = fold
        shap_rows.append(ranking)
        candidates = candidate_sets(ranking)
        chosen, all_inner = select_candidate(
            train_df, y_train, groups[train], candidates,
            BASE_SEED + repeat * 10000 + fold,
        )
        chosen_f1 = choose_candidate(all_inner, "legacy_f1")
        feats = chosen["features"].split(";")
        band = (float(chosen["q_low"]), float(chosen["q_high"]))
        bounds = intervals(train_df, y_train, feats, band)
        pred_v2[test] = apply_rule(test_df, bounds)
        feats_f1 = chosen_f1["features"].split(";")
        band_f1 = (float(chosen_f1["q_low"]), float(chosen_f1["q_high"]))
        bounds_f1 = intervals(train_df, y_train, feats_f1, band_f1)
        pred_v2_f1[test] = apply_rule(test_df, bounds_f1)
        v1_bounds = intervals(train_df, y_train, ORIGINAL, (5, 95))
        pred_v1[test] = apply_rule(test_df, v1_bounds)
        selection_rows.append({"repeat":repeat, "outer_fold":fold, "objective":"robust_mcc", **chosen.to_dict(),
                               "bounds_json":json.dumps(bounds, sort_keys=True)})
        selection_rows.append({"repeat":repeat, "outer_fold":fold, "objective":"legacy_f1", **chosen_f1.to_dict(),
                               "bounds_json":json.dumps(bounds_f1, sort_keys=True)})
        fold_rows.append({"repeat":repeat, "outer_fold":fold, "rule":"v2_nested", **metrics(y_test, pred_v2[test])})
        fold_rows.append({"repeat":repeat, "outer_fold":fold, "rule":"v2_nested_legacy_f1", **metrics(y_test, pred_v2_f1[test])})
        fold_rows.append({"repeat":repeat, "outer_fold":fold, "rule":"v1_rederived", **metrics(y_test, pred_v1[test])})
    if (pred_v2 < 0).any() or (pred_v2_f1 < 0).any() or (pred_v1 < 0).any():
        raise RuntimeError("Incomplete OOF predictions")
    repeat_rows = [
        {"repeat":repeat, "rule":"v2_nested", **metrics(y, pred_v2)},
        {"repeat":repeat, "rule":"v2_nested_legacy_f1", **metrics(y, pred_v2_f1)},
        {"repeat":repeat, "rule":"v1_rederived", **metrics(y, pred_v1)},
    ]
    predictions = pd.DataFrame({"repeat":repeat, "row":np.arange(len(data)), "reference_id":groups,
                                "TC":data["TC"], "target":y, "pred_v2":pred_v2,
                                "pred_v2_legacy_f1":pred_v2_f1, "pred_v1":pred_v1})
    return repeat_rows, fold_rows, selection_rows, shap_rows, predictions


def final_rule(data, objective="robust_mcc"):
    y = data["target"].to_numpy(int); groups = data["reference_id"].astype(str).to_numpy()
    ranking = shap_ranking(data, y, BASE_SEED + 999999)
    candidates = candidate_sets(ranking)
    all_runs = []
    for repeat in range(N_REPEATS):
        _, results = select_candidate(data, y, groups, candidates, BASE_SEED + 700000 + repeat, n_splits=N_OUTER)
        results["repeat"] = repeat
        all_runs.append(results)
    pooled = pd.concat(all_runs, ignore_index=True)
    summary = pooled.groupby(["candidate","features","n_features","q_low","q_high"], as_index=False).agg(
        MCC=("MCC","mean"), balanced_accuracy=("balanced_accuracy","mean"),
        enrichment=("enrichment","mean"), recall=("recall","mean"), precision=("precision","mean"), F1=("F1","mean"),
        MCC_sd=("MCC","std"), BA_sd=("balanced_accuracy","std"),
    )
    chosen = choose_candidate(summary.rename(columns={"recall":"recall", "precision":"precision",
                                                       "balanced_accuracy":"balanced_accuracy", "enrichment":"enrichment",
                                                       "MCC":"MCC"}), objective)
    feats = chosen["features"].split(";"); band=(float(chosen["q_low"]),float(chosen["q_high"]))
    bounds = intervals(data, y, feats, band)
    return ranking, pooled, summary, chosen, bounds


def main():
    global DATA, OUT, N_REPEATS, N_OUTER, N_INNER, BASE_SEED, N_JOBS
    parser = argparse.ArgumentParser(
        description="Publication-grouped nested SHAP-guided rule selection for the 250–350 K target."
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--outer-folds", type=int, default=N_OUTER)
    parser.add_argument("--inner-folds", type=int, default=N_INNER)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    args = parser.parse_args()

    DATA = args.data
    OUT = args.output_dir
    N_REPEATS = args.repeats
    N_OUTER = args.outer_folds
    N_INNER = args.inner_folds
    BASE_SEED = args.seed
    N_JOBS = args.n_jobs

    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    data["target"] = data["TC"].between(250, 350, inclusive="both").astype(int)
    if data[FEATURES + ["TC"]].isna().any().any(): raise ValueError("Missing values in analysis matrix")
    repeat_rows=[]; fold_rows=[]; selection_rows=[]; shap_rows=[]; prediction_rows=[]
    for repeat in range(N_REPEATS):
        rr, fr, sr, shr, pr = evaluate_repeat(data, repeat)
        repeat_rows.extend(rr); fold_rows.extend(fr); selection_rows.extend(sr); shap_rows.extend(shr); prediction_rows.append(pr)
        print(f"completed repeat {repeat+1}/{N_REPEATS}")
    repeat_df=pd.DataFrame(repeat_rows); fold_df=pd.DataFrame(fold_rows); selections=pd.DataFrame(selection_rows)
    shap_df=pd.concat(shap_rows,ignore_index=True); predictions=pd.concat(prediction_rows,ignore_index=True)
    summary=repeat_df.groupby("rule",as_index=False).agg({
        "balanced_accuracy":["mean","std"],"MCC":["mean","std"],"F1":["mean","std"],
        "precision":["mean","std"],"recall":["mean","std"],"specificity":["mean","std"],"enrichment":["mean","std"]})
    summary.columns=["rule"]+[f"{a}_{b}" for a,b in summary.columns.tolist()[1:]]
    shap_summary=shap_df.groupby("feature",as_index=False).agg(mean_rank=("rank","mean"),sd_rank=("rank","std"),
                                                                mean_abs_shap=("mean_abs_shap","mean"),top3_rate=("rank",lambda x:(x<=3).mean()))
    shap_summary=shap_summary.sort_values("mean_rank")
    final_rank, final_runs, final_summary, chosen, bounds = final_rule(data, "robust_mcc")
    _, _, _, chosen_f1, bounds_f1 = final_rule(data, "legacy_f1")
    final_payload={"selection":chosen.to_dict(),"features":chosen["features"].split(";"),"bounds":bounds,
                   "selection_note":"Selected only from the 141-record development data by repeated publication-grouped CV; external sets were not used numerically."}
    final_payload_f1={"selection":chosen_f1.to_dict(),"features":chosen_f1["features"].split(";"),"bounds":bounds_f1,
                      "selection_note":"Sensitivity analysis preserving the original F1-first ranking objective."}
    repeat_df.to_csv(OUT/"repeat_level_metrics.csv",index=False)
    fold_df.to_csv(OUT/"outer_fold_metrics.csv",index=False)
    selections.to_csv(OUT/"outer_fold_rule_selections.csv",index=False)
    shap_df.to_csv(OUT/"outer_training_shap_rankings.csv",index=False)
    shap_summary.to_csv(OUT/"shap_rank_stability.csv",index=False)
    predictions.to_csv(OUT/"repeated_oof_predictions.csv",index=False)
    summary.to_csv(OUT/"nested_rule_comparison_summary.csv",index=False)
    final_rank.to_csv(OUT/"full_development_shap_ranking.csv",index=False)
    final_runs.to_csv(OUT/"final_rule_cv_all_runs.csv",index=False)
    final_summary.to_csv(OUT/"final_rule_candidate_summary.csv",index=False)
    (OUT/"final_v2_rule.json").write_text(json.dumps(final_payload,indent=2),encoding="utf-8")
    (OUT/"final_v2_rule_legacy_f1_objective.json").write_text(json.dumps(final_payload_f1,indent=2),encoding="utf-8")
    print("\nNested comparison\n",summary.to_string(index=False))
    print("\nSHAP stability\n",shap_summary.to_string(index=False))
    print("\nFinal v2\n",json.dumps(final_payload,indent=2))
    print("\nFinal v2 legacy F1 objective\n",json.dumps(final_payload_f1,indent=2))


if __name__ == "__main__": main()
