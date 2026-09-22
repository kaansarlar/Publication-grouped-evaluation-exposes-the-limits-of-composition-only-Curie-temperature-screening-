from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from lxml import etree
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
RANDOM_STATE = 42
OUTER_SPLITS = 10
INNER_SPLITS = 5
N_STRAT_BINS = 5

HEA_FEATURES = ["dX", "VEC", "sigma", "dHmix", "dSmix"]
MATMINER_FEATURES = [
    "nunfilled_mean",
    "mag_moment_mean",
    "avg_d_valence_electrons",
    "bandgap_mean",
    "melting_point_mean",
    "electronegativity_range",
]
FEATURE_SETS = {
    "hea": HEA_FEATURES,
    "matminer": MATMINER_FEATURES,
    "combined": HEA_FEATURES + MATMINER_FEATURES,
}
TARGET = "TC"
ROOT = Path(__file__).resolve().parents[1]


def normalize_pair(h, ds):
    return (round(float(str(h).replace(",", ".")), 6), round(float(str(ds).replace(",", ".")), 6))


def docx_root(path: Path):
    with ZipFile(path) as zf:
        return etree.fromstring(zf.read("word/document.xml"))


def extract_composition_rows(docx_path: Path):
    root = docx_root(docx_path)
    tables = root.xpath("//w:body/w:tbl", namespaces=NS)
    if not tables:
        raise ValueError("Composition document does not contain a table")
    rows = []
    for tr in tables[0].xpath("./w:tr", namespaces=NS)[2:]:
        cells = tr.xpath("./w:tc | ./w:sdt/w:sdtContent/w:tc", namespaces=NS)
        values = ["".join(tc.xpath(".//w:t/text()", namespaces=NS)).strip() for tc in cells]
        if len(values) != 4:
            raise ValueError(f"Unexpected composition-table row with {len(values)} cells: {values}")
        rows.append(values)
    return rows


def extract_bibliography(docx_path: Path):
    root = docx_root(docx_path)
    references = {}
    for p in root.xpath("//w:body/w:sdt/w:sdtContent/w:p", namespaces=NS):
        text = "".join(p.xpath(".//w:t/text()", namespaces=NS)).strip()
        match = re.match(r"^\[(\d+)\]\s*(.*)$", text)
        if match:
            references[f"[{int(match.group(1))}]"] = match.group(2).strip()
    return references


def build_reference_mapping(main_csv: Path, composition_docx: Path, output_dir: Path):
    main = pd.read_csv(main_csv)
    main.columns = [str(c).lstrip("\ufeff").replace("δ", "sigma") for c in main.columns]
    doc_rows = extract_composition_rows(composition_docx)
    bibliography = extract_bibliography(composition_docx)

    if len(main) != 199:
        raise ValueError(f"Expected 199 main records, found {len(main)}")
    if len(doc_rows) != 200:
        raise ValueError(f"Expected 200 composition-document rows, found {len(doc_rows)}")

    extra_name = "LaFe10(Fe0.2Co0.2Ni0.2Cr0.2Mn0.2)Si2"
    extra_candidates = [i for i, row in enumerate(doc_rows) if row[0] == extra_name]
    if len(extra_candidates) != 1:
        raise ValueError(f"Expected one known extra row, found {extra_candidates}")
    removed_index = extra_candidates[0]
    removed_row = doc_rows.pop(removed_index)

    mapped_reference_ids = []
    mapped_compositions = []
    ambiguous_flags = []
    block_audit = []
    main_start = 0
    i = 0
    while i < len(doc_rows):
        ref_id = doc_rows[i][3]
        j = i
        while j < len(doc_rows) and doc_rows[j][3] == ref_id:
            j += 1
        block = doc_rows[i:j]
        block_n = len(block)
        main_block = main.iloc[main_start : main_start + block_n]

        doc_pairs = Counter(normalize_pair(row[1], row[2]) for row in block)
        main_pairs = Counter(normalize_pair(h, ds) for h, ds in zip(main_block["H"], main_block["dS"]))
        if doc_pairs != main_pairs:
            raise ValueError(f"Reference block {ref_id} does not match main rows {main_start}:{main_start + block_n}")

        composition_queues = defaultdict(deque)
        for row in block:
            composition_queues[normalize_pair(row[1], row[2])].append(row[0])
        for h, ds in zip(main_block["H"], main_block["dS"]):
            pair = normalize_pair(h, ds)
            ambiguous = len(composition_queues[pair]) > 1
            mapped_compositions.append(composition_queues[pair].popleft())
            ambiguous_flags.append(ambiguous)
            mapped_reference_ids.append(ref_id)

        block_audit.append(
            {
                "reference_id": ref_id,
                "n_records": block_n,
                "main_start_zero_based": main_start,
                "main_end_exclusive": main_start + block_n,
                "hdS_multiset_match": True,
                "reference_text": bibliography.get(ref_id, ""),
            }
        )
        main_start += block_n
        i = j

    if main_start != len(main):
        raise ValueError(f"Mapped {main_start} records, expected {len(main)}")

    full = main.copy()
    full.insert(0, "source_row", np.arange(2, len(full) + 2))
    full.insert(1, "composition", mapped_compositions)
    full.insert(2, "reference_id", mapped_reference_ids)
    full.insert(3, "reference_text", [bibliography.get(x, "") for x in mapped_reference_ids])
    full.insert(4, "composition_match_ambiguous_within_reference", ambiguous_flags)

    canonical = FEATURE_SETS["combined"] + [TARGET]
    unique = full.drop_duplicates(subset=canonical, keep="first").reset_index(drop=True)
    if len(unique) != 141:
        raise ValueError(f"Expected 141 unique records, found {len(unique)}")
    if unique["reference_id"].nunique() != 42:
        raise ValueError(f"Expected 42 reference groups in unique data, found {unique['reference_id'].nunique()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(output_dir / "TC_full_199_with_composition_and_reference.csv", index=False)
    unique.to_csv(output_dir / "TC_unique_141_with_composition_and_reference.csv", index=False)
    pd.DataFrame(block_audit).to_csv(output_dir / "reference_block_audit.csv", index=False)
    mapping_audit = {
        "main_records": len(full),
        "composition_document_rows_before_exclusion": 200,
        "excluded_document_row_index_zero_based": removed_index,
        "excluded_document_row": removed_row,
        "reference_groups": int(full["reference_id"].nunique()),
        "unique_records": len(unique),
        "unique_reference_groups": int(unique["reference_id"].nunique()),
        "ambiguous_composition_assignments_within_reference": int(sum(ambiguous_flags)),
        "all_reference_block_hdS_multisets_match": True,
    }
    (output_dir / "reference_mapping_audit.json").write_text(json.dumps(mapping_audit, indent=2), encoding="utf-8")
    return full, unique, mapping_audit


def load_prepared_data(unique_csv: Path, full_csv: Path | None, output_dir: Path):
    """Load the public analysis-ready tables without requiring a Word source map."""
    unique = pd.read_csv(unique_csv)
    required = {"source_row", "composition", "reference_id", TARGET, *FEATURE_SETS["combined"]}
    missing = sorted(required.difference(unique.columns))
    if missing:
        raise ValueError(f"Prepared dataset is missing required columns: {missing}")
    if len(unique) != 141:
        raise ValueError(f"Expected 141 unique records, found {len(unique)}")
    if unique["reference_id"].nunique() != 42:
        raise ValueError(
            f"Expected 42 publication groups, found {unique['reference_id'].nunique()}"
        )

    full = pd.read_csv(full_csv) if full_csv is not None else unique.copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(output_dir / "TC_full_199_with_composition_and_reference.csv", index=False)
    unique.to_csv(output_dir / "TC_unique_141_with_composition_and_reference.csv", index=False)
    audit = {
        "input_mode": "analysis-ready CSV",
        "main_records": int(len(full)),
        "reference_groups": int(full["reference_id"].nunique()),
        "unique_records": int(len(unique)),
        "unique_reference_groups": int(unique["reference_id"].nunique()),
    }
    (output_dir / "reference_mapping_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return full, unique, audit


def regression_bins(y, q=N_STRAT_BINS):
    y = pd.Series(np.asarray(y))
    for candidate in [q, 4, 3, 2]:
        try:
            bins = pd.qcut(y, q=candidate, labels=False, duplicates="drop")
            if bins.nunique() >= 2:
                return bins.astype(int).to_numpy()
        except Exception:
            pass
    return pd.cut(y, bins=2, labels=False, include_lowest=True).astype(int).to_numpy()


def grouped_splits(y, groups, n_splits, random_state):
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(y)), regression_bins(y), groups))


def sample_grid(grid, n_candidates, seed):
    if len(list(ParameterGrid(grid))) < n_candidates:
        raise ValueError("Parameter grid smaller than requested equal candidate budget")
    return [{k: [v] for k, v in row.items()} for row in ParameterSampler(grid, n_iter=n_candidates, random_state=seed)]


def model_specs():
    specs = {
        "DummyMean": {
            "pipeline": Pipeline([("regressor", DummyRegressor(strategy="mean"))]),
            "grid": None,
        },
        "Ridge": {
            "pipeline": Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(random_state=RANDOM_STATE))]),
            "grid": {
                "regressor__alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 1000],
                "regressor__fit_intercept": [True, False],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([("regressor", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))]),
            "grid": {
                "regressor__n_estimators": [100, 200, 400, 600, 800],
                "regressor__max_depth": [None, 5, 10, 15, 20, 30],
                "regressor__min_samples_split": [2, 3, 5, 8],
                "regressor__min_samples_leaf": [1, 2, 3, 5],
                "regressor__max_features": ["sqrt", 0.6, 0.8, 1.0],
            },
        },
        "GradientBoosting": {
            "pipeline": Pipeline([("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
            "grid": {
                "regressor__n_estimators": [100, 200, 400, 600, 800],
                "regressor__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "regressor__max_depth": [2, 3, 4, 5],
                "regressor__subsample": [0.6, 0.8, 0.9, 1.0],
                "regressor__max_features": [None, "sqrt", 0.8],
                "regressor__min_samples_leaf": [1, 2, 3],
            },
        },
        "SVR": {
            "pipeline": Pipeline([("scaler", StandardScaler()), ("regressor", SVR(kernel="rbf"))]),
            "grid": {
                "regressor__C": [1, 3, 10, 30, 100, 300, 500, 1000],
                "regressor__gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                "regressor__epsilon": [0.01, 0.03, 0.05, 0.1, 0.2],
            },
        },
    }
    if XGBRegressor is not None:
        specs["XGBoost"] = {
            "pipeline": Pipeline([("regressor", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1, verbosity=0))]),
            "grid": {
                "regressor__n_estimators": [100, 200, 400, 600, 800],
                "regressor__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "regressor__max_depth": [2, 3, 4, 5, 6],
                "regressor__subsample": [0.6, 0.8, 1.0],
                "regressor__colsample_bytree": [0.6, 0.8, 1.0],
                "regressor__min_child_weight": [1, 3, 5],
                "regressor__reg_lambda": [0.5, 1.0, 2.0],
            },
        }
    if LGBMRegressor is not None:
        specs["LightGBM"] = {
            "pipeline": Pipeline([("regressor", LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1))]),
            "grid": {
                "regressor__n_estimators": [100, 200, 400, 600, 800],
                "regressor__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "regressor__max_depth": [-1, 3, 5, 8, 12],
                "regressor__num_leaves": [7, 15, 20, 31, 40],
                "regressor__subsample": [0.6, 0.8, 1.0],
                "regressor__colsample_bytree": [0.6, 0.8, 1.0],
                "regressor__min_child_samples": [3, 5, 10, 20],
            },
        }
    return specs


def metrics(y, pred):
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
    }


def run_model(df, features, feature_set, model_name, spec, out_dir, candidate_budget, n_jobs):
    X = df[features].astype(float).reset_index(drop=True)
    y = df[TARGET].astype(float).reset_index(drop=True)
    groups = df["reference_id"].astype(str).reset_index(drop=True).to_numpy()
    outer = grouped_splits(y, groups, OUTER_SPLITS, RANDOM_STATE)
    y_oof = np.full(len(y), np.nan)
    fold_rows = []
    param_rows = []
    split_rows = []
    candidate_list = None if spec["grid"] is None else sample_grid(spec["grid"], candidate_budget, RANDOM_STATE + list(model_specs()).index(model_name) * 17)

    for fold_idx, (train_idx, test_idx) in enumerate(outer, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train, g_test = groups[train_idx], groups[test_idx]
        inner = grouped_splits(y_train, g_train, INNER_SPLITS, RANDOM_STATE + fold_idx)
        estimator = clone(spec["pipeline"])
        if candidate_list is not None:
            search = GridSearchCV(
                estimator,
                candidate_list,
                scoring="r2",
                cv=inner,
                n_jobs=n_jobs,
                refit=True,
                error_score=np.nan,
                verbose=0,
            )
            search.fit(X_train, y_train, groups=g_train)
            fitted = search.best_estimator_
            best_params = search.best_params_
            best_inner = float(search.best_score_)
        else:
            fitted = estimator.fit(X_train, y_train)
            best_params = {}
            best_inner = np.nan
        pred = fitted.predict(X_test)
        y_oof[test_idx] = pred
        fm = metrics(y_test, pred)
        fold_rows.append({
            "feature_set": feature_set,
            "model": model_name,
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_references": len(set(g_train)),
            "n_test_references": len(set(g_test)),
            "inner_best_r2": best_inner,
            **fm,
        })
        param_rows.append({"feature_set": feature_set, "model": model_name, "fold": fold_idx, **best_params})
        split_rows.append({
            "feature_set": feature_set,
            "model": model_name,
            "fold": fold_idx,
            "train_references": ";".join(sorted(set(g_train))),
            "test_references": ";".join(sorted(set(g_test))),
            "test_tc_min": float(y_test.min()),
            "test_tc_max": float(y_test.max()),
            "test_tc_mean": float(y_test.mean()),
        })
        print(f"{feature_set:9s} {model_name:17s} fold {fold_idx:02d}/{len(outer)} R2={fm['r2']:.3f} MAE={fm['mae']:.1f}", flush=True)

    if np.isnan(y_oof).any():
        raise RuntimeError("OOF prediction vector contains missing values")
    oof_metrics = metrics(y, y_oof)
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "feature_set": feature_set,
        "model": model_name,
        "candidate_budget": 0 if candidate_list is None else candidate_budget,
        "n_records": len(df),
        "n_references": int(df["reference_id"].nunique()),
        "n_features": len(features),
        "oof_r2": oof_metrics["r2"],
        "oof_rmse": oof_metrics["rmse"],
        "oof_mae": oof_metrics["mae"],
        "fold_r2_mean": float(fold_df["r2"].mean()),
        "fold_r2_sd": float(fold_df["r2"].std(ddof=1)),
        "fold_mae_mean": float(fold_df["mae"].mean()),
        "fold_mae_sd": float(fold_df["mae"].std(ddof=1)),
        "n_negative_predictions": int((y_oof < 0).sum()),
        "min_prediction_K": float(y_oof.min()),
        "max_prediction_K": float(y_oof.max()),
    }
    oof_df = df[["source_row", "composition", "reference_id", TARGET]].copy()
    oof_df["feature_set"] = feature_set
    oof_df["model"] = model_name
    oof_df["y_true_TC"] = y
    oof_df["y_pred_TC"] = y_oof
    oof_df["residual_TC"] = y - y_oof
    oof_df["absolute_error_K"] = np.abs(y - y_oof)

    stem = f"{feature_set}__{model_name}"
    pd.DataFrame([summary]).to_csv(out_dir / f"summary__{stem}.csv", index=False)
    fold_df.to_csv(out_dir / f"folds__{stem}.csv", index=False)
    pd.DataFrame(param_rows).to_csv(out_dir / f"params__{stem}.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out_dir / f"splits__{stem}.csv", index=False)
    oof_df.to_csv(out_dir / f"oof__{stem}.csv", index=False)
    return summary, fold_df, pd.DataFrame(param_rows), pd.DataFrame(split_rows), oof_df


def plot_results(summary_df, oof_df, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    order = summary_df.sort_values(["feature_set", "oof_r2"], ascending=[True, False])
    labels = [f"{r.feature_set}\n{r.model}" for r in order.itertuples()]
    colors = [{"hea": "#4C78A8", "matminer": "#F58518", "combined": "#54A24B"}[x] for x in order["feature_set"]]
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)
    ax.bar(np.arange(len(order)), order["oof_r2"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Publication-grouped OOF $R^2$")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "publication_grouped_model_comparison.png", dpi=300)
    fig.savefig(fig_dir / "publication_grouped_model_comparison.pdf")
    plt.close(fig)

    best = summary_df.sort_values("oof_r2", ascending=False).iloc[0]
    b = oof_df[(oof_df["feature_set"] == best["feature_set"]) & (oof_df["model"] == best["model"])].copy()
    lo = float(min(b["y_true_TC"].min(), b["y_pred_TC"].min()))
    hi = float(max(b["y_true_TC"].max(), b["y_pred_TC"].max()))
    pad = 0.04 * (hi - lo)
    fig, ax = plt.subplots(figsize=(6.2, 5.5), dpi=180)
    ax.scatter(b["y_true_TC"], b["y_pred_TC"], s=28, alpha=0.75, edgecolor="white", linewidth=0.35)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.axhline(0, color="#B22222", lw=0.8, ls=":")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Experimental $T_C$ (K)")
    ax.set_ylabel("Publication-grouped OOF prediction (K)")
    ax.set_title(f"{best['model']} with {best['feature_set']} descriptors")
    ax.text(0.03, 0.97, f"$R^2$={best['oof_r2']:.3f}\nMAE={best['oof_mae']:.1f} K", transform=ax.transAxes, va="top")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "publication_grouped_best_model_parity.png", dpi=300)
    fig.savefig(fig_dir / "publication_grouped_best_model_parity.pdf")
    plt.close(fig)

    return {"best_feature_set": best["feature_set"], "best_model": best["model"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/development_unique_141.csv",
        help="Analysis-ready 141-row table used by the public reproducibility path.",
    )
    parser.add_argument(
        "--full-data",
        type=Path,
        default=ROOT / "data/processed/development_full_199.csv",
        help="Analysis-ready 199-row provenance table copied into the rerun output.",
    )
    parser.add_argument("--main-csv", type=Path)
    parser.add_argument("--composition-docx", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--feature-sets", nargs="+", default=["hea", "matminer", "combined"])
    parser.add_argument("--models", nargs="+", default=["DummyMean", "Ridge", "RandomForest", "GradientBoosting", "SVR", "XGBoost", "LightGBM"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.main_csv is not None or args.composition_docx is not None:
        if args.main_csv is None or args.composition_docx is None:
            parser.error("--main-csv and --composition-docx must be supplied together")
        full, unique, mapping_audit = build_reference_mapping(
            args.main_csv, args.composition_docx, args.output_dir
        )
    else:
        full, unique, mapping_audit = load_prepared_data(
            args.data, args.full_data, args.output_dir
        )
    specs = model_specs()
    summaries, folds, params, splits, oofs = [], [], [], [], []
    start = time.time()
    for feature_set in args.feature_sets:
        features = FEATURE_SETS[feature_set]
        for model_name in args.models:
            if model_name not in specs:
                raise RuntimeError(f"Requested model unavailable: {model_name}")
            result = run_model(unique, features, feature_set, model_name, specs[model_name], args.output_dir, args.candidate_budget, args.n_jobs)
            summaries.append(result[0]); folds.append(result[1]); params.append(result[2]); splits.append(result[3]); oofs.append(result[4])

    summary_df = pd.DataFrame(summaries).sort_values(["feature_set", "oof_r2"], ascending=[True, False])
    fold_df = pd.concat(folds, ignore_index=True)
    param_df = pd.concat(params, ignore_index=True)
    split_df = pd.concat(splits, ignore_index=True)
    oof_df = pd.concat(oofs, ignore_index=True)
    summary_df.to_csv(args.output_dir / "publication_grouped_model_summary.csv", index=False)
    fold_df.to_csv(args.output_dir / "publication_grouped_fold_metrics.csv", index=False)
    param_df.to_csv(args.output_dir / "publication_grouped_best_params.csv", index=False)
    split_df.to_csv(args.output_dir / "publication_grouped_split_audit.csv", index=False)
    oof_df.to_csv(args.output_dir / "publication_grouped_oof_predictions.csv", index=False)
    best = plot_results(summary_df, oof_df, args.output_dir)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "candidate_budget": args.candidate_budget,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "n_jobs": args.n_jobs,
        "elapsed_seconds": time.time() - start,
        "lightgbm_available": LGBMRegressor is not None,
        "xgboost_available": XGBRegressor is not None,
        **best,
        **mapping_audit,
    }
    (args.output_dir / "analysis_environment_and_audit.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
