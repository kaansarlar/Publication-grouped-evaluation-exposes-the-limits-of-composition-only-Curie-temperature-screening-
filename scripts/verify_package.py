#!/usr/bin/env python3
"""Fast, model-free integrity checks for the reviewer package and headline results."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from development_window_rules import derive
from evaluate_frozen_rules import metric_row


ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[dict] = []


def check(name: str, condition: bool, detail: str) -> None:
    CHECKS.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    if not condition:
        raise RuntimeError(f"{name}: {detail}")


def close(value: float, expected: float, atol: float = 1e-6) -> bool:
    return bool(np.isclose(float(value), expected, rtol=0, atol=atol))


def main() -> None:
    raw = pd.read_csv(ROOT / "data/raw/development_measurements_199.csv")
    full = pd.read_csv(ROOT / "data/processed/development_full_199.csv")
    unique = pd.read_csv(ROOT / "data/processed/development_unique_141.csv")
    check("raw-row-count", len(raw) == 199, f"found {len(raw)}; expected 199")
    check("curated-row-count", len(full) == 199, f"found {len(full)}; expected 199")
    check("deduplicated-row-count", len(unique) == 141, f"found {len(unique)}; expected 141")
    check("publication-groups", unique["reference_id"].nunique() == 42,
          f"found {unique['reference_id'].nunique()}; expected 42")
    group_sizes = unique.groupby("reference_id").size()
    check("publication-group-size-distribution",
          int(group_sizes.min()) == 1 and close(group_sizes.median(), 3.0)
          and int(group_sizes.max()) == 18 and close(group_sizes.mean(), 141 / 42),
          f"min/median/mean/max={group_sizes.min()}/{group_sizes.median():.1f}/{group_sizes.mean():.2f}/{group_sizes.max()}")
    descriptor_key = ["dX", "VEC", "sigma", "dHmix", "dSmix", "nunfilled_mean", "mag_moment_mean",
                      "avg_d_valence_electrons", "bandgap_mean", "melting_point_mean",
                      "electronegativity_range", "TC"]
    check("deduplication-key", len(full.drop_duplicates(descriptor_key)) == 141,
          "descriptor–target key must produce 141 rows")

    splits = pd.read_csv(ROOT / "results/grouped_cv/splits__hea__LightGBM.csv")
    test_refs = [ref for text in splits["test_references"].astype(str) for ref in text.split(";")]
    check("outer-test-reference-coverage", len(test_refs) == len(set(test_refs)) == 42,
          f"test-fold references={len(test_refs)}, unique={len(set(test_refs))}")

    summary = pd.read_csv(ROOT / "results/grouped_cv/publication_grouped_model_summary.csv")
    best = summary.loc[(summary["feature_set"] == "hea") & (summary["model"] == "LightGBM")].iloc[0]
    check("headline-regression", close(best.oof_r2, 0.606784, 1e-6) and close(best.oof_rmse, 93.643123, 1e-6)
          and close(best.oof_mae, 57.664585, 1e-6),
          f"R2={best.oof_r2:.6f}, RMSE={best.oof_rmse:.6f}, MAE={best.oof_mae:.6f}")
    check("nonnegative-best-model-predictions", int(best.n_negative_predictions) == 0 and best.min_prediction_K > 0,
          f"negative={int(best.n_negative_predictions)}, minimum={best.min_prediction_K:.3f} K")

    window_summary, _, _ = derive(unique)
    target_expected = {
        "250–350 K": (0.650000, 0.764706, 0.702703, 0.601532, 2.695588),
        "280–320 K": (0.578947, 0.578947, 0.578947, 0.513416, 4.296398),
        "300–400 K": (0.548387, 0.739130, 0.629630, 0.553566, 3.361853),
    }
    for label, expected in target_expected.items():
        row = window_summary.set_index("target_window").loc[label]
        values = (row.precision, row.recall, row.F1, row.MCC, row.enrichment)
        check(f"development-rule-{label}", all(close(v, e, 1e-4) for v, e in zip(values, expected)),
              "precision/recall/F1/MCC/enrichment=" + "/".join(f"{v:.6f}" for v in values))

    nested = pd.read_csv(ROOT / "results/rule_reassessment/nested_rule_comparison_summary.csv").set_index("rule")
    f1_first = nested.loc["v2_nested_legacy_f1"]
    check("nested-f1-first", close(f1_first.balanced_accuracy_mean, 0.694173, 1e-6)
          and close(f1_first.balanced_accuracy_std, 0.038788, 1e-6)
          and close(f1_first.MCC_mean, 0.374176, 1e-6),
          f"BA={f1_first.balanced_accuracy_mean:.6f}±{f1_first.balanced_accuracy_std:.6f}; MCC={f1_first.MCC_mean:.6f}")

    historical = pd.read_csv(ROOT / "data/processed/historical_audit_32.csv")
    challenge = pd.read_csv(ROOT / "data/processed/targeted_challenge_26.csv")
    h_v1 = metric_row("historical-v1", historical, "pass::250–350 K", "positive_v2")
    h_mn = metric_row("historical-mn", historical.loc[historical["chemistry_stratum"].str.startswith("Mn")],
                      "pass::250–350 K", "positive_v2")
    c_v2 = metric_row("challenge-v2", challenge.loc[challenge["cohort"].eq("primary")],
                      "pass::v2_250_350", "target_250_350")
    check("historical-pooled-v1", (h_v1["TP"], h_v1["TN"], h_v1["FP"], h_v1["FN"]) == (7, 18, 7, 0)
          and close(h_v1["balanced_accuracy"], 0.86), str(h_v1))
    check("historical-mn-stratum-v1", (h_mn["TP"], h_mn["TN"], h_mn["FP"], h_mn["FN"]) == (7, 1, 7, 0)
          and close(h_mn["enrichment"], 1.0714285714), str(h_mn))
    check("targeted-challenge-v2", (c_v2["TP"], c_v2["TN"], c_v2["FP"], c_v2["FN"]) == (8, 0, 4, 1)
          and close(c_v2["balanced_accuracy"], 0.4444444444) and close(c_v2["MCC"], -0.1924500897), str(c_v2))
    primary = challenge.loc[challenge["cohort"].eq("primary")]
    check("challenge-composition", len(challenge) == 26 and challenge["doi"].nunique() == 5 and len(primary) == 13
          and int(primary["target_250_350"].sum()) == 9 and primary["target_250_350"].eq(0).sum() == 4,
          f"all={len(challenge)}, papers={challenge['doi'].nunique()}, primary={len(primary)}, positives={int(primary['target_250_350'].sum())}")
    process = pd.read_csv(ROOT / "data/processed/process_state_sensitivity.csv")
    ag = process.loc[process["pair"].eq("ROC-AG-020")].iloc[0]
    check("Ag-process-counterexample", close(ag.TC_K_as_rolled, 209) and close(ag.TC_K_annealed, 295)
          and bool(ag.label_changed), f"{ag.TC_K_as_rolled:.0f}→{ag.TC_K_annealed:.0f} K")

    expected_labels = {
        "Zheng et al. [24]", "Wei et al. [25]", "Wei et al. [26]", "Qu et al. [27]", "Gao et al. [28]",
        "Guisado-Arenas et al. [29]", "Zhai et al. [30]", "Pei et al. [31]", "Huang et al. [32]",
        "Xue et al. [33]", "Johnson and Frederick [51]",
    }
    check("historical-source-numbering", set(historical["Source"].unique()) == expected_labels,
          f"labels={sorted(historical['Source'].unique())}")

    reference_text = (ROOT / "references/manuscript_references.md").read_text(encoding="utf-8")
    ref_numbers = [int(value) for value in re.findall(r"(?m)^(\d+)\.\s", reference_text)]
    check("reference-sequence", ref_numbers == list(range(1, 52)),
          f"found {len(ref_numbers)} entries")

    development_sources = pd.read_csv(ROOT / "references/development_sources_42.csv")
    historical_sources = pd.read_csv(ROOT / "references/historical_audit_sources.csv")
    challenge_sources = pd.read_csv(ROOT / "references/targeted_challenge_sources.csv")
    check("source-manifest-coverage",
          len(development_sources) == 42 and int(development_sources["n_unique_records"].sum()) == 141
          and int(historical_sources["n_records"].sum()) == 32
          and len(challenge_sources) == 5 and int(challenge_sources["n_records"].sum()) == 26,
          "development/historical/challenge=42 sources/32 rows/5 sources and 26 rows")

    missing_figures = [str(i) for i in range(1, 8) if not (ROOT / f"figures/Fig{i}.png").exists()]
    check("figure-set", not missing_figures, f"missing={missing_figures or 'none'}")
    redistributed_articles = [
        path for path in ROOT.rglob("*.pdf")
        if "figures" not in path.parts and "reproduced" not in path.parts
    ]
    check("no-source-article-pdfs", not redistributed_articles,
          f"PDF files found={len(redistributed_articles)}")

    report = {
        "package": ROOT.name,
        "status": "PASS",
        "checks_passed": len(CHECKS),
        "checks": CHECKS,
    }
    report_path = ROOT / "docs/VERIFICATION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in CHECKS:
        print(f"[{item['status']}] {item['check']}: {item['detail']}")
    print(f"\nAll {len(CHECKS)} checks passed.")


if __name__ == "__main__":
    main()
