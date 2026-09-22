# Publication-grouped validation of composition-only Curie-temperature screening

This repository contains the public reproducibility package for the manuscript:

> **Publication-grouped evaluation exposes the limits of composition-only Curie-temperature screening in magnetocaloric high-entropy alloys**

The repository is intentionally compact. It contains the analysis-ready data, publication-group assignments, source manifests, executable analysis code, principal numerical outputs, and final figures required to audit or reproduce the reported results. Manuscript drafts, downloaded source articles, duplicate workbooks, and exploratory files that do not support a reported result are deliberately excluded.

## Quick verification

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --mode verify
```

This model-free verification normally completes in seconds. It checks dataset sizes, publication groups and their size distribution, descriptor–target deduplication, held-out publication coverage, headline regression metrics, development-window metrics, nested rule-selection results, historical and targeted literature outcomes, the processing-state counterexample, source manifests, reference numbering, figure availability, and the absence of redistributed source-article PDFs.

The generated report is written to `docs/VERIFICATION_REPORT.json`.

## Full computational rerun

```bash
python scripts/run_pipeline.py --mode full --n-jobs 4
```

This reruns publication-grouped nested model selection, full-data TreeSHAP interpretation, development-window rules, the historical audit, repeated nested target-window rule selection, external-rule metrics, and Figures 2–7. New files are written under `reproduced/`; supplied reference outputs under `results/` are not overwritten. Runtime is hardware-dependent and may be tens of minutes or longer.

To regenerate only the deterministic window and rule figures:

```bash
python scripts/run_pipeline.py --mode figures
```

## Repository contents

| Path | Contents |
|---|---|
| `data/raw/` | Original 199-record numerical table and the record-random comparison used in the validation-design figure |
| `data/processed/` | 199-row provenance table, 141-row analysis table, 32-record historical audit, 26-row five-paper challenge table, and processing-state sensitivity pairs |
| `references/` | Source manifests for all analysis cohorts and the 51-entry v28 bibliography |
| `scripts/` | Verification, grouped regression, TreeSHAP, rule-selection, external evaluation, and figure-generation code |
| `results/` | Supplied numerical outputs needed to audit the manuscript claims |
| `figures/` | Final manuscript Figures 1–7 in PNG format |
| `docs/` | Data dictionary, claim-to-file map, reviewer guide, reproducibility notes, verification report, and checksums |

## Data cohorts

- **Development data:** 199 literature measurement records mapped to 42 publication sources and consolidated to 141 unique descriptor–Tc rows using eleven composition descriptors plus Tc as the deduplication key.
- **Historical audit:** 32 records absent from the 141-row development table: 14 experimental HEA/MEA/CCA controls, 15 ordered Mn–Ni–Si-based out-of-domain stress records, and 3 simulation sensitivity records.
- **Targeted five-paper challenge:** 26 rows retaining primary states, alternate processing states, transition-ambiguity cases, and an exclusion log. The primary classification analysis uses 13 rows. Because target relevance influenced retrieval, this cohort is a stress test rather than a population-level external validation set.

## Headline values

- Publication-grouped HEA-descriptor LightGBM: OOF R² = 0.607, RMSE = 93.6 K, MAE = 57.7 K.
- Repeated nested F1-first rule selection: balanced accuracy = 0.694 ± 0.039 and MCC = 0.374 ± 0.075.
- Historical 32-record fixed-family audit: balanced accuracy = 0.860; within the Mn-based family, balanced accuracy = 0.563 and enrichment = 1.07×.
- Full-development candidate in the 13-record primary challenge: TP/TN/FP/FN = 8/0/4/1, balanced accuracy = 0.444, MCC = −0.192.
- Same nominal Ag-containing composition: reported Tc changes from 209 K in the as-rolled state to 295 K after annealing.

## Source articles and copyright

Copyrighted article PDFs are not redistributed. Each literature-derived record is traceable through the DOI-bearing source manifests in `references/`. See `LICENSE.md` for the repository's split code/data licensing and third-party-material exception.

## Citation

Please cite the associated manuscript and the archived repository release. After the GitHub release is connected to Zenodo, add the Zenodo DOI to `CITATION.cff` and to the manuscript's Data and Code Availability statements.

Start with `docs/REVIEWER_GUIDE.md` for a short audit route and `docs/ANALYSIS_MAP.md` for exact table/figure provenance.
