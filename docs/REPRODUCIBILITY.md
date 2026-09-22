# Reproducibility notes

## Validation design

All supervised regression estimates use publication reference as the grouping variable. The outer loop uses ten-fold `StratifiedGroupKFold`; the inner loop performs grouped hyperparameter selection. Every record from a publication is assigned wholly to either training or test data in a given fold. The supplied split-audit files make these assignments explicit.

The target-specific rule study repeats ten-fold publication-grouped outer validation ten times. TreeSHAP ranking, candidate descriptor-family construction, percentile-band selection, and rule fitting occur inside each outer training partition. The held-out publication groups are used only for evaluation.

## Randomness

- Principal regression base seed: 42.
- Repeated nested rule-selection base seed: 42017.
- Figure jitter seed: 42.
- All explicit seeds are visible in the scripts.

## Supplied versus regenerated outputs

The supplied reference results are under `results/`. Reruns are written to `reproduced/`, which is ignored by Git. This separation prevents accidental replacement of the values reviewed with the manuscript.

`python scripts/run_pipeline.py --mode verify` does not refit a model. It recalculates deterministic counts and rule metrics from row-level data and checks stored model outputs against the manuscript values.

`python scripts/run_pipeline.py --mode full --n-jobs 4` refits the models and writes every computational stage under `reproduced/`. The public rerun begins with the analysis-ready 141-row CSV and its publication-group identifiers; it does not depend on a manuscript file or a private reference-manager database. Small last-digit variation can occur across platforms or library builds for threaded tree algorithms; confusion counts and rounded manuscript values should remain stable under the pinned LightGBM and XGBoost versions.

## Computational environment

The reference grouped run used Python 3.12, 10 outer folds, 5 inner folds, a 60-candidate budget, and 8 worker jobs. The recorded run metadata are in `results/grouped_cv/analysis_environment_and_audit.json`. `requirements.txt` pins the two external boosting libraries and constrains the remaining scientific stack to compatible major versions.

## Literature-derived tables

The historical and targeted challenge records were manually transcribed and standardized from the cited publications. The package distributes the curated values and source/DOI manifests, not the copyrighted PDFs. Alternate process states, ambiguity cases, and excluded records remain in the 26-row challenge table so that the 13-row primary analysis can be reconstructed rather than inferred.

## Integrity

`docs/CHECKSUMS.sha256` contains SHA-256 hashes for every packaged file except the checksum file itself. Run from the repository root:

```bash
sha256sum --check docs/CHECKSUMS.sha256
```

After an intentional repository update, regenerate the manifest with:

```bash
python scripts/generate_checksums.py
```
