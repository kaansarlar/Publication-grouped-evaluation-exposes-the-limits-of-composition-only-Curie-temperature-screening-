# Data dictionary

## `data/raw/development_measurements_199.csv`

Original 199-record numerical table used as the starting point. Measured magnetocaloric outputs are retained for provenance but excluded from the Tc feature matrix.

| Field | Meaning |
|---|---|
| `TC` | Reported Curie/transition temperature (K); regression target |
| `H` | Applied field reported with the source record; excluded from predictors |
| `dS`, `dS_ref` | Magnetic entropy-change fields; excluded from predictors |
| `RC` | Refrigerant-capacity field; excluded from predictors |
| `dX`, `VEC`, `sigma`, `dHmix`, `dSmix` | Classical composition-derived HEA descriptors |
| `nunfilled_mean`, `mag_moment_mean`, `avg_d_valence_electrons`, `bandgap_mean`, `melting_point_mean`, `electronegativity_range` | Composition-derived elemental-statistics descriptors |

## `data/processed/development_full_199.csv` and `development_unique_141.csv`

The processed files add row-level provenance to the numerical table.

| Field | Meaning |
|---|---|
| `source_row` | Source-table row identifier |
| `composition` | Retained nominal composition formula |
| `reference_id` | Publication-group identifier used by grouped cross-validation |
| `reference_text` | Full source citation recovered from the composition/source document |
| `composition_match_ambiguous_within_reference` | True where repeated H–dS keys within a publication prevent a unique formula ordering; publication assignment remains exact |
| remaining fields | Same numerical variables as the raw table |

`development_unique_141.csv` removes repeats using the eleven composition descriptors plus `TC`. Formula strings are preserved for traceability but are not required to be unique.

## `data/processed/historical_audit_32.csv`

| Field group | Meaning |
|---|---|
| `Composition`, `TC_K`, `Transition basis`, `Source` | Literature-reported identity, transition, and v28 source label |
| `chemistry_stratum`, `validation_role`, `experiment_stratum`, `is_simulation` | Prespecified interpretation strata |
| `pass::*` | Whether a record satisfies the indicated frozen rule |
| `positive::*`, `positive_v2` | Whether the reported transition lies in the associated target interval |
| `mag_moment_mean`, `avg_d_valence_electrons`, `melting_point_mean` | Features used by the full-development candidate |

## `data/processed/targeted_challenge_26.csv`

| Field group | Meaning |
|---|---|
| `record_id`, `source`, `doi` | Stable row ID and publication provenance |
| `composition`, `composition_dict`, `state` | Nominal composition and processing/measurement state |
| `TC_K`, `TC_relation`, `target_250_350` | Reported numerical/censored transition and target label |
| `cohort` | `primary`, `process_sensitivity`, `ambiguity_stress`, or `exclusion_log` |
| `analysis_role`, `phase_state`, `quality_flag`, `transition_assignment`, `note` | Inclusion logic and material-state metadata |
| `*_direct` | Descriptors calculated on the stated atomic-composition basis |
| `*_legacy` | Legacy mass-normalized sensitivity calculation |
| `rule_pass_direct`, `pass::v2_250_350` | Fixed-family and full-development candidate outcomes |
| `model_pred_TC_K`, `model_abs_error_K` | Descriptive full-data model transfer; not an unbiased validation estimate |
| `nearest_training_distance_z`, `within_training_feature_ranges` | Descriptor-domain diagnostics |

## `data/processed/process_state_sensitivity.csv`

Paired as-rolled and annealed states for seven nominal compositions from the Rocha series. `label_changed` identifies whether processing changes membership in the 250–350 K target window.

## Results

- `results/development_windows/`: exact percentile bounds, record-level rule selections, and Table 2 metrics.
- `results/grouped_cv/`: fold assignments, best parameters, OOF predictions, fold metrics, and aggregate summaries for 3 descriptor families × 7 models.
- `results/interpretation/`: final LightGBM parameters and row-level TreeSHAP contributions.
- `results/rule_reassessment/`: 100 outer-fold rule selections, target-specific SHAP stability, repeated OOF predictions, frozen candidate definition, and literature-set outcomes.
- `results/literature_audits/`: historical-stratum and targeted-challenge summaries.
