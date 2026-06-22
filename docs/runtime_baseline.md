# Runtime baseline — dev spike

## Scope
This is a dev/test baseline, not a production freeze.

## Chosen runtime family
- Databricks Runtime: 17.3 LTS for ML
- Goal: use the same runtime family for scoring and training dev tests

## Why
- reduce environment drift between notebooks
- validate imports, MLflow, Delta and feature-related dependencies on one baseline first

## Rules
- avoid ad-hoc `%pip install` inside notebooks unless strictly necessary
- every extra dependency must be documented
- any package added beyond runtime defaults must be justified and pinned separately

## Success criteria
- 01_cdf_stream runs on the baseline runtime in dev
- training notebooks import and start correctly on the same runtime family
- required extra libraries are explicitly listed
- no hidden notebook-only dependency remains