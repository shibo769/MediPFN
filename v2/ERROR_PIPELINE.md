# MediPFN v2 learned-error pipeline

This implementation is separate from the frozen factual training release. It
has passed local Python syntax checks. Its torch contracts and real-checkpoint
integration must pass in the SCC environment before production release.

## Stages and data separation

1. `train.py` fits treatment propensity `pi(X)`, augmented propensity `g(X,M)`,
   and factual outcome regression `mu(X,A,M)` using observed labels only.
2. `train_nested.py` freezes the selected factual network and fits a separate
   network for `eta0(X)` and `eta1(X)`. Inner out-of-fold predictions of
   `mu(X,1,M)` are its support responses. Outer-query outcomes are unavailable to
   these predictions. The factual teacher is never updated in this stage.
3. `train_error.py cache` freezes both networks, constructs an outer cross-fitted
   NIE estimate, and saves observed-only features and signed-error responses.
   The response is `(estimate - true NIE) / scale`; true NIE is never a feature.
4. `train_error.py fit` fits signed-error quantiles and selects a checkpoint by
   independent same-source development pinball risk. Calibration and source-OOD
   data are unavailable to this selection.
5. `posttrain_error.py` fits a coverage-confidence head on fresh post-training
   tasks. It predicts coverage across a positive scale grid; independent
   post-training DEV selects the head and its temperature. A small correction
   network then adjusts a shared central location and positive lower/upper
   widths under source-group/mechanism-balanced quantile risk. DEV selection
   limits loss of ID coverage and interval inflation relative to the confidence
   reference, then favors the weakest source-group coverage margin.
6. `calibrate.py` expands the selected post-trained intervals using a fresh iid
   calibration sample of tasks from the registered ID source-group/sample-size
   mixture. Every subsequent model or post-training change requires fresh
   calibration.

The stages retain the earlier method's confidence-guided scaling and
group-balanced endpoint correction. Their implementation is a v2 adaptation,
not a byte-for-byte port: one ordered quantile head replaces the earlier
global/specialist mixture; the correction jointly handles 90%, 95%, and 99%;
and the eight training source groups replace the earlier four correction
groups. Strict source-OOD mechanisms are never used for post-training. A
global/specialist distributional mixture remains a possible later addition.
This code does not implement an RL router or a TabPFN backbone.

## Observed features and interval definition

Features are the frozen factual network's pooled outer-support context and
twelve observed summaries: sample size, estimate and standard-error scales,
score skewness/kurtosis/central quantiles, propensity clipping fractions,
maximum estimated density ratio, and observed treatment fraction. Neither
source identity, mechanism coefficients, nor true NIE enters the model.

For the raw score standard error `s` and observed outcome standard deviation
`s_y`, the registered scale is

`scale = max(s, relative_scale_floor * max(s_y, 1e-6))`.

The same scale is used in the response, the learned interval, and calibration.
For fitted signed-error quantiles `(q_lo, q_hi)`, the interval is

`[estimate - scale*q_hi, estimate - scale*q_lo]`.

The head produces the 90%, 95%, and 99% intervals jointly. Positive cumulative
left/right widths ensure ordered quantiles and nested confidence intervals.
Independent calibration uses the score

`max(q_lo - signed_error, signed_error - q_hi)`

and order-statistic rank `ceil((m+1)*(1-alpha))`. Nonnegative expansions are
made nondecreasing across the three confidence levels to preserve nesting.
This extra expansion can only enlarge each separately calibrated interval.

The calibration argument concerns **marginal coverage of exchangeable tasks
from the registered ID mixture**, conditional on all frozen models. It is not
a per-source, conditional, or arbitrary-OOD guarantee. Per-mechanism tables in
the final benchmark still require paired/clustered statistical reporting.

## Cache and provenance safeguards

- Training/DEV feature caches balance the eight training source groups; the
  calibration cache draws source groups and sample sizes iid from the same
  registered uniform mixture.
- New error/post-training/calibration phase-specific seeds exceed 160 bits, whereas the factual phase uses
  at most 65-bit seed values. Train, DEV, and calibration seed namespaces are
  distinct; mechanism fingerprints are also checked for prohibited overlaps.
- Source/mechanism identities determine balanced training weights and DEV
  summaries only. Confidence and correction inference use observed features,
  initial quantile geometry, and the confidence level.
- Each cache binds the factual and eta checkpoint hashes, source/config
  hashes, feature order, score settings, scale floor, table/mechanism seeds,
  source provenance, and successful denominator.
- A failed feature extraction stops the run and retains a failure report. It
  does not silently replace a difficult mechanism with another task.
- Cache files, checkpoints, and calibration assets are protected against
  accidental overwrite. Feature extraction is performed once per task; fitting
  the small quantile network reuses the frozen cache.
- Cache loading checks all array/provenance denominators, the permitted
  training-source registry and commit, mechanism-parameter hashes, exact stage
  seeds, group counts, and consistency of `(estimate-truth)/scale` with the
  cached signed-error response. Every fitting stage requires the same frozen
  feature construction, model hashes, score settings, and sample-size grid.

## Example commands

Run from the isolated v2 release directory. The following filenames illustrate
the dependency order; they do not assert that any checkpoint already exists.

```bash
python error_tests.py

python train_error.py cache --checkpoint factual/best.pt \
  --eta-checkpoint nested/best.pt --split train --cases 10000 \
  --output error_features/train.npz --device cuda --cpu-threads 8

python train_error.py cache --checkpoint factual/best.pt \
  --eta-checkpoint nested/best.pt --split dev --cases 1024 \
  --output error_features/dev.npz --device cuda --cpu-threads 8

python train_error.py fit --train-cache error_features/train.npz \
  --dev-cache error_features/dev.npz --output error_fit --device cuda

python train_error.py cache --checkpoint factual/best.pt \
  --eta-checkpoint nested/best.pt --split post_train --cases 1024 \
  --output error_features/post_train.npz --device cuda --cpu-threads 8

python train_error.py cache --checkpoint factual/best.pt \
  --eta-checkpoint nested/best.pt --split post_dev --cases 512 \
  --output error_features/post_dev.npz --device cuda --cpu-threads 8

python posttrain_error.py --error-checkpoint error_fit/best.pt \
  --train-cache error_features/post_train.npz \
  --dev-cache error_features/post_dev.npz --output posttrain --device cuda

# Generate/use the final calibration sample after post-training is selected.
python train_error.py cache --checkpoint factual/best.pt \
  --eta-checkpoint nested/best.pt --split calibration --cases 1024 \
  --output error_features/calibration.npz --device cuda --cpu-threads 8

python calibrate.py --error-checkpoint posttrain/best.pt \
  --calibration-cache error_features/calibration.npz \
  --output calibration.json
```

For a quick integration fixture, use 24 training tasks, 24 DEV tasks, 24 fresh
post-training tasks, 24 fresh post-training DEV tasks, and 104 calibration
tasks. Use `--n-values 64`, two quantile-fitting epochs with `--width 16`, and
two confidence/correction epochs.
Such a fixture checks execution and bindings; it is not a scientific result or
a basis for choosing a production model.

Deploy with `BoundNIEEstimator(factual_checkpoint, eta_checkpoint, ...)` from
`error_binding.py`. Its `estimate(table, return_scores=True)` method loads the
actual checkpoint bytes and tags results with checkpoint, imported-source,
and observed-table hashes. Load the selected calibrated uncertainty model
with `load_calibrated_error(...)`, then call `predict_intervals(...)` with the
same table and bound result. Raw unbound `estimate_table` results are rejected.
The deployment checks reject mismatched nuisance/error checkpoints, point
estimator source, probability floor, folds, feature order, and observed table.
