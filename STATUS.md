# Project status

Verified update: 2026-11-04 (America/New_York).

## MediPFN v1

The retained first-version code, seven checkpoint files and normalization/
calibration records are archived. All 97 production source/asset hashes passed
verification. All seven weights were downloaded back from GitHub and matched
their original SHA-256 values; the restored pipeline returned finite ordered
intervals on an independent synthetic table. See the
[checkpoint index](CHECKPOINTS.md) and [restoration record](v1/restore_smoke_report.json).

## MediPFN v2 primary model

**Training, post-training and fresh ID task-mixture calibration are complete.**
The [primary bundle](v2/checkpoints/primary/bundle.json) binds the selected
factual, nested, learned-error and post-trained checkpoints to the final
calibration file and exact inference source. This is a trained research
prototype. Scientific ID/OOD tests under the fixed shared protocol are
complete and independently verified; [all aggregate results](evaluation/v2_shared_20261103/results/README.md)
retain favorable and adverse comparisons.

| Stage | Verified result |
|---|---|
| Factual nuisance pretraining, job `8425910` | Actual scheduler exit 0; 20,000 steps and 160,000 table draws; selected step 17,000 with DEV loss 1.526461 |
| Nested-regression stage, job `8425970` | Actual scheduler exit 0; 5,000 steps and 40,000 table draws; selected step 5,000 with DEV loss 0.229999; exact factual-teacher binding |
| Primary integration, job `8426127` | Thirteen CPU contracts and actual GPU integration passed; actual scheduler exit 0 |
| Error learning and post-training, job `8426136` | Completed training, post-training and cache generation; process exited 1 when the calibration output name collided with immutable cache metadata |
| Calibration finalizer, job `8426219` | Actual scheduler exit 0; used the same selected weights and fresh calibration cache, without retraining or selection; bound 200-row CPU inference restore passed |

The finalization correction changes only the workflow output filename to
`final_calibration.json`. The initial failure, successful finalizer, scheduler
records and asset identities are preserved in the
[verification record](v2/checkpoints/primary/verification.json).

Error-stage budgets are 10,000 training tasks, 1,024 DEV tasks, 1,024
post-training tasks, 512 post-training DEV tasks and 1,024 fresh iid calibration
tasks. The verified cache records contain zero generation failures. No ID/OOD
scientific test labels were used for selection or calibration.

The nuisance weights retain their original
[59-file source release](v2/manifests/factual_65fff8193cab7488.json).
Error/post-training weights were produced by the
[68-file error release](v2/manifests/error_475a2abf480e9e36.json).
The [current release](v2/release_manifest.json) changes only the workflow output
filename. All scientific estimator, error-model and calibration source bytes
remain unchanged. See the [manifest index](v2/manifests/README.md).

## Optional pretrained-backbone experiments

The real TabPFN-v3 fixture `8426159` passed actual classifier/regressor input
gradients, adapter updates across contexts, frozen-backbone identity and
checkpoint restoration. Actual scheduler accounting returned exit 0.

| Job | Purpose | Verified state at this update |
|---|---|---|
| `8426161` | Frozen V3 nuisance backbone with an NIE error head and fresh calibration | Complete, actual scheduler exit 0; source/cache/model audit and full CPU restoration passed; checkpoints archived |
| `8426163` | Cross-task mediation input/output adapter, then its own NIE error head and calibration | Complete, actual scheduler exit 0; real-gradient/source/cache/model audit and full CPU restoration passed; checkpoints archived |

These experiments use a separately frozen implementation. They are mediation
adaptations rather than original TabCon reproductions. Official third-party
weights are not redistributed. The [optional-backbone guide](v2/OPTIONAL_BACKBONES.md)
records the package, pinned weight metadata, licenses and distributed source
subset. Frozen-V3 fitting used 1,024 training, 384 DEV and 512 fresh calibration
tasks, with 1,920 disjoint mechanisms and zero failures. Its selected head and
calibration are bound in the [archive manifest](v2/checkpoints/frozen_v3/manifest.json).
The adapter completed 2,000 real-gradient steps on 8,000 independent factual
tables; its selected step 500 was chosen on 16 independent DEV mechanisms.
Its own 1,024/384/512-task head/DEV/calibration stages and full 40-row CPU
restoration also passed. The [adapter archive](v2/checkpoints/adapter_v3/manifest.json)
preserves those bindings and all trained components. All three model routes
are trained, restored and backed up. Their shared scientific evaluation is
complete, with actual accounting and independent scalar recomputation passed. See the [adaptations model card](MODEL_CARD_V3_ADAPTATIONS.md).

## Data split and checkpoint archive

The bank contains 15 registry records and 14 distinct complete-data presets.
Training uses 10 presets from eight provenance families. Eric and the two
Xia-Chan presets are source OOD; Jones NNN is composition OOD. Those presets
remain outside fitting, DEV selection and calibration.

Selected primary weights, final calibration, configuration and training history
are archived under [`v2/checkpoints/`](v2/checkpoints/). Larger nuisance
optimizer-resume states are private GitHub release assets. Both optional
models, trained heads and matching calibrations are also archived and verified. The v1 assets and manuscript remain unchanged.

## Shared ID/OOD evaluation: complete and independently verified

Array `8427083` completed all 96 tasks with actual `failed=0`, `exit_status=0`.
Summary `8427084` and independent audit `8427085` also completed successfully.
The audit recomputed 1,008 metric and 1,008 paired cells. All 3,456 method
attempts succeeded across 1,152 shared tables and 10,368 interval levels.
The bank contains 576 structural populations: 768 ID, 288 source-OOD and
96 composition-OOD tables. Model, source, bank, folds and output hashes passed.

At the primary 95% level, custom v2 has ID coverage 95.99% and mean interval
length 0.5816, versus frozen V3 96.72%/0.9547 and adapted V3 96.46%/0.7691.
The custom route has about 21–24 times lower already-loaded per-table inference
time. Its source-OOD intervals are wider, and composition-OOD coverage is only
73.96%, versus 92.71% for both V3 routes. There is no universal best-model claim.

[Complete results](evaluation/v2_shared_20261103/results/README.md) preserve
all three levels, source and sample-size cells, failure denominators,
structural-population paired uncertainty and separate startup records.
No test-driven model selection or calibration was performed. The inherited
V3 pretraining source distribution is not controlled by our source holdout.
Raw test tables, per-case truth/endpoints, full task seeds and authenticated
transport receipts are excluded. The v1 manuscript and bibliography are unchanged.
