# Project status

Verified update: 2026-11-03 (America/New_York).

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
prototype; scientific ID/OOD evaluation has not yet been performed.

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

| Active job | Purpose | Verified state at this update |
|---|---|---|
| `8426161` | Frozen V3 nuisance backbone with an NIE error head and fresh calibration | Running; feature-task generation observed |
| `8426163` | Cross-task mediation input/output adapter, then its own NIE error head and calibration | Running; adapter optimization with nonzero gradients observed |

These experiments use a separately frozen implementation. They are mediation
adaptations rather than original TabCon reproductions. Official third-party
weights are not redistributed. The [optional-backbone guide](v2/OPTIONAL_BACKBONES.md)
records the package, pinned weight metadata, licenses and distributed source
subset. Optional production completion and evaluation are still pending.

## Data split and checkpoint archive

The bank contains 15 registry records and 14 distinct complete-data presets.
Training uses 10 presets from eight provenance families. Eric and the two
Xia-Chan presets are source OOD; Jones NNN is composition OOD. Those presets
remain outside fitting, DEV selection and calibration.

Selected primary weights, final calibration, configuration and training history
are archived under [`v2/checkpoints/`](v2/checkpoints/). Larger nuisance
optimizer-resume states are private GitHub release assets. Later optional
checkpoints will be archived after their producers complete and their assets
are verified. The v1 assets and manuscript remain unchanged.
