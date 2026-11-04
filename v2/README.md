# MediPFN v2: rich mediation simulations

**Status: factual and nested nuisance training completed; the signed-error,
post-training and calibration workflow is submitted.**
Factual pretraining ran under SCC job `8425910`.
The completed run contains 20,000 optimizer steps and 160,000 training tables
across all eight source families. Step 17,000 was selected by same-source DEV
factual loss (1.526461). Its [checkpoint manifest](checkpoints/factual/manifest.json)
records the verified assets. The retained v1 weights and results are stored
separately in [`../v1`](../v1); final v2 interval evaluation follows the later
stages.

Nested-regression job `8425970` completed 5,000 steps and 40,000 training-table
draws. Its selected checkpoint is bound to the exact factual teacher and is
recorded in the [nested checkpoint manifest](checkpoints/nested/manifest.json).
The pair passed an independent native-table cross-fitted inference restore
check with finite scores and no synthetic truth supplied to inference.

For a small generated-table loading/inference check from this directory:

```sh
python evaluate.py --checkpoint checkpoints/factual/best.pt \
  --eta-checkpoint checkpoints/nested/best.pt --source he_song_xu \
  --n 200 --device cpu --output smoke_score.json
```

This entry point reports the point estimate and an uncalibrated score-Wald
interval. The learned-error and calibrated interval stages have separate
artifacts and will be added after successful completion and verification.

## Signed-error and calibrated intervals

The current [error pipeline](ERROR_PIPELINE.md) learns the standardized error
of the restored nuisance estimator, selects post-training corrections on fresh
development tasks, and calibrates on a new ID task mixture. The production
workflow is submitted as job `8426136`. Thirteen SCC CPU contracts passed;
the GPU integration log reports a pass, with its final accounting check still
separate from production completion. No final v2 coverage results are claimed.

The entry point from this directory is:

```sh
python run_error_pipeline.py --checkpoint checkpoints/factual/best.pt \
  --eta-checkpoint checkpoints/nested/best.pt \
  --output runs/error_seed0 --cpu-threads 8
```

This is a full CUDA training workflow with fresh outputs, not a smoke check.
Its frozen budgets are 10,000 error-training tasks, 1,024 error-DEV tasks,
1,024 post-training tasks, 512 post-training DEV tasks, and 1,024 independent
calibration tasks. OOD presets and scientific test banks are excluded from
those stages. `error_binding.py` checks table, source, checkpoint and feature
identities before calibrated inference.

Optional frozen-V3 and adapter variants are described separately in
[OPTIONAL_BACKBONES.md](OPTIONAL_BACKBONES.md). They await actual-model fixture
verification and do not change the custom backbone's completed checkpoints.

This version introduces a new custom support-query network for scalar and
vector mediator inputs, factual propensity/outcome learning, Bayes-based density
ratios, and a separate nested-regression stage. The frozen source release covers
the factual and nested nuisance stages. The current release adds learned-error
training, post-training and fresh calibration; scientific evaluation follows
their verified completion.

The generator bank adapts 15 registry records into 14 distinct complete-data
presets, with a superpopulation natural indirect effect as the common target.
Eight provenance families supply training and independent ID partitions.
Eric and the Xia-Chan family are source OOD; Jones NNN is composition OOD.
These held-out presets must remain absent from training, development selection
and calibration. See [`dgp/README.md`](dgp/README.md) for adaptations, truth
calculation, parameter domains and source provenance.

The current source is frozen by [`release_manifest.json`](release_manifest.json),
release SHA-256
`475a2abf480e9e36bfebeac6d584dc6a5d5139377340fac2ea0c221e41daedf6`.
The original nuisance-stage manifest is retained separately as
[`manifests/factual_65fff8193cab7488.json`](manifests/factual_65fff8193cab7488.json).
Its 59 source hashes are unchanged. Optional-source distribution has its own
manifest; see the [manifest index](manifests/README.md).
The run design is recorded in [`protocol.json`](protocol.json). Any code change
requires a new release digest and an explicitly identified training run.

From this directory, the local generator checks can be run with:

```sh
python -m dgp.test_generators
```

Training entry points are `train.py` and `train_nested.py`; inspect their
`--help` output for the source release's supported configuration. A production
run requires the separately recorded frozen configuration, resource profile,
successful preflight and fresh output directory. No scheduler session,
authentication material, raw test bank or environment is stored here.

The status in the immutable `protocol.json` describes the source-freeze point.
Current progress is recorded in [`../STATUS.md`](../STATUS.md) and the
[`v2 model card`](../MODEL_CARD_V2.md).

The upstream vendored materials remain private and retain their original
[`LICENSE`](dgp/vendor/LICENSE) and [`NOTICE`](dgp/vendor/NOTICE.md).
