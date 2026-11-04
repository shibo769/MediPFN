# MediPFN v2: rich mediation simulations

**Status: factual and nested nuisance training completed.**
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
artifacts and will be added after validation.

This version introduces a new custom support-query network for scalar and
vector mediator inputs, factual propensity/outcome learning, Bayes-based density
ratios, and a separate nested-regression stage. The frozen source release covers
the factual and nested nuisance stages; learned-error training, post-training,
fresh calibration and final evaluation are later stages.

The generator bank adapts 15 registry records into 14 distinct complete-data
presets, with a superpopulation natural indirect effect as the common target.
Eight provenance families supply training and independent ID partitions.
Eric and the Xia-Chan family are source OOD; Jones NNN is composition OOD.
These held-out presets must remain absent from training, development selection
and calibration. See [`dgp/README.md`](dgp/README.md) for adaptations, truth
calculation, parameter domains and source provenance.

The source is frozen by [`release_manifest.json`](release_manifest.json), release
SHA-256 `65fff8193cab74881e14a3262c254578255a78d2f6ca56aa4d7f6086a25ed36d`.
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
