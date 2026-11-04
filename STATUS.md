# Project status

Verified update: 2026-11-03 (America/New_York).

## MediPFN v1

The retained first-version code, seven checkpoint files and normalization/
calibration records are archived. All 97 production source/asset hashes passed
verification, all seven weights were downloaded back from GitHub and matched
their original SHA-256 values, and the restored pipeline returned finite
ordered intervals at the three retained levels on an independent synthetic
table. See [checkpoints](CHECKPOINTS.md) and the [restoration record](v1/restore_smoke_report.json).

## MediPFN v2

Factual nuisance pretraining **completed** on SCC under job `8425910`.
Actual scheduler accounting returned `failed=0` and `exit_status=0`.
The verified completed run records:

| Item | Observation |
|---|---:|
| Optimizer steps | 20,000 |
| Training-table draws | 160,000 |
| Represented training provenance families | 8 of 8 |
| Selected checkpoint step | 17,000 |
| Selected same-source DEV factual loss | 1.526461 |

The selected factual checkpoint and its configuration/training log are
[archived with a completion record](v2/checkpoints/factual/manifest.json).
This completes the factual stage. The nested-regression stage has also
completed 5,000 optimizer steps and 40,000 training-table draws under job
`8425970`, with actual scheduler `failed=0` and `exit_status=0`. Step 5,000
was selected with DEV pseudo-outcome loss 0.229999. Its teacher hash matches
the selected factual checkpoint.
The completed nuisance stages retain their original source digest
`65fff8193cab74881e14a3262c254578255a78d2f6ca56aa4d7f6086a25ed36d`
and [immutable manifest](v2/manifests/factual_65fff8193cab7488.json).
The current error-pipeline release extends those exact 59 source files with
nine error-learning, post-training, calibration and binding files. Its digest is
`475a2abf480e9e36bfebeac6d584dc6a5d5139377340fac2ea0c221e41daedf6`;
see the [current manifest](v2/release_manifest.json) and
[manifest index](v2/manifests/README.md).

| Stage | Status |
|---|---|
| Factual propensity/outcome nuisance pretraining | Completed and checkpoint archived |
| Nested-regression nuisance stage | Completed and checkpoint archived |
| Learned signed-error distribution | Production workflow job `8426136` submitted with dependency guards |
| Confidence-guided post-training | Sequential stage of job `8426136` |
| Fresh independent calibration | Sequential stage after model/post-training selection in job `8426136` |
| ID, source-OOD and composition-OOD evaluation | After final calibration |

Thirteen CPU contracts passed on SCC. GPU integration job `8426127` produced
a passing integration log; its final scheduler accounting still requires
verification. Production job `8426136` depends on that integration guard and
the completed nested stage. A log message or dependency release alone is not
a completed production result. Final learned-error weights and calibrated
interval results have not yet been archived.

## Optional pretrained-backbone experiments

The optional source release is
`67409e5e616eea452020131b7beef22de3c6833c1c441e2fa7b30df569d43408`.
The repository includes its scientific implementation, tests, pinned public
model metadata and license, with operational transport scripts and runtime
receipts excluded through an [explicit subset manifest](v2/manifests/optional_67409e5e616eea45_repository.json).

| Active job | Purpose | Current evidence |
|---|---|---|
| `8426159` | Actual TabPFN-v3 model/gradient and checkpoint fixture | Submitted; successful fixture execution not yet verified |
| `8426161` | Frozen V3 nuisance backbone with an NIE error head and independent calibration | Waiting for the fixture guard |
| `8426163` | Cross-task mediation input/output adapter, followed by its own NIE error head and calibration | Waiting for the fixture guard |

These are mediation adaptations. They are separate from the custom MediPFN v2
backbone and are not original TabCon reproductions. Official third-party
weights remain outside this repository. See the
[optional-backbone guide](v2/OPTIONAL_BACKBONES.md) for the pinned environment,
runtime gates and source distribution.

## Shared data split and checkpoint archive

The bank contains 15 registry records and 14 distinct complete-data presets.
Training uses 10 presets from eight provenance families. Eric and the two
Xia-Chan presets are source OOD; Jones NNN is composition OOD. These sources
remain outside training, DEV selection and calibration.

The factual and nested best checkpoints are stored directly in the repository.
Their optimizer-resume checkpoints are stored as private GitHub release assets.
The two selected models passed a joint CPU cross-fitted inference restoration
check on an independent 64-row generated table; all scores and score-Wald
endpoints were finite, and synthetic truth was not computed or supplied.
Later-stage checkpoints remain on SCC while their producers write them and
will be archived only after successful completion and verification. The v1
assets remain unchanged.
