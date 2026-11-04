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
This completes the factual stage; final NIE interval results require the
subsequent stages.
The registered source digest is
`65fff8193cab74881e14a3262c254578255a78d2f6ca56aa4d7f6086a25ed36d`.
The unchanged source release and design are preserved in
[v2/release_manifest.json](v2/release_manifest.json) and
[v2/protocol.json](v2/protocol.json).

| Stage | Status |
|---|---|
| Factual propensity/outcome nuisance pretraining | Completed and checkpoint archived |
| Nested-regression nuisance stage | Started under job `8425970`; completion verification pending |
| Learned signed-error distribution | Implementation/integration validation |
| Confidence-guided post-training | Implementation/integration validation |
| Fresh independent calibration | After the model and post-training choices are frozen |
| ID, source-OOD and composition-OOD evaluation | After final calibration |

The bank contains 15 registry records and 14 distinct complete-data presets.
Training uses 10 presets from eight provenance families. Eric and the two
Xia-Chan presets are source OOD; Jones NNN is composition OOD. These sources
remain outside training, DEV selection and calibration.

The factual best checkpoint is stored directly in the repository. Its larger
optimizer-resume checkpoint is stored as a private GitHub release asset.
Later-stage checkpoints remain on SCC while their producers write them and
will be archived only after successful completion and verification. The v1
assets remain unchanged.
