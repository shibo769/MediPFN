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

The factual nuisance pretraining stage is **running** on SCC, job `8425910`.
The latest verified training snapshot records:

| Item | Observation |
|---|---:|
| Optimizer steps | 3,250 of the registered 20,000 |
| Training-table draws | 26,000 |
| Represented training provenance families | 8 of 8 |
| Observed same-source DEV factual loss at this snapshot | 1.55682 |

These are progress observations, not a completed model or final benchmark.
The registered source digest is
`65fff8193cab74881e14a3262c254578255a78d2f6ca56aa4d7f6086a25ed36d`.
The unchanged source release and design are preserved in
[v2/release_manifest.json](v2/release_manifest.json) and
[v2/protocol.json](v2/protocol.json).

| Stage | Status |
|---|---|
| Factual propensity/outcome nuisance pretraining | Running |
| Nested-regression nuisance stage | Subsequent stage |
| Learned signed-error distribution | Implementation/integration validation |
| Confidence-guided post-training | Implementation/integration validation |
| Fresh independent calibration | After the model and post-training choices are frozen |
| ID, source-OOD and composition-OOD evaluation | After final calibration |

The bank contains 15 registry records and 14 distinct complete-data presets.
Training uses 10 presets from eight provenance families. Eric and the two
Xia-Chan presets are source OOD; Jones NNN is composition OOD. These sources
remain outside training, DEV selection and calibration.

The active checkpoint files remain on SCC while the producer writes them.
A stable selected v2 checkpoint will be archived with its source/configuration
and SHA-256 after completion and verification. The v1 assets remain unchanged.
