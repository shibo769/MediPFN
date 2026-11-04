# MediPFN

Private research archive for MediPFN, created on 2026-11-03 before starting the
new simulation-bank version. The retained first-version model and the new v2
development must remain in separate directories.

## Retained first version

`v1/project/` preserves the actual SCC production source files and model assets.
`v1/manifest.json` records every archived file's byte length and SHA-256. The
checkpoint selection is bound to the completed evaluation protocol with SHA-256
`3da1386f3d019bfd9287f07bc04f0330888e16c4e15db479370197b10dfbda83`.

The seven `.pt` files contain the retained nuisance backbone, error mixture,
confidence model, selected 95% post-training model/selection states, and the 90%
and 99% heads. The corresponding normalization values and calibration expansion
coefficients are preserved in the checkpoint files and frozen JSON records.
The path `capacity_full/large_seed0` is a historical directory name; the retained
backbone identity is determined by its checkpoint configuration and SHA-256,
not by that directory name. Legacy small models and fictional fixture weights
are excluded.

The historical source directory called `tabcon` contains MediPFN's own mixture
implementation. Its name does not identify an original TabCon implementation or
an independently trained RL comparison.

## Verify and restore

Clone this private repository using an authenticated GitHub account, then run:

```sh
python v1/restore.py --verify-only
```

Inference requires Python 3.12 or later, NumPy, PyTorch, SciPy and scikit-learn.
The completed comparisons used PyTorch 2.5.1 and NumPy 2.2.6. In an environment
with those dependencies, the synthetic restore check is:

```sh
python v1/restore.py --smoke
```

This check loads the archived production weights, creates an independent small
synthetic table, and verifies finite intervals at all three retained levels. It
is an installation/restore check, not a coverage experiment.

Python callers can import `v1/restore.py` and call
`predict(rows, fold_ids, level=.95)`. The original v1 input convention and model
support remain unchanged. `restore.py` verifies all archived bytes before
loading the model. It resolves historical absolute references within the local
archive without rewriting the original source files or checkpoints.

## Scope

The archive includes training/inference code, retained checkpoint assets, and
aggregate verified comparison tables. It excludes passwords, authentication
helpers, raw observed datasets, test-table banks, temporary environments,
fictional model fixtures, and third-party TabPFN weights. Original SCC launch
and audit scripts retain their historical paths and guards; they are preserved
for provenance and are not intended to be run unchanged on another cluster.

The new rich-simulation MediPFN v2 will be added separately under `v2/` as its
implementation and training become available. Its development must not replace
or relabel the retained first-version weights.
