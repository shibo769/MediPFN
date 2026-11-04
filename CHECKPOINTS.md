# Retained checkpoints

All files below are the actual retained first-version production artifacts.
They are stored directly in Git because each file is below the GitHub file-size
limit. They are not placeholder fixtures or links to unavailable weights.

| File | Role | Bytes | SHA-256 |
|---|---|---:|---|
| [confidence.pt](v1/project/capacity_4m_v10_20261028/confidence_calibration_4m_20261028/fitted/confidence.pt) | Temperature-scaled interval-coverage confidence network | 81,000 | `e7a80da447fbd4e61264d154ccb2c32ecb5c4a3ce27a5d0914ec3f811c4139ac` |
| [mixture.pt](v1/project/capacity_4m_v10_20261028/mixture_top2/mixture.pt) | Error experts, retained SFT router, training normalization | 1,971,124 | `19c37e2a5b45549bd56d29214aa9c26919fa1c8c2163e7ed850b1d1c8922bf3d` |
| [head_90.pt](v1/project/capacity_4m_v10_20261028/multilevel_4m_20261028/trained_heads/head_90.pt) | 90% quantile head and normalization | 48,848 | `1fa7426659ad78176c9ad395735fceb7b7af2103c037824cf4fc182fdbe596af` |
| [head_99.pt](v1/project/capacity_4m_v10_20261028/multilevel_4m_20261028/trained_heads/head_99.pt) | 99% quantile head and normalization | 48,848 | `fdf705bf8f1203a314e417bebe7a4b9aa8d0ab530ee1bd7c7af292fb9795f63a` |
| [model.pt](v1/project/capacity_4m_v10_20261028/robust_partial_composition_retry_20261028/trained_v1/model.pt) | 95% selected-head descriptor | 1,496 | `6cbd813b84513f43d003f970a50cf202ca5c4d46b408be7d886def72189b69ed` |
| [selection.pt](v1/project/capacity_4m_v10_20261028/robust_partial_composition_retry_20261028/trained_v1/selection.pt) | 95% candidate states, normalization, selection metadata | 170,638 | `d7e35ca89fe6829a5630307e713d6b443f8c976c8799c5627cf357574fa57612` |
| [model.pt](v1/project/capacity_full/large_seed0/checkpoint/model.pt) | Custom nuisance backbone | 18,840,932 | `8310f18835db289350147e1ac011ef4096a6e6984407de7406f96e127e1cb1dc` |

The small selected `model.pt` is a descriptor, rather than a second nuisance
backbone: its learned candidate states are stored in `selection.pt`. The seven
files therefore should not be described as seven independently trained models.

The calibration records are archived alongside the weights. The actual 90% /
99% CPU calibration freeze is `alpha_calibration_freeze.json`; the 95% selected
descriptor preserves its frozen expansion coefficient. Normalization arrays are
inside the relevant `.pt` files.

Run `python v1/restore.py --verify-only` after cloning. The full original source
and asset bindings are recorded in [v1/manifest.json](v1/manifest.json).
All seven files were independently downloaded back from the private GitHub
repository after the first push and matched against these SHA-256 values.

No trained v2 checkpoint has been released yet. Any future v2 checkpoint must
identify its source-release digest, training configuration, data/split identity,
selection record and calibration state.
