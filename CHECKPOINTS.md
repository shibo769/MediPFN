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

## MediPFN v2 factual-stage checkpoint

Factual nuisance pretraining completed with 20,000 optimizer steps and 160,000
training-table draws. The selected checkpoint is from step 17,000. It supplies
factual nuisance predictions; the final NIE interval pipeline requires the
subsequent nested-regression, uncertainty and calibration stages.

| Asset | Bytes | SHA-256 |
|---|---:|---|
| [Selected factual checkpoint](v2/checkpoints/factual/best.pt) | 27,005,256 | `722172242bb4f80957b02a9f2227d557d29379db20906a63a5e5e2aa01a2d66b` |
| [Last optimizer-resume checkpoint](https://github.com/shibo769/MediPFN/releases/download/v2-factual-20261103/factual-seed0-last.pt) | 62,627,426 | `fd83b1aa2c32fdf7b8933da5583b3cdfb6d0e111bb98b2a702dc36033490d3a1` |

[The factual checkpoint manifest](v2/checkpoints/factual/manifest.json) records
the frozen source digest, configuration, training log, source-family counts,
selection record and actual scheduler-completion evidence. The larger resume
file is a private [release asset](https://github.com/shibo769/MediPFN/releases/tag/v2-factual-20261103).

## MediPFN v2 nested-regression checkpoint

The nested stage completed 5,000 optimizer steps and 40,000 training-table
draws. Its frozen teacher is the factual best checkpoint above. The selected
factual/nested pair passed a joint native-table cross-fitted inference restore
check. The completed primary uncertainty and calibration assets are indexed
below; the historical nested-stage manifest preserves its original stage status.

| Asset | Bytes | SHA-256 |
|---|---:|---|
| [Selected nested checkpoint](v2/checkpoints/nested/best.pt) | 16,939,208 | `82fc00689e50526f130263beebecd09afefe4ab4cbe1fccf7239dd599d659cd7` |
| [Last nested optimizer-resume checkpoint](https://github.com/shibo769/MediPFN/releases/download/v2-nested-20261103/nested-seed0-last.pt) | 49,429,126 | `34372ad78109131c9293b044ebe25a9b45b8e2461ac26be132a41bf8056a8a7d` |

See the [nested manifest](v2/checkpoints/nested/manifest.json) and
[joint restoration record](v2/checkpoints/nuisance_restore_smoke.json).

## Complete primary v2 inference bundle

The [bundle manifest](v2/checkpoints/primary/bundle.json) references the archived
factual and nested weights and adds the completed uncertainty assets:

| Role | File | SHA-256 |
|---|---|---|
| Selected signed-error head | [error_best.pt](v2/checkpoints/primary/error_best.pt) | `ea603ec80c1f70238a14bfad9fa24576cf7685a17686d34cac5fa75f9f52f3ed` |
| Selected post-trained head | [post_best.pt](v2/checkpoints/primary/post_best.pt) | `2a8e8c3ac56ceb34e718acc29838ec4b0a1785fd62c92b1dced910e856bcc3c3` |
| Final fresh calibration | [final_calibration.json](v2/checkpoints/primary/final_calibration.json) | `dc857f520a4af6405d1e26922d28faae677ee294fb232b97f92127b5fda22427` |

The [completion](v2/checkpoints/primary/completion.json),
[actual accounting verification](v2/checkpoints/primary/verification.json),
configuration and training histories remain beside the bundle. The initial
workflow filename collision and successful finalizer are preserved; the
finalizer did not retrain or reselect the model. Run the following from the
repository root for full model/calibration/source verification:

```sh
python v2/predict.py --verify-only
```

The complete
[private release](https://github.com/shibo769/MediPFN/releases/tag/v2-primary-20261103)
is a trained research prototype, with scientific ID/OOD evaluation pending.

## Optional frozen-V3 NIE head and calibration

The [optional manifest](v2/checkpoints/frozen_v3/manifest.json) binds trained head
files, fresh calibration, selection history, source and verified completion.
The official V3 classifier/regressor weights are separate licensed inputs;
they are not mirrored here.

| Asset | Bytes | SHA-256 |
|---|---:|---|
| [Selected NIE error head](v2/checkpoints/frozen_v3/best.pt) | 1,365,340 | `df406ce3d43bc097860473ba83817cb19f3e48e23d6dfabece0bfcd545e40ba3` |
| [Last head state](v2/checkpoints/frozen_v3/last.pt) | 1,365,212 | `c70f4655b29dcac72fbe5ced7e546e057d0df57627bf0d75b299d15ffd19b859` |
| [Fresh ID calibration](v2/checkpoints/frozen_v3/calibration.json) | 3,195 | `e013c2d849375d9ad5eeaee7fd1ca03234fa7b34871678e4faa962e8917df3cc` |

Actual scheduler exit 0 and a full CPU observed-table restoration were
independently verified. These checks establish a restored trained model;
scientific ID/OOD coverage and runtime evaluation remain pending. See the
[model card](MODEL_CARD_V3_ADAPTATIONS.md) for the licensed-backbone requirements.

## Optional V3 mediation adapter and its own NIE head

The [adapter-route manifest](v2/checkpoints/adapter_v3/manifest.json) preserves
the selected/last adapter, its separate selected/last NIE head, calibration,
configuration and independent actual-accounting/full-restoration proof.

| Asset | SHA-256 |
|---|---|
| [Selected adapter](v2/checkpoints/adapter_v3/adapter/best.pt) | `dbc4c0fbc7c099e665ea65e8b26c9594a721711bef7825aed7cb5f56cf79f201` |
| [Selected matching NIE head](v2/checkpoints/adapter_v3/head/best.pt) | `ca7c0bf657fe6c927799daa4e74c05de3efe790a8419c11665f559c50f87a7e3` |
| [Fresh matching calibration](v2/checkpoints/adapter_v3/head/calibration.json) | `dcef6f5e668b0756d78f9a7d3a4acfc1ce22bc994711e69536f5991537b66ad3` |

Both optional bundles and their scientific source/license references are in the
[private adaptations release](https://github.com/shibo769/MediPFN/releases/tag/v2-v3-adaptations-20261103).
Official backbone weights remain separate licensed inputs. Scientific ID/OOD
evaluation is pending; restored training artifacts are not coverage results.
