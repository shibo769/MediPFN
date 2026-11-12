---
language:
- en
library_name: pytorch
tags:
- causal-inference
- causal-mediation
- tabular-data
- uncertainty-quantification
---

# Pretrained-V3 mediation adaptations

These optional models retain the official TabPFN-v3 nuisance backbone and
learn mediation-specific components from independent synthetic tasks. They
are separate from the custom MediPFN v2 primary model and are not original
TabCon reproductions. Their shared ID/OOD coverage, point-error and runtime
evaluation is [complete and independently verified](evaluation/v2_shared_20261103/results/README.md).
At 95%, frozen/adapted ID coverage is 96.72%/96.46%, with mean lengths
0.9547/0.7691. Source-OOD coverage is 94.79%/97.14%; composition-OOD coverage
is 92.71% for each. Both routes cover the composition holdout more often than
the custom route, while using longer ID intervals and slower per-table
inference. Full paired uncertainty, denominators and timing boundaries are
retained in the linked results.

## Frozen V3 with a learned NIE error head

Training and fresh ID task-mixture calibration completed under job `8426161`.
Actual scheduler accounting returned `failed=0`, `exit_status=0`. The
[checkpoint manifest](v2/checkpoints/frozen_v3/manifest.json) records the trained
head, calibration, configuration, selection history and source identities.

The point estimator fits treatment probabilities, outcome regressions and
nested regression from each observed support table using frozen V3 weights.
Cross-fitted mediation scores yield an NIE estimate and estimated scale. A
monotone six-quantile neural head learns signed standardized NIE-estimator
errors from pooled V3 query embeddings and observed score/table summaries.
Known simulated effects supply offline error targets; nuisance fitting and
deployment use observed data. The fitted head uses 1,038 input features.

The head was trained for 2,000 optimization steps on 1,024 tasks, selected on
384 independent DEV tasks, then calibrated on 512 fresh iid ID tasks. Train
and DEV contain equal numbers from each of eight training provenance families;
calibration draws families iid from their uniform mixture. The audit verified
1,920 disjoint mechanism identities, all partition denominators, recomputed
truth/error targets, source/configuration bindings and zero failures. Step 50
was selected with DEV pinball loss 0.069208; this is a selection loss, not a
scientific coverage result.

An independent full CPU restoration produced finite nested 90%/95%/99%
intervals on one 64-row observed-table fixture. Recomputed calibration
corrections agreed with saved values to within 7.16e-7. The
[verification record](v2/checkpoints/frozen_v3/verification.json) preserves the
actual accounting and restoration evidence. Fixture time is not a benchmark.

| Asset | SHA-256 |
|---|---|
| [Selected NIE head](v2/checkpoints/frozen_v3/best.pt) | `df406ce3d43bc097860473ba83817cb19f3e48e23d6dfabece0bfcd545e40ba3` |
| [Final ID calibration](v2/checkpoints/frozen_v3/calibration.json) | `e013c2d849375d9ad5eeaee7fd1ca03234fa7b34871678e4faa962e8917df3cc` |

## V3 with a cross-task mediation adapter

The second optional route completed under job `8426163`, with actual scheduler
`failed=0`, `exit_status=0`. It trains a nonlinear column input adapter and
prediction-output adjustment across independent factual tasks while retaining
the official V3 neural weights. Its own NIE error head and fresh calibration
are fitted after adapter selection. Deployment freezes all trained components;
there is no per-test gradient update.

Adapter fitting completed 2,000 positive-gradient optimizer steps on 8,000
reconstructed independent factual mechanisms. Selection used 16 independent
DEV mechanisms (two per training family), choosing step 500 with factual DEV
loss 0.511505. The audit verified positive actual gradients, parameter changes
and separation from the subsequent 1,920 error-stage mechanisms. The NIE head
uses 1,024 training tasks and 384 DEV tasks, selecting step 50 with DEV pinball
loss 0.068109, followed by 512 fresh iid calibration tasks. These selection
losses are not effect-estimation performance measures.

All source/model/partition bindings passed. A full 40-row CPU observed-table
restore produced finite nested 90%/95%/99% intervals using the selected adapter,
matching head, calibration and licensed V3 weights. See the
[adapter bundle](v2/checkpoints/adapter_v3/manifest.json) and
[independent verification](v2/checkpoints/adapter_v3/verification.json).

| Asset | SHA-256 |
|---|---|
| [Selected adapter](v2/checkpoints/adapter_v3/adapter/best.pt) | `dbc4c0fbc7c099e665ea65e8b26c9594a721711bef7825aed7cb5f56cf79f201` |
| [Selected matching NIE head](v2/checkpoints/adapter_v3/head/best.pt) | `ca7c0bf657fe6c927799daa4e74c05de3efe790a8419c11665f559c50f87a7e3` |
| [Matching fresh calibration](v2/checkpoints/adapter_v3/head/calibration.json) | `dcef6f5e668b0756d78f9a7d3a4acfc1ce22bc994711e69536f5991537b66ad3` |

## Runtime and licensed backbone

The pinned package is TabPFN 9.2.0 from official source commit
`953a4b916a19856fbd8dcda089a206b5bfe83696`. The official classifier/regressor
weights come from `Prior-Labs/tabpfn_3` at revision
`24a16a89d245878b846555110985634aa2e656d7`. Their required SHA-256 values are
recorded in each trained model's calibration/binding manifest. Third-party
weight binaries are not redistributed in this repository or its releases.
Obtain them from the official source under the retained
[research license](v2/tabular_backbones/v3_runtime/TABPFN-3-LICENSE).

The optional scientific code is bound to release
`67409e5e616eea452020131b7beef22de3c6833c1c441e2fa7b30df569d43408`.
The [repository distribution manifest](v2/manifests/optional_67409e5e616eea45_repository.json)
records included files and excluded cluster transport/receipt files. See the
[optional setup guide](v2/OPTIONAL_BACKBONES.md) for exact package and API gates.

After installing that environment and obtaining the licensed weights, inference
from the `v2` directory can use:

```python
import json
from pathlib import Path
from predict import load_observed_table
from tabular_backbones.frozen import BackboneConfig, make_shared_frozen_v3_factory
from tabular_backbones.train_uncertainty import calibrated_interval

folder = Path("checkpoints/frozen_v3")
cal = json.loads((folder / "calibration.json").read_text())
settings = dict(cal["backbone_binding"]["base_config"])
settings.update(classifier_path="/licensed/local/classifier.ckpt",
                regressor_path="/licensed/local/regressor.ckpt", device="cpu")
config = BackboneConfig(**settings)
factory = make_shared_frozen_v3_factory(config)
intervals = calibrated_interval(load_observed_table("observed_table.npz"), config,
                               folder / "best.pt", folder / "calibration.json",
                               factory=factory)
```

For the adapted route, use its own calibration-derived configuration and
selected adapter rather than the frozen-route factory:

```python
from tabular_backbones.adapter import load_adapter, adapted_factory

folder = Path("checkpoints/adapter_v3")
cal = json.loads((folder / "head/calibration.json").read_text())
settings = dict(cal["backbone_binding"]["base_config"])
settings.update(classifier_path="/licensed/local/classifier.ckpt",
                regressor_path="/licensed/local/regressor.ckpt", device="cpu")
config = BackboneConfig(**settings)
adapter_path = folder / "adapter/best.pt"
adapter, _ = load_adapter(adapter_path, config)
factory = adapted_factory(config, adapter)
intervals = calibrated_interval(load_observed_table("observed_table.npz"), config,
                               folder / "head/best.pt", folder / "head/calibration.json",
                               factory=factory, adapter_checkpoint=adapter_path)
```

These snippets use the imports and observed NPZ schema shown above. Weight
hashes, package/version and model settings must match the recorded
binding; only local paths and execution device should change. The deployment
entry point performs no gradient updates and accepts no truth argument.

## Scope

The data split and natural indirect effect target match the v2 generator bank.
Eric/Xia-Chan source holdouts and Jones NNN composition holdout are excluded
from mediation-specific fitting, selection and calibration. For these
pretrained routes, source OOD refers to our adaptation tasks; this study did
not control the official backbone's upstream pretraining distribution or
establish that those DGP forms were absent from it. The custom random-initialized
v2 model has a fully controlled training-source split. Calibration concerns the registered
iid ID task mixture, not conditional or arbitrary OOD coverage. It does not
replace causal identification/overlap assumptions. No accuracy or speed
advantage is inferred from successful training or restoration.

Both optional routes are trained, calibrated and restored. Their archived
learned components and source/license references are available in the
[adaptations release](https://github.com/shibo769/MediPFN/releases/tag/v2-v3-adaptations-20261103).
Current evidence is recorded in [STATUS.md](STATUS.md). The retained
custom primary model and first-version manuscript remain unchanged.
