# Optional pretrained tabular backbones

These files are separate from the custom MediPFN v2 run. No optional training job
has been submitted by these scripts. `optional_protocol.json` gives the order and gates.

## Actual v3 runtime now prepared on SCC

The isolated `medipfn_v2_20261103/env_tabpfn_v3` environment now contains TabPFN
9.2.0 from pinned commit `953a4b916a19856fbd8dcda089a206b5bfe83696`. Import checks
confirm V3 and both differentiable-input APIs. It shares the existing CUDA PyTorch
2.5.1 runtime read-only; new packages were installed only into the new environment.
The old v1 and v2 environments were not modified.

Both official V3 default checkpoints were downloaded from pinned HF revision
`24a16a89d245878b846555110985634aa2e656d7`. Their downloaded bytes match independent
HF LFS SHA256 metadata. The HF repository reported `gated=false`; no credentials or
form responses were supplied. The non-commercial research license, exact package
lock, configuration and bootstrap manifest are retained in `v3_runtime/`.
This establishes availability/importability; **actual model/gradient execution is
still gated on the GPU fixture** and must not be inferred from a completed download.

## What the source audit establishes

TabCon uses a frozen **TabPFN-v3** point-prediction backbone. Its Appendix C.3
describes a global residual head as an MLP on the frozen backbone's 512-dimensional
query embeddings; specialist heads and a router are trained later. Our proposed
mediation adaptation would learn the error of an NIE estimator, rather than the
error of a conditional-mean prediction at a query point. It must be called a
mediation adaptation, not original TabCon or an original TabCon reproduction.

* Primary paper: https://arxiv.org/html/2609.33114v1
* Pinned installed-API source: https://github.com/PriorLabs/TabPFN/tree/4103e892d8c3059183ac7c507b923924ac9b1091
* Current official source inspected: https://github.com/PriorLabs/TabPFN/tree/953a4b916a19856fbd8dcda089a206b5bfe83696

The initial SCC read-only inspection found package 6.0.6, supporting model versions v2 and
v2.5 only, and the previously audited v2 classifier/regressor checkpoints. There
is no v3 checkpoint in the inspected project weight directory or the inspected
home-cache directories. This is not a claim that no v3 file exists anywhere on SCC.
The isolated bootstrap above now supplies that missing environment and weights;
the original environment remains unchanged. Execution still refuses implicit
downloads or silent version fallback.

Code and weights have different licenses. Official TabPFN code is Apache 2.0.
The current README specifies a separate non-commercial license/access flow for v3
weights; v2 has the Prior Labs attribution license. No credentials have been
requested or displayed by these scripts. The public V3 download was performed for
the user-authorized academic experiment, with the research license retained.

## Implemented routes

`frozen.py` implements nested cross-fitted mediation nuisance regression with
any explicitly supported pinned v2/v3 checkpoint. It fits treatment probabilities
pi(X), treatment probabilities g(X,M), outcome means mu1(X,M), nested eta0(X), and
direct eta1(X). Bayes' rule supplies the joint mediator density ratio, including
vector mediators. No simulator truth enters nuisance fitting. An optional feature
path pools frozen outcome-backbone query embeddings for a task-level error head.
The four local mock contracts pass; they test data separation and orchestration,
not actual TabPFN performance.

`error_head.py` defines monotone, unbounded standardized-NIE-error quantiles with
a pinball objective. A fresh independent calibration stage is still required.
Fitted backbone identity and feature schema are bound to its checkpoint.

`adapter.py` and `train_adapter.py` now implement the preferred second route on
**actual frozen V3 weights**: a nonlinear shared column adapter and task-level
output adjustment trained across independent factual contexts. Fresh official
ModelSpecs wrappers share the frozen network by reference and create new context
preprocessing for each table, allowing variable feature counts without mixing
tables. Sixteen fixed DEV contexts (two per source family) select `best.pt`; the
last checkpoint is retained separately. Backbone weights remain frozen and
deployment invokes no optimizer.

`train_backbone.py` additionally implements genuine offline cross-task **TabPFN-v2** weight
fine-tuning using the installed release's official `get_preprocessed_datasets`,
`meta_dataset_collator`, `fit_from_preprocessed`, and `forward` APIs. The API accepts
a list of independent datasets; we pass one table per optimizer item and accumulate
gradients across independent contexts. Factual classification and outcome-regression
losses are optimized; there is no concatenation of unrelated DGP observations.
Training keeps source OOD/composition OOD out of its inputs. `finetuned.py` freezes
the trained weights and uses the official ModelSpecs cloning path for deployment,
preventing a normal table-specific `fit()` from resetting those weights.

The legacy 6.0.6 release rejects differentiable regression inputs. The modern
isolated environment supports that API. The V3 adapter implementation must pass
its actual checkpoint/gradient fixture before being described as validated.

## Mandatory real-model validation before training submission

For the V3 routes, in a compute allocation with the new environment and no old
TabPFN dependency paths:

```text
env_tabpfn_v3/bin/python -m tabular_backbones.test_v3_runtime --config tabular_backbones/v3_runtime/v3_local_weights_config.json --output optional_v3_fixture
```

The fixture tests actual gradients through the frozen V3 classifier and regressor
into the nonlinear adapter, updates across independent table dimensions, unchanged
backbone weights, and checkpoint prediction roundtrip. It has not yet executed.

For the additional legacy fine-tuning route, with the audited old dependency path:

```text
python -m tabular_backbones.test_real_gradients --config tabular_backbones/v2_local_weights_config.json --output optional_backbone_fixture
```

This fixture uses two small fictional independent contexts, checks actual
classifier/regressor gradients and weight changes, confirms parameter identity
across tables, saves/loads the checkpoint, verifies deployed normal `fit()` retains
the learned weights, and verifies error-head gradients/checkpoint roundtrip.
**It has not yet been executed.** A PASS report must come from real execution;
mock tests, source inspection and successful syntax compilation cannot replace it.

Initial optional fine-tuning budget is 2,000 updates, gradient accumulation 4,
256 rows/table, learning rate 1e-6 and one ensemble member. Profile one GPU first;
the optional run must wait for the custom v2 priority allocation. These are an
initial fixed budget and resource proposal, not a measured runtime or final model.

All model comparison results, including failures and unfavorable findings, remain
unavailable until the proper train/DEV/calibration/test pipeline completes.
