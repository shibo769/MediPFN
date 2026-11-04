---
language:
- en
library_name: pytorch
tags:
- causal-inference
- causal-mediation
- in-context-learning
- tabular-data
- uncertainty-quantification
---

# MediPFN v2 model card

## Model and training status

MediPFN v2 is a new custom support-query network for causal mediation nuisance
learning across literature-derived simulation tasks. Factual nuisance
pretraining completed on SCC under job `8425910`, with 20,000 optimizer steps
and 160,000 training-table draws across all eight training provenance families.
Step 17,000 was selected by independent same-source DEV factual loss (1.526461).
Actual scheduler accounting recorded successful completion. Final effect-estimation and interval results
follow completion of the remaining uncertainty stages.

Nested-regression training also completed under job `8425970`: 5,000 optimizer
steps and 40,000 training-table draws, with selected DEV pseudo-outcome loss
0.229999 at step 5,000. Its frozen factual-teacher SHA-256 matches the selected
factual checkpoint exactly. The two nuisance checkpoints passed a joint
cross-fitted inference restoration check on an independent 64-row generated
table, without computing or supplying its true effect.

The implementation is initialized from new weights. Optional TabPFN-based
variants are separate experiments and do not identify this custom backbone.
The first-version model and results are preserved in [`v1/`](v1/).

Each custom nuisance network has **4,217,345 parameters**. Factual and nested
models share the architecture and retain separate trained weights.

## Target and data

The common target is the superpopulation natural indirect effect
`E[Y(1, M(1)) - Y(1, M(0))]`. Input roles distinguish baseline covariates,
treatment, mediator coordinates and outcome. The new interface supports scalar
and vector mediators with type masks.

The simulation bank adapts **15 registry records into 14 distinct complete-data
presets** from the frozen upstream commit
`8182caf7185845a31a2122e032811bb2254d6342`. The two Shan observation schemes
share one complete-data mechanism. Ordered and dynamic designs are excluded
from this version. Treatment-induced confounder paths are removed structurally
where required; complete variables are exported before missingness masks; the
common natural-effect truth is recomputed after adaptation.

| Partition | Presets | Provenance families |
|---|---|---:|
| Training and independent same-source ID partitions | Cheng, Diaz-Hejazi, Diaz intermediate, He-Song-Xu, Jones LNL/LNN/correlated, Miles, Shan, Zuo | 8 |
| Source OOD | Eric; Xia-Chan discrete and continuous | 2 held-out families |
| Composition OOD | Jones NNN | Jones is seen through other presets |

Training samples a family uniformly and then a retained preset within that
family. Distinct registry records therefore do not automatically receive
greater weight. Source-OOD and composition-OOD presets are excluded from
fitting, DEV selection and calibration. Details and upstream provenance are in
the [generator guide](v2/dgp/README.md).

## Estimation and uncertainty stages

Factual training learns treatment probabilities given baseline covariates or
baseline-plus-mediator values, and outcome regressions. Their Bayes relation
provides the mediator distribution ratio. A separate nested-regression stage
uses inner out-of-fold pseudo-outcomes from the frozen factual teacher.

The restored nuisance estimator uses outer cross-fitted mediation scores. A
separate model learns signed, standardized estimator errors from independently
generated synthetic tasks. Post-training and fresh calibration follow model
selection. Known synthetic effects are offline supervision/evaluation targets;
deployment uses the observed table and frozen model outputs.

The current signed-error workflow is submitted as job `8426136`, with the
completed nested checkpoint and integration guard as dependencies. Thirteen
CPU contracts passed, and the GPU integration log reports a pass; final
scheduler accounting and production completion remain separate checks.
The implementation binds inference to the actual table, estimator source,
selected factual/nested checkpoints, feature schema and calibration records.

The primary interval level is 95%, with 90% and 99% as secondary levels. Final
evaluation will report coverage, interval length, point error, failures and
runtime, with paired uncertainty grouped by mechanism. There are no completed
v2 evaluation results in this source snapshot.

## Frozen source and configuration

Factual/nested source digest:
`65fff8193cab74881e14a3262c254578255a78d2f6ca56aa4d7f6086a25ed36d`.

The registered factual budget is 20,000 optimizer steps with effective batch
size 8, sample sizes 200/600/1000, 64 held-out query rows per task, and learning
rate 0.0003. Selection uses independent same-source DEV factual loss. Runtime
settings use mixed precision, TF32, shared inducing-token contexts and
asynchronous data preparation; final measured runtime will be reported with
the actual resource and timing boundaries.

The error-learning, post-training and calibration extension is frozen as
`475a2abf480e9e36bfebeac6d584dc6a5d5139377340fac2ea0c221e41daedf6`.
All 59 factual/nested source file hashes are unchanged. The
[original nuisance manifest](v2/manifests/factual_65fff8193cab7488.json)
remains available independently of the current release.

See the [current release manifest](v2/release_manifest.json),
[protocol](v2/protocol.json), and [current status](STATUS.md). The protocol's
status field records the source-freeze point; current execution status is
maintained separately. Optional pretrained TabPFN-v3 variants have their own
[source and runtime guide](v2/OPTIONAL_BACKBONES.md), pending an actual-model
fixture. They do not replace or rename the custom nuisance backbone.

## Checkpoints and access

The completed factual-stage [selected checkpoint](v2/checkpoints/factual/best.pt)
has SHA-256 `722172242bb4f80957b02a9f2227d557d29379db20906a63a5e5e2aa01a2d66b`.
The [checkpoint manifest](v2/checkpoints/factual/manifest.json) binds its source,
configuration, training log and completion record. The larger optimizer-resume
checkpoint is a [private release asset](https://github.com/shibo769/MediPFN/releases/tag/v2-factual-20261103).

The [selected nested checkpoint](v2/checkpoints/nested/best.pt) has SHA-256
`82fc00689e50526f130263beebecd09afefe4ab4cbe1fccf7239dd599d659cd7`.
Its [manifest](v2/checkpoints/nested/manifest.json) preserves the teacher binding
and verified completion record; the optimizer-resume state is a separate
[release asset](https://github.com/shibo769/MediPFN/releases/tag/v2-nested-20261103).
The [joint restoration record](v2/checkpoints/nuisance_restore_smoke.json)
checks complete finite scores from the two selected models.

These checkpoints provide the nuisance estimator; learned-error training,
post-training and fresh calibration remain separate stages. The private repository preserves
upstream licenses and notices; raw observed test banks and authentication
material are excluded.
