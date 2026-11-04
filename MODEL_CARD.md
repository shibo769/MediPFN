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

# MediPFN v1 model card

## Model overview

**MediPFN: Amortized In-Context Nuisance Learning and Inference for Causal
Mediation Analysis** is a framework for estimating a natural indirect effect
and constructing an uncertainty interval from a labeled table. It combines a
custom support-query nuisance network, a cross-fitted mediation estimator,
learned standardized-error distributions, post-training, and independent
calibration. The retained deployment weights are fixed; a new table provides
the inference context rather than triggering gradient updates.

**Author:** Shi Bo, Boston University.  
**Version:** retained first-version production pipeline, archived 2026-11-03.  
**Status:** verified research checkpoint archive; separate from the new v2
rich-simulation development.  
**Implementation:** PyTorch; the retained backbone is custom MediPFN and does
not reuse a pretrained TabPFN backbone.

## Inputs and outputs

The v1 input is an observed table in the original `[X..., A, M, Y]` format and
a two-fold assignment. The original implementation validates its supported
variable types and dimensions. Users should inspect these checks before using
the checkpoint with a new data schema. Broader mediator representations belong
to v2 development and must not be inferred from this v1 archive.

The output contains a population natural indirect effect estimate, its
score-based standard error, and the frozen interval endpoints. Supported
nominal levels in this archive are 90%, 95%, and 99%. The intervals concern the
mediation effect parameter, rather than future individual outcomes.

## Training and inference

The nuisance network learns from factual labels in simulated tables. Its
deployment predictions enter a cross-fitted mediation score. Error models
subsequently learn the standardized estimation error using the known synthetic
effect. The final retained 95% model uses a selected quantile post-training
head and a frozen calibration expansion. The 90% and 99% levels have their own
retained heads and calibration coefficients.

Training, development, calibration, and evaluation roles are distinct. Source
code and asset SHA-256 values are available in [the manifest](v1/manifest.json)
and [production bindings](v1/production_bindings.json). Historical filenames
containing `tabcon`, `dpo`, or `rlvr` do not establish a separate original
TabCon/RL model comparison. The restored retained mixture uses its frozen SFT
router.

The new v2 work will use a different simulation bank and model interface. Its
checkpoints, split manifests, and validation results must be recorded separately
and cannot replace these v1 files.

## Evaluation

The completed shared evaluation bank contains **2,952 tables**: 360 in the
in-distribution setting, 1,872 using real covariates with simulated treatment,
mediator and outcome generation, and 144 in each of five stress settings.
There are two sampled tables per scenario mechanism. Aggregate results and
paired comparisons are stored in [v1/results](v1/results).

At the 95% nominal level:

| Evaluation setting | Coverage | Mean interval length | Tables |
|---|---:|---:|---:|
| In distribution | 96.67% | 0.4348 | 360 |
| Real covariates with simulated responses | 92.95% | 0.4921 | 1,872 |

The retained method improves coverage over the five full-bank comparators in
the seven evaluated settings, while producing longer intervals. Higher
coverage alone is not a complete efficiency comparison: the supplied tables
retain interval lengths, failure counts, and unfavorable comparisons.

A separate paired full-process CPU timing study reported a median of
**4.73 seconds** for the retained MediPFN pipeline and **450.35 seconds** for
100-replicate RF g-computation bootstrap across ten measured cases, with four
allocated CPU slots per case. This includes process startup, imports,
table reading, model loading/fitting, inference, output and process exit; it
excludes simulation and scheduler waiting. It is not a general runtime claim
for other hardware, table sizes or implementations.

## Intended use and assumptions

This is a research artifact for causal mediation estimation under appropriate
identification and overlap conditions. Causal interpretation depends on the
scientific design and assumptions about treatment, mediator, outcome and
confounding; network pretraining does not establish those assumptions. The
reported coverage applies to the stated evaluation designs. New source
distributions, support violations or unsupported schemas require independent
evaluation.

The real-covariate experiments use simulated causal responses, so their known
effect labels are not identified effects from the corresponding original
real-world outcome data. Preserve this distinction when describing results.

## Checkpoints and reproducibility

See [CHECKPOINTS.md](CHECKPOINTS.md) for the seven retained `.pt` files and
their SHA-256 values. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) explains archive
verification, restoration and the scope of the included training code.
The archived weights passed a read-only restoration check with all 97 source
and asset hashes verified, followed by finite ordered intervals at all three
nominal levels on an independent synthetic table. That check is a restoration
test and does not estimate coverage.

## Access and redistribution

This repository is private. Raw study tables and third-party pretrained
TabPFN weights are not redistributed. A public release requires an explicit
license and a separate review of the rights associated with any included
third-party source material. No open-source license is granted by the model
card itself.
