# Shared evaluation of three frozen mediation models

**Status: TESTS_SUBMITTED; array execution has begun.**
Task completion, complete accounting and scientific results remain unverified.
No comparative performance results are released yet.

This snapshot evaluates the completed custom MediPFN v2, frozen TabPFN-v3 with
its NIE error head, and V3 with the selected cross-task mediation adapter and
its own NIE head. All selected weights, interval settings and calibration
assets were fixed before these tests. Their identities are recorded in
[`frozen_model_hashes.json`](frozen_model_hashes.json) and the repository
[model-binding index](model_bindings.json).

## Fixed bank and allocation

| Stratum | Tables | Structural populations |
|---|---:|---:|
| Same-source ID | 768 | 384 |
| Source OOD | 288 | 144 |
| Composition OOD | 96 | 48 |
| Total | 1,152 | 576 |

Sample sizes are 200, 600 and 1,000. Each prespecified mechanism block has two
independent row samples. ID uses eight training provenance families; source
OOD uses Eric and the two Xia-Chan presets; composition OOD uses Jones NNN.
The holdouts concern our mediation-specific fitting, DEV selection and
calibration. The official V3 backbone's upstream pretraining distribution is
not controlled by this experiment.

Within each stratum and sample size, analysis gives equal weight to provenance
families, then equal weight to presets within a family. Mechanism blocks and
row replicates are equally weighted within their allocation; sample sizes are
equally weighted in pooled summaries. The Jones 6/5/5 block allocation rotates
across sample sizes, and analysis weights preserve equal preset weight.
Structural-population clusters are defined by the active SEM, rather than
unused configuration or provenance fields. See the frozen
[bank protocol](bank_protocol.json) and [model-free admission](bank_admission.json).

Mechanism and observation seeds use a purpose-specific namespace with minimum
bit length 319, beyond the registered earlier seed namespaces. Stored common
fold seeds use at most 31 bits for compatibility with 32-bit random-state APIs.
Those fold seeds do not replace or truncate the independent task seeds.

The [bank index](bank_manifest_index.json) retains table hashes, structural
identities, counts, fold seeds and analysis weights. It binds the original
complete bank manifest SHA-256
`532d590e08e55be17a6603cfa3ed11e243476a0bd52a5fe01dfbaaa17038974c`.
It omits observed NPZ tables, synthetic truth labels and the full task-seed
values; it is an archive index rather than an evaluator input. The frozen full
bank remains a separate input in the registered experiment workspace.

## Execution and result gates

Array job `8427083` has 96 fixed shards of 12 tables. Each task evaluates all
three routes sequentially on the same allocated L40S GPU and eight CPUs;
route order rotates by shard. There is no artificial array-concurrency cap.
Scheduler quotas and hardware availability determine the simultaneous tasks.
The [launch plan](launch_plan.json) and [stable job identities](jobs.json) bind
the exact source, bank and execution design.

All routes receive the same observed bytes and exact outer-fold assignment.
Known synthetic effects are used only after inference to calculate metrics.
Initialization and fixed non-test warmup are recorded separately from
device-synchronized per-table inference. Shared CUDA process and file-cache
effects are documented; initialization is not represented as three independent
cold processes.

The real GPU fixture passed actual scheduler accounting and the
[production admission](fixture_admission.json). That fixture does not provide
scientific coverage, length or speed results. Summary job `8427084` must verify
all 96 actual task-accounting records, identities and attempts. Independent
audit job `8427085` must then recompute the summary from saved endpoints and
confirm the actual summary accounting. A released dependency or output file
alone does not satisfy those conditions.

All attempted tables enter coverage, with method failures counted as misses.
Successful-case metrics and paired length differences retain their respective
denominators. The analysis preserves every stratum, preset, sample size,
failure and adverse comparison. The primary level is 95%; 90% and 99% are
prespecified secondary levels. No tuning, replacement, post-training or
calibration is allowed on this test bank.

## Source and reproduction

The 16-file source release is
`0c15d1c24fd52ac98a89cd7336e749432de5b8c682b7f8eee2cf8adcc190c005`;
its byte identities are in [`release_manifest.json`](release_manifest.json).
The launch-bound source-manifest SHA-256 is
`78fac42196727a23f3c6a40cd781319181c09438a66994b3741ead6670418257`.
The source includes bank verification, shard evaluation, summary, independent
audit and statistical contracts. Execution options are documented in
[`EVALUATION.md`](EVALUATION.md).

Evaluation requires the frozen full bank, registered model/runtime release
layout and licensed official V3 weights. The checkpoint index links the same
archived learned weights; the evaluator retains the original producer-source
bindings rather than silently rebinding to newer files. Third-party weights,
raw test tables, scheduler transport scripts and authenticated receipts are
not redistributed here. This source snapshot starts no jobs by itself.
