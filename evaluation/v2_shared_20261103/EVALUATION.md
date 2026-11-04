# Frozen three-route evaluation

The evaluator compares the already selected custom MediPFN v2, frozen TabPFN-v3
with its NIE error head, and frozen TabPFN-v3 with its trained mediation adapter
and matching NIE error head. Every route receives the same observed NPZ arrays
and outer-fold assignment. Known synthetic effects enter offline metrics only.
No model, interval, calibration, or hyperparameter is updated during evaluation.

The four additional ID-only fixture tables check real GPU execution, no-X and
100-mediator inputs, n=200/1000, matching native fold assignments, checkpoint
bindings, and parity with the unchanged V3 interval helper at its default seed.
They are separate from the 1,152-table scientific bank and are not performance
results. Scientific submission should be released only after this fixture's
actual accounting and PASS artifact have been verified.

## Execution

Use the isolated real V3 runtime for all routes. Each array task evaluates 12
tables and all three routes on the same L40S GPU and eight allocated CPUs.
Routes are loaded sequentially and discarded between routes. Their order
rotates across shards. There are 96 prespecified shards; scheduler quotas and
available hardware determine concurrency.

`fixture.py` accepts `--bank`, `--bank-manifest-sha`, `--source-manifest`,
`--source-manifest-sha`, `--models-root`, `--output`, `--device cuda`,
`--cpu-threads 8`, and `--require-gpu L40S`.

`evaluate_shard.py` uses the same options, plus `--shard-index` (one-based) and
`--shard-size 12`. Its output should be `shards/shard_<index>.json`.

`summarize.py` accepts `--bank`, `--bank-manifest-sha`, `--shards`,
`--source-manifest-sha`, `--expected-job-id`, `--shard-size 12`, and a new
`--output` directory. It checks actual accounting for every array task, all
model/source/table/fold identities, all attempts, failures, and interval
arithmetic before producing `metrics.csv`, `paired.csv`, `startup.csv`,
`failures.json`, and `audit.json`.

`audit_summary.py` independently recomputes displayed coverage, lengths, point
errors, paired differences, weights, and denominators from raw endpoints. It
also requires the summary job's actual successful accounting. Its options are
`--summary`, `--shards`, `--bank`, `--bank-manifest-sha`, `--summary-job-id`,
and a new `--output` JSON file. A scheduler dependency ending is not evidence
that a predecessor succeeded.

## Timing and uncertainty

Device synchronization brackets each observed-table inference, including
preprocessing/context fits, nested nuisance predictions, score construction,
error-head computation, and fixed calibration. Bank I/O, known truth/metrics,
model loading, and a fixed non-test bootstrap are excluded. Shared process startup
and each route's loading/bootstrap times are retained separately. Later routes
share an existing CUDA process and operating-system file cache; these are not
three independent cold-process measurements. Raw total,
mean, weighted median, and weighted 90th-percentile times are reported.

Bank weights give equal source-family and preset weight within each stratum
and sample size, then equal weight across sample sizes. Identical active SEM
configurations share a structural-population cluster even when seeded blocks
differ. Allocation-stratum finite-sample-corrected cluster sandwich standard errors and conditional table-draw Monte
Carlo standard errors answer different questions and are both retained. A
cluster standard error is left missing when an allocation stratum has fewer
than two distinct populations.

All attempted tables enter coverage; a method failure counts as a miss.
Lengths, interval scores, bias, and RMSE use successful cases and report that denominator.
Paired length differences use jointly successful cases. The 95% level is
primary; 90% and 99% are prespecified secondary levels. Every source, stratum,
sample size, and adverse result remains in the outputs.

Source-OOD refers to exclusion from our mediation-specific fitting. The
inherited upstream V3 pretraining distribution is not controlled here. These
are fixed-checkpoint comparisons; their original training budgets and
upstream initialization differ.
