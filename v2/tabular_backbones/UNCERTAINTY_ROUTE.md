# Frozen V3 backbone with learned NIE-error head

`train_uncertainty.py` provides an executable training/calibration entry point for
the mediation-specific frozen-backbone route. It is not original TabCon and does
not claim to reproduce TabCon's specialists, router or reinforcement learning.
It is also a separate initial comparison to the complete custom MediPFN v2 pipeline.

Example, after the real V3 fixture passes and the priority custom stages finish:

```text
env_tabpfn_v3/bin/python -m tabular_backbones.train_uncertainty --config tabular_backbones/v3_runtime/v3_local_weights_config.json --fixture-report <actual-passing-fixture>/fixture_report.json --output <new-empty-run-directory> --cases 1024 --dev-cases 384 --cal-cases 512 --n-values 200,600,1000
```

The program uses a source-balanced bank of fresh synthetic tasks: exactly balanced
training/DEV provenance families, then fresh IID draws from the uniform mixture of
the same eight training families for calibration. Neither source-OOD nor Jones NNN
composition-OOD observations are generated or inspected. Source labels and true
mechanism parameters are recorded for audit and excluded from head features.

The point estimator and all TabPFN weights remain frozen. Each task uses nested
cross-fitting and Bayes-ratio nuisances. Observed-data task features combine pooled
query embeddings with estimated-score summaries; true NIE labels are used only
to form the synthetic estimator-error target. All feature vectors and task seeds
are retained in append-only per-partition JSONL caches. Failed extraction stops the
run and preserves an explicit failure report rather than dropping that case.

The standardized target is `(estimate - truth) / max(estimated_SE, SE_floor)`.
Exactly that positive scale is used when decoding interval endpoints. A monotone
six-quantile head is trained with pinball loss; independent family-balanced DEV
pinball loss selects `best.pt`, while `last.pt` is retained. Feature centering and
scaling are fitted on training features only. No calibration labels are generated
until selection is complete.

Fresh calibration adds an order-statistic correction to the lower/upper error
quantiles. Corrections are nonnegative and made nondecreasing across 90%, 95% and
99% levels so intervals remain nested. This calibration concerns the stated ID
task mixture; no OOD coverage claim follows from it. It is not a mechanism-specific
conditional coverage claim. Scientific ID/OOD test performance remains unmeasured.

The head checkpoint and `calibration.json` bind the exact backbone checkpoint SHA,
configuration, optional trained adapter SHA, feature schema/normalization, real
fixture report and feature-cache hashes. `calibrated_interval` loads a matching
frozen head/calibration pair for deployment with no true target or optimizer.

An optional `--adapter-checkpoint <trained-best.pt>` changes the point estimator
to the separately trained, then frozen, V3 mediation adapter. Its error head and
fresh calibration must be refitted after that adapter freezes. Head/calibration
assets from the plain frozen route cannot be reused for the adapter route.

Default initial budgets are 1,024 training tasks, 384 DEV tasks and 512 calibration
tasks; sample size cycles over draws from 200/600/1,000. The head uses 2,000 small
optimizer updates with batch size 64. Actual GPU runtime must be measured from the
fixture and early feature-extraction logs before extrapolating a completion time.
All output directories are protected against accidental overwrite.
