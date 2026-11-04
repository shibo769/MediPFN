# Optional pretrained tabular backbones

These experiments compare two mediation adaptations while retaining the custom
MediPFN v2 backbone and its checkpoints. They use the same training-source
families, source/composition holdouts and common natural indirect effect target.

| Route | Trained components | Deployment |
|---|---|---|
| Frozen TabPFN-v3 | NIE error-distribution head on observed task summaries and pooled backbone query embeddings | Frozen nuisance backbone and selected head; fresh ID task-mixture calibration |
| TabPFN-v3 with a mediation adapter | Nonlinear column input adapter and prediction-output adjustment trained across independent factual tasks, followed by a separate NIE error head | Frozen selected adapter/backbone/head; no per-test gradient updates |

These are new mediation adaptations rather than original TabCon executions.
The repository also preserves a distinct legacy TabPFN-v2 weight-fine-tuning
implementation; it is not the active V3 adapter route.

## Current execution gates

The active actual-model fixture is job `8426159`. It must verify real V3
classifier/regressor input gradients, parameter updates across tasks, unchanged
frozen backbone weights and checkpoint restoration. Successful completion has
not yet been verified in this snapshot.

Frozen-V3 head job `8426161` and V3 adapter/head job `8426163` are submitted
with the fixture as a dependency. A dependency alone does not establish that
the fixture passed. Production entry points also inspect the required fixture
report and pinned runtime identities. Current execution evidence is maintained
in [STATUS.md](../STATUS.md).

## Package, weights and license

The isolated runtime uses official TabPFN source commit
`953a4b916a19856fbd8dcda089a206b5bfe83696` (package 9.2.0), with official
classifier/regressor checkpoints from Hugging Face repository
`Prior-Labs/tabpfn_3` at revision
`24a16a89d245878b846555110985634aa2e656d7`.
Public metadata and the independently checked LFS identities are recorded in
[`tabular_backbones/v3_runtime/hf_metadata.json`](tabular_backbones/v3_runtime/hf_metadata.json).

Third-party checkpoint binaries are not mirrored here. Their retained
[research license](tabular_backbones/v3_runtime/TABPFN-3-LICENSE) and official
access terms apply separately from the package source license. Runtime config
files contain pinned paths and hashes from the SCC environment; local setup
must supply corresponding licensed files and adjust only their locations.

The standalone [SCC bootstrap utility](tools/bootstrap_v3_scc.py) creates an
isolated environment, obtains the pinned official package and weights, checks
their identities and retains notices. It contains a deliberate cluster-path
guard and is outside the frozen scientific release. Review that guard and
environment paths before adapting it to a different machine. It contains no
authenticated SCC transport or credentials.

## Source and execution

The source digest is
`67409e5e616eea452020131b7beef22de3c6833c1c441e2fa7b30df569d43408`.
The [distribution manifest](manifests/optional_67409e5e616eea45_repository.json)
binds each included source file and records six excluded runtime receipts or
cluster orchestration helpers. The original full manifest is retained for
provenance; it is not a claim that excluded files were distributed.

The scientific implementation and tests live in [`tabular_backbones/`](tabular_backbones/).
After creating a compatible environment and a valid local-weight config, run
the real-model fixture from this `v2` directory:

```sh
python -m tabular_backbones.test_v3_runtime \
  --config tabular_backbones/v3_local_weights_config.json \
  --output runs/optional_v3_fixture
```

Inspect `train_adapter.py` and `train_uncertainty.py` with `--help` before
choosing fresh output directories and explicit training budgets. Head selection
uses independent same-source DEV tasks; calibration tasks are generated only
after the selected head is frozen. Scientific ID/OOD evaluation remains a
separate stage. The immutable source README and protocol describe the release
freeze point; this guide and the root status track subsequent submissions.
