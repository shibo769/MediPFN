# Reproducibility

## Restore the retained model

```sh
git clone https://github.com/shibo769/MediPFN.git
cd MediPFN
python v1/restore.py --verify-only
```

The hash check requires only the Python standard library. It checks all 97
archived production source/asset files against `v1/manifest.json`, including
all seven `.pt` files. These files were independently downloaded from GitHub
after the initial push and checked against the original SCC SHA-256 values.

For inference, use Python 3.12 or later with NumPy, PyTorch, SciPy and
scikit-learn. The original verified restoration environment used PyTorch
2.5.1+cu121 and NumPy 2.2.6; restoration itself ran on the CPU.

```sh
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install numpy==2.2.6 torch==2.5.1 scipy scikit-learn
python v1/restore.py --smoke
```

The CUDA-specific PyTorch wheel is unnecessary for CPU restoration. The smoke
test generates a new synthetic input without opening the study's test bank.
It checks loading and finite output; it does not reproduce a benchmark table.

## Python inference

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("medipfn_v1", Path("v1/restore.py"))
medipfn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(medipfn)

# Supply an observed table and its fixed two-fold assignment.
# rows follow the original v1 [X..., A, M, Y] convention.
result = medipfn.predict(rows, fold_ids, level=.95, threads=4)
print(result)
```

The wrapper verifies all archived assets before loading. It uses the retained
SFT mixture router, selected 95% head, and frozen calibration coefficients.
The 90% and 99% calls use their own frozen heads. Historical absolute paths in
the source and calibration records are resolved to the local archive without
rewriting their original bytes.

## Training source map

| Stage | Archived implementation |
|---|---|
| Mechanism/table generation | `v1/project/code/data.py`, `generate_data.py` |
| Nuisance pretraining | `v1/project/code/train_backbone.py`, `model.py` |
| Cross-fitted score/features | `v1/project/code/inference.py`, `build_cache.py` |
| Error-mixture fitting | `v1/project/posttrain_v10_20261026/tabcon/mixture.py` |
| Confidence fitting | `v1/project/capacity_4m_v10_20261028/confidence_calibration_4m_20261028/calibration.py` |
| 95% post-training | `v1/project/capacity_4m_v10_20261028/robust_retrain_train_seedfix_v2.py` |
| 90% / 99% heads | archived `alpha_heads.py` and retained head checkpoints |
| Completed CPU evaluation | archived `cpu_evaluation_stage_v3.py` and bound inference modules |

The original training scripts preserve the original cluster paths and protocol
guards. Training from scratch requires constructing new training/development
and calibration inputs, choosing a fresh output directory, and explicitly
adapting those paths. This backup is sufficient to restore the retained
inference pipeline; it does not package raw training/test tables or claim a
single-command regeneration of every historical experiment.

For example, the nuisance training script exposes its configuration through:

```sh
python v1/project/code/train_backbone.py --help
```

Its defaults are defaults in preserved source, not a statement of the exact
arguments used for the retained checkpoint. Do not overwrite the retained
weights when running a new experiment. A fresh v2 training run must use its
own configuration, source hashes, seeds, split manifest and checkpoint folder.

## Evaluation and provenance

`v1/results/metrics.csv` contains the complete shared-bank aggregate metric
rows. `paired.csv` contains matched comparisons with mechanism-cluster
uncertainty. Their file hashes are stored in `v1/results/sha256.json`.

`v1/production_bindings.json` preserves the evaluation source/asset bindings
without redistributing observed test tables. Some preserved audit JSONs
reference original cluster files that are not part of this inference archive;
their presence records provenance rather than promising those original paths
exist after cloning.

`v1/restore_smoke_report.json` records the actual restoration check. Its timing
is a small installation test and must not be used as a benchmark speed result.
