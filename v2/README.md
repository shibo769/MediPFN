# MediPFN v2: rich mediation simulations

**Primary research prototype: trained, post-trained, freshly calibrated and
restored. Shared scientific ID/OOD evaluation is complete and independently verified.**

The custom backbone is trained from new weights. The bank adapts 15 registry
records into 14 distinct complete-data presets, with the superpopulation
natural indirect effect `E[Y(1,M(1))-Y(1,M(0))]` as the common target. Ten presets
from eight provenance families supply training and independent same-source ID
partitions. Eric and Xia-Chan are source OOD; Jones NNN is composition OOD.
Those holdouts remain outside fitting, DEV selection and calibration. See the
[generator guide](dgp/README.md) for adaptations and truth calculations.

## Primary model inference

From the repository root, verify all bundle weights, calibration and 68 source
files without importing NumPy or PyTorch:

```sh
python v2/predict.py --verify-only
```

The [bundle manifest](checkpoints/primary/bundle.json) records the four selected
checkpoint identities, final calibration and source manifest. It references
the existing factual/nested checkpoints rather than duplicating them. The
selected factual/nested networks retain separate weights. A single ordered
signed-error head and confidence-guided post-training supply the interval
quantiles; this is distinct from the v1 mixture-of-experts construction.

With the recorded NumPy/PyTorch/SciPy environment, provide a complete observed
table as an NPZ:

```sh
python -m pip install -r v2/requirements-primary.txt
```

The packaged CPU restoration used PyTorch 2.5.1+cu121, NumPy 2.2.6 and SciPy
1.15.3. Select the appropriate official PyTorch build for the local hardware.

Create the NPZ from observed arrays:

```python
import numpy as np
np.savez("observed_table.npz", x=x, a=a, m=m, y=y,
         x_types=np.asarray(x_types, dtype="U"),
         mediator_types=np.asarray(mediator_types, dtype="U"))
```

| Key | Shape and interpretation |
|---|---|
| `x` | Numeric baseline covariates, `[n,p]`, with 0–32 columns; use shape `[n,0]` when no baseline covariates are observed |
| `a` | Treatment, `[n]`, with values 0/1 for the trained contrast |
| `m` | Numeric mediator coordinates, `[n,q]`, with 1–128 columns |
| `y` | Numeric outcome, `[n]` |
| `x_types`, `mediator_types` | Optional one-dimensional Unicode/string arrays matching their columns; entries `continuous`, `binary` or `categorical`; omitted columns default to `continuous` |

Values must be finite and complete, with at least eight observations in each
treatment group for nested cross-fitting. Type tags describe the model input
roles; they do not select a new model or create a new causal identification
claim. Extra NPZ fields, including synthetic truth, are ignored and never
materialized by this entry point. Pickled NPZ objects are prohibited.

```sh
python v2/predict.py --bundle v2/checkpoints/primary/bundle.json \
  --data observed_table.npz --output interval.json --device cpu
```

The JSON output contains the point estimate, estimated scale and 90%/95%/99%
calibrated intervals. `--seed` controls nested cross-fitting. Existing output
files are not overwritten. The CLI rejects altered source/weight/calibration
bytes and incompatible construction contracts before prediction.

Fresh calibration uses the registered iid ID task mixture. It supplies no
per-source, conditional or arbitrary OOD coverage guarantee. The
[shared results](../evaluation/v2_shared_20261103/results/README.md) retain
strong ID efficiency together with composition-OOD undercoverage. A loading
fixture is separate from that scientific coverage and speed benchmark. Causal interpretation still requires the
natural-effect identification and overlap assumptions.

## Completed training stages

Factual nuisance job `8425910` completed 20,000 steps and 160,000 table draws;
step 17,000 was selected by same-source DEV loss (1.526461). Nested job
`8425970` completed 5,000 steps and 40,000 draws, selecting step 5,000 with
DEV pseudo-outcome loss 0.229999. Its frozen factual-teacher hash matches the
selected factual checkpoint.

Signed-error training uses 10,000 new tasks and 1,024 DEV tasks. Fresh
post-training uses 1,024 training tasks and 512 DEV tasks, followed by 1,024
independent iid calibration tasks. Job `8426136` completed training and
post-training but exited 1 on a calibration-output filename collision.
Finalizer `8426219` successfully calibrated the same selected weights using
the same fresh cache, without retraining or selection, and passed a bound
200-row CPU restoration fixture. The original exit and finalizer accounting
are retained in the [verification record](checkpoints/primary/verification.json).

Selected checkpoint, configuration and training-history files are under
[`checkpoints/`](checkpoints/). A complete bundle is in the
[private primary release](https://github.com/shibo769/MediPFN/releases/tag/v2-primary-20261103).
The larger nuisance optimizer-resume states are separate private releases.

## Frozen sources and training entry points

The current primary source release is
`c1bd3404749513a923eb3da4f0586e3fb9544effd499dc0b7fb03faa8420a923`.
The [source manifest](release_manifest.json) retains the original 59 factual/
nested file hashes and adds nine error-learning/binding/calibration files.
Only the workflow calibration-output filename differs from the release that
produced the error/post-training weights. All source identities remain in the
[manifest index](manifests/README.md).

Training entry points are `train.py`, `train_nested.py` and
`run_error_pipeline.py`; inspect their `--help` before allocating resources.
For example, from this `v2` directory with trained nuisance weights:

```sh
python run_error_pipeline.py --checkpoint checkpoints/factual/best.pt \
  --eta-checkpoint checkpoints/nested/best.pt \
  --output runs/error_seed1 --cpu-threads 8
```

This is a full CUDA training workflow with a new output directory, not a smoke
check. The immutable [error guide](ERROR_PIPELINE.md) and
[protocol](protocol.json) describe their freeze points;
[STATUS.md](../STATUS.md) records subsequent execution. Optional pretrained
backbone experiments have a separate [guide](OPTIONAL_BACKBONES.md) and source
manifest. Their actual-model fixture passed; the frozen-V3 error head and
fresh calibration are complete and archived. The V3 adapter, separate NIE head
and calibration are also complete, independently restored and archived.
Shared ID/OOD evaluation is complete, with all attempts and independently
recomputed results [archived](../evaluation/v2_shared_20261103/results/README.md).

The retained v1 assets are separate in [`../v1`](../v1). Vendored source
materials retain their [license](dgp/vendor/LICENSE) and [notice](dgp/vendor/NOTICE.md).
Third-party TabPFN weights, raw test tables and authentication material are
not redistributed.
