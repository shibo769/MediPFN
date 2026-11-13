# MediPFN

**Amortized In-Context Nuisance Learning and Inference for Causal Mediation Analysis**

Shi Bo and Jiaming Hu · Boston University

[Paper](https://arxiv.org/abs/2611.13219) ·
[Hugging Face checkpoints](https://huggingface.co/shibo769/MediPFN) ·
[Meeting-run model](arena_n500/MODEL_CARD.md) ·
[Inference guide](arena_n500/README.md) ·
[Evaluation results](ARENA_RESULTS.md) ·
[Model release](https://github.com/shibo769/MediPFN/releases/tag/arena-n500-20261109)

MediPFN estimates a natural indirect effect and its uncertainty from an
observed table. It combines nuisance predictions with a cross-fitted mediation
score and uses a learned error distribution to construct confidence intervals.

The current **Arena meeting-run checkpoint** uses a frozen TabPFN-v3 backbone,
a mediation adapter, an SFT error head and independent calibration. It accepts
500-row tables with binary treatment and a joint mediator vector. The
published manuscript and this later meeting-run checkpoint use distinct
training and evaluation protocols.

## Quick Start

Use Python 3.10. The first run downloads about 446 MB of official TabPFN-v3
weights from Prior Labs. Both Hugging Face repositories support anonymous
downloads.

```sh
python -m pip install "huggingface_hub==2.4.0"
hf download shibo769/MediPFN arena_n500/requirements-research.txt --revision 315046cf9838ebcf2b920dad0f87e95bbb5bbc04 --local-dir medipfn_hf
python -m pip install -r medipfn_hf/arena_n500/requirements-research.txt
```

Save the following as `quickstart.py` and run `python quickstart.py`. The example
downloads a fixed MediPFN release and the two matching TabPFN-v3 checkpoints,
then scores the included synthetic table on CPU.

```python
import json
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download, snapshot_download

model_dir = Path(snapshot_download(
    repo_id="shibo769/MediPFN",
    revision="315046cf9838ebcf2b920dad0f87e95bbb5bbc04",
    allow_patterns=["arena_n500/**"],
    local_dir="medipfn_hf",
    token=False,
)) / "arena_n500"

backbone_repo = "Prior-Labs/tabpfn_3"
backbone_revision = "e923ba9be85784206c9e2f43b0035c84d5fd5747"
classifier, regressor = [
    hf_hub_download(
        repo_id=backbone_repo,
        filename=f"tabpfn-v3-{role}-v3_default.ckpt",
        revision=backbone_revision,
        token=False,
    )
    for role in ("classifier", "regressor")
]

sys.path.insert(0, str(model_dir.resolve()))
from predict import Predictor, read_observed

predictor = Predictor(classifier, regressor, device="cpu", threads=4)
observed = read_observed(model_dir / "example" / "observed_n500.npz")
result = predictor(observed, fold_seed=18031)
print(json.dumps({
    "estimate": result["estimate"],
    "standard_error": result["standard_error"],
    "intervals": result["intervals"],
}, indent=2))
```

With the example seed, the NIE is approximately `0.24574`, the score standard
error is `0.06337`, and the 95% confidence interval is `[0.06425, 0.42723]`.

For **Arena**, initialize `Predictor` once and reuse it across tables. Pass the
evaluator-assigned `fold_seed` on each call. Supply an observed 500-row table
with `x`, `a`, `m`, `y`, `x_types`, and `mediator_types`.

See the [inference guide](arena_n500/README.md) for the full input schema,
command-line usage and checkpoint verification.

## Meeting-run evaluation

The fitting budget comprises 20,000 distinct simulated tables, with separate
development and calibration data. A held-out evaluation used 1,000 ID and
1,000 source-OOD tables, each with 500 rows. There were no recorded prediction
failures.

| Setting | 95% coverage | Mean interval length | Tables |
|---|---:|---:|---:|
| ID | 95.39% | 0.5519 | 1,000 |
| Source OOD | 97.90% | 2.0203 | 1,000 |

These results describe two specified simulation mixtures. The source-OOD
families were excluded from mediation-specific fitting, development and
calibration; upstream TabPFN pretraining exposure is unknown. Mean lengths
across settings can reflect different effect and outcome scales. Same-table
baselines for the meeting run have not yet been evaluated. See the
[evaluation report](ARENA_RESULTS.md) for per-configuration results, weighting,
timing and reproducibility details.

## Repository contents

- [`arena_n500/`](arena_n500/) contains the meeting-run inference code, model
  card, adapter and uncertainty checkpoints, calibration, and a synthetic
  observed-only example.
- [`ARENA_RESULTS.md`](ARENA_RESULTS.md) records the evaluation protocol and
  results. The raw held-out tables and maintainer evaluator are separate.

The meeting-run checkpoint has been restored and tested on a synthetic table.
The public meeting-run package provides inference and the fitted checkpoints;
the complete meeting-run data-generation and training workflow is not bundled.
Across Windows and SCC CPU environments, the same input produced a small
numerical difference; see [ARENA_RESULTS.md](ARENA_RESULTS.md) for details.
The causal interpretation of the NIE requires the identification assumptions
described in the [model card](arena_n500/MODEL_CARD.md).

## Citation

```bibtex
@article{bo2026medipfn,
  title={MediPFN: Amortized In-Context Nuisance Learning and Inference for Causal Mediation Analysis},
  author={Bo, Shi and Hu, Jiaming},
  journal={arXiv preprint arXiv:2611.13219},
  year={2026}
}
```

## License and provenance

The repository does not currently provide a blanket redistribution license.
The meeting-run package includes the applicable TabPFN-3 license and notice.
Official pretrained TabPFN weights are obtained separately and are subject to
their own terms.
