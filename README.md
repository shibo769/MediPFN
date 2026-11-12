# MediPFN

**Amortized In-Context Nuisance Learning and Inference for Causal Mediation Analysis**

Shi Bo and Jiaming Hu · Boston University

[Paper](https://arxiv.org/abs/2611.13219) ·
[Meeting-run model](arena_n500/MODEL_CARD.md) ·
[Inference guide](arena_n500/README.md) ·
[Evaluation results](ARENA_RESULTS.md) ·
[Checkpoint index](CHECKPOINTS.md)

MediPFN estimates a natural indirect effect and its uncertainty from an
observed table. It combines nuisance predictions with a cross-fitted mediation
score and uses a learned error distribution to construct confidence intervals.

The current **Arena meeting-run checkpoint** uses a frozen TabPFN-v3 backbone,
a mediation adapter, an SFT error head and independent calibration. It accepts
500-row tables with binary treatment and a joint mediator vector. The
published manuscript and this later meeting-run checkpoint use distinct
training and evaluation protocols.

## Inference

Clone the repository and obtain the official TabPFN-v3 classifier and regressor
weights separately from [Prior Labs](https://huggingface.co/Prior-Labs/tabpfn_3)
under their license. With a compatible Python 3.10 environment:

```sh
git clone https://github.com/shibo769/MediPFN.git
cd MediPFN/arena_n500
pip install -r requirements-research.txt
python predict.py --observed example/observed_n500.npz \
  --classifier /path/to/tabpfn-v3-classifier-v3_default.ckpt \
  --regressor /path/to/tabpfn-v3-regressor-v3_default.ckpt \
  --device cpu --output prediction.json
```

The input contains observed `x`, `a`, `m`, `y` arrays and variable-type
metadata. The output contains a point estimate, standard error and 90%, 95%
and 99% confidence intervals. See the [inference guide](arena_n500/README.md)
for the input schema, exact checkpoint names, dependency versions and hash
checks. The adapter, error head and calibration are included in the repository.

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
- [`v1/`](v1/) and [`v2/`](v2/) preserve earlier custom-backbone research
  pipelines and checkpoints. Their model cards and historical evaluation are
  indexed in [CHECKPOINTS.md](CHECKPOINTS.md).
- [`evaluation/`](evaluation/) contains earlier aggregate evaluation results.
  Raw held-out tables and the meeting-run maintainer evaluator are separate.

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
Vendored upstream generator materials retain their [Apache-2.0 license](v2/dgp/vendor/LICENSE)
and [notice](v2/dgp/vendor/NOTICE.md). The meeting-run package includes the
applicable TabPFN-3 license and notice. Official pretrained TabPFN weights are
obtained separately and are subject to their own terms.
