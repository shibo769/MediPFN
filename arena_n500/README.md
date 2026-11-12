# MediPFN V2 for Arena

This checkpoint estimates the population natural indirect effect at treatment 1, `E[Y(1,M(1)) - Y(1,M(0))]`, for one observed table with 500 rows. It includes the selected Arena mediation adapter, the final error-quantile head, and the independently fitted symmetric calibration. TabPFN remains frozen. The 90%, 95% and 99% intervals target the same contrast under the documented mediation identification assumptions. Empirical source-OOD coverage is a separate measured quantity.

## Reproduce a prediction

Use Python 3.10 and install `requirements-research.txt`. Obtain the official classifier and regressor named `tabpfn-v3-classifier-v3_default.ckpt` and `tabpfn-v3-regressor-v3_default.ckpt` from the [Prior Labs model repository](https://huggingface.co/Prior-Labs/tabpfn_3) under its supplied license. The predictor verifies both file hashes. Use a compatible CPU or CUDA PyTorch wheel.

```text
python -B predict.py --observed example/observed_n500.npz --classifier /path/to/tabpfn-v3-classifier-v3_default.ckpt --regressor /path/to/tabpfn-v3-regressor-v3_default.ckpt --device cuda --output prediction.json
```

The input NPZ contains finite numeric `x` and `m` arrays of shape `(500, p)` and `(500, q)`, numeric `a` and `y` vectors of length 500, and the column type vectors `x_types` and `mediator_types`. Treatment must be 0/1 with at least four observations in each arm. A zero-column `x` is allowed. The loader rejects extra arrays. It reads no true effects or evaluator configuration. `--fold-seed` controls the common outer fold split and defaults to 18031. For an evaluator integration, initialize `Predictor` once and use the evaluator's fixed per-table seed.

The supplied fixture is an independent, synthetic observed-only example. `--verify-only` replaces `--observed` to validate the binding of the local checkpoint and calibration to this method. The fixture's prediction timing includes a first model use and is unsuitable as an Arena timing estimate.

## Training exposure and provenance

The adapter used 8,000 training tables. The error head used 10,000 additional tables, and SFT used 2,000 fresh tables plus replay. Independent ID calibration used 1,024 tables outside this fitting budget. All stages used n=500, with source-family holdout enforced for the mediation-specific training, selection and calibration. Upstream TabPFN pretraining exposure is uncontrolled.

The package manifest binds the selected checkpoint bytes, immutable runtime sources and training/evaluation plan hashes. The evaluator is delivered separately to the maintainers. The official TabPFN weights are external assets governed by the included Prior Labs license.
