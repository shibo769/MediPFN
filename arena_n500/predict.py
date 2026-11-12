"""Restore the frozen n=500 Arena estimator and score one observed table."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
sys.path.insert(0, str(RUNTIME))
from common import observed_hash, sha256
from features import BoundV3Point
from fit_arena import load_prediction_head
from tabular_backbones.adapter import adapted_factory, load_adapter
from tabular_backbones.frozen import BackboneConfig, _assignment


LEVELS = ("90", "95", "99")
FIELDS = {"x", "a", "m", "y", "x_types", "mediator_types"}


def read_observed(path):
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != FIELDS:
            raise ValueError("Provide exactly x, a, m, y, x_types and mediator_types")
        observed = {name: source[name] for name in FIELDS}
    if (observed["x"].ndim != 2 or observed["m"].ndim != 2
            or observed["a"].shape != (500,) or observed["y"].shape != (500,)
            or any(len(observed[name]) != 500 for name in ("x", "m")) or not observed["m"].shape[1]):
        raise ValueError("Expected 500 rows, two-dimensional x and m, and one-dimensional a and y")
    for name, values_name in (("x_types", "x"), ("mediator_types", "m")):
        if (observed[name].shape != (observed[values_name].shape[1],)
                or (observed[name].size and observed[name].dtype.kind != "U")):
            raise ValueError(f"Expected one type string per {values_name} column")
    if any(observed[name].dtype.kind not in "biuf" or not np.isfinite(observed[name]).all()
           for name in ("x", "a", "m", "y")) or not np.isin(observed["a"], [0, 1]).all():
        raise ValueError("Observed values must be finite numeric data with treatment coded 0/1")
    return observed


class Predictor:
    """The frozen estimator and error head, with local backbone paths supplied at load."""

    def __init__(self, classifier, regressor, device="cpu", threads=4):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or threads < 1:
            raise ValueError("Use a positive thread count and a CPU or CUDA device")
        torch.set_num_threads(threads)
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Packaged file changed: {name}")
        contract = manifest["point_contract"]
        classifier, regressor = Path(classifier).resolve(), Path(regressor).resolve()
        for path, role in ((classifier, "classifier"), (regressor, "regressor")):
            if sha256(path) != contract[f"{role}_sha256"]:
                raise ValueError(f"Unexpected official TabPFN {role} checkpoint")
        self.config = BackboneConfig(
            **contract["point_config"], classifier_path=str(classifier), regressor_path=str(regressor),
            device=str(self.device),
        )
        config = asdict(self.config)
        for field in ("device", "classifier_path", "regressor_path"):
            config.pop(field)
        if config != contract["point_config"]:
            raise ValueError("Runtime point configuration differs from the frozen contract")
        self.contract = contract
        adapter_path = ROOT / "checkpoints" / "arena_adapter.pt"
        if sha256(adapter_path) != contract["adapter_sha256"]:
            raise ValueError("Wrong meeting adapter")
        adapter, metadata = load_adapter(adapter_path, self.config)
        if metadata["training_metadata"]["step"] != 1750:
            raise ValueError("Wrong selected adapter step")
        self.factory = adapted_factory(self.config, adapter)
        self.head = load_prediction_head(ROOT / "checkpoints" / "arena_sft_head.pt",
                                         ROOT / "checkpoints" / "arena_calibration.json",
                                         contract, device=str(self.device))
        self.threads = threads

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __call__(self, observed, fold_seed=18031):
        if len(observed["a"]) != 500:
            raise ValueError("This checkpoint was trained and evaluated for n=500")
        folds = _assignment(observed["a"], fold_seed)
        self.sync()
        started = time.perf_counter()
        point = BoundV3Point.extract(self, observed, fold_seed, expected_folds=folds)
        endpoints = self.head(np.asarray(point["features"])[None, :], np.array([point["estimate"]]),
                              np.array([point["scale"]]))
        if (endpoints.shape != (1, len(LEVELS), 2)
                or np.any(endpoints[0, :, 0] > point["estimate"])
                or np.any(endpoints[0, :, 1] < point["estimate"])):
            raise ValueError("Invalid confidence interval geometry")
        self.sync()
        fold_sha = hashlib.sha256(np.asarray(folds, np.int8).tobytes()).hexdigest()
        if (point["observed_sha256"] != observed_hash(observed)
                or point["fold_assignment_sha256"] != fold_sha):
            raise ValueError("Input or fold identity differs from prediction")
        return {
            "estimate": point["estimate"],
            "standard_error": point["standard_error"],
            "intervals": {level: endpoints[0, i].tolist() for i, level in enumerate(LEVELS)},
            "observed_sha256": point["observed_sha256"],
            "fold_assignment_sha256": fold_sha,
            "fold_seed": int(fold_seed),
            "n": 500,
            "prediction_seconds": time.perf_counter() - started,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path)
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--regressor", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=18031)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Verify and load adapter, head and calibration")
    args = parser.parse_args()
    if args.verify_only == (args.observed is not None):
        parser.error("Supply either --observed or --verify-only")
    if args.output.exists():
        parser.error("Choose a new output file")
    observed = read_observed(args.observed) if args.observed is not None else None
    started = time.perf_counter()
    predictor = Predictor(args.classifier, args.regressor, args.device, args.threads)
    load_seconds = time.perf_counter() - started
    if observed is None:
        result = {"status": "PASS_CHECKPOINT_BINDING", "point_contract": predictor.contract}
    else:
        result = {"status": "PASS_PREDICTION", **predictor(observed, args.fold_seed)}
    result.update(device=str(predictor.device), threads=predictor.threads, load_seconds=load_seconds)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
