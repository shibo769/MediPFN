"""Fit the meeting n=500 error head, SFT and independent symmetric calibration.

Run with the frozen V2 model_pipeline source on PYTHONPATH. The only inputs
are the five admitted fitting, development and calibration feature caches.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from common import TRAIN_SOURCES, assert_contract, check_pairs, digest, sha256, write_json_new
from heads import PosttrainedV3Error, load_model, pinball, save_model


COUNTS = {"head_train": 10000, "head_dev": 1024, "sft_train": 2000,
          "sft_dev": 512, "calibration": 1024}
LEVELS = (.90, .95, .99)
DIMENSION = 1038
FIT_FIELDS = {"case_id", "source_group", "source_id", "mechanism_id", "features",
              "estimate", "scale", "truth", "weight"}
CORE_PARAMETERS = 335394
SELECTED_ADAPTER_SHA = "87c13c5e32a3c1b441c06f22876343a3a75d3684b65d870e58263fe4431eef37"
SFT_STAGE = "arena_n500_mixed_replay_quantile_sft"
CAL_STATUS = "PASS_ARENA_N500_INDEPENDENT_SYMMETRIC_CALIBRATION"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def weights_for(rows):
    """Equal family, then preset, then mechanism, and then table weights."""
    groups = rows["source_group"].tolist()
    sources = rows["source_id"].tolist()
    mechanisms = rows["mechanism_id"].tolist()
    presets, blocks = defaultdict(set), defaultdict(set)
    triples = list(zip(groups, sources, mechanisms))
    counts = Counter(triples)
    for group, source, mechanism in triples:
        presets[group].add(source)
        blocks[group, source].add(mechanism)
    return np.asarray([
        1 / (len(presets) * len(presets[g]) * len(blocks[g, p]) * counts[g, p, m])
        for g, p, m in triples
    ])


def read_partition(path, phase, expected_sha, counts):
    if sha256(path) != expected_sha:
        raise ValueError(f"{phase} cache SHA mismatch")
    with np.load(path, allow_pickle=False) as src:
        if not FIT_FIELDS <= set(src.files):
            raise ValueError(f"{phase} cache missing required arrays")
        rows = {name: src[name].copy() for name in FIT_FIELDS}
        for key in ("configuration_sha256", "observed_sha256", "observation_seed_sha256"):
            if key in src.files:
                rows[key] = src[key].copy()
        if "n" in src.files and not np.all(src["n"] == 500):
            raise ValueError(f"{phase} contains a table size other than 500")
    n = counts[phase]
    if rows["features"].shape != (n, DIMENSION) or any(
        rows[key].shape != (n,) for key in FIT_FIELDS - {"features"}
    ):
        raise ValueError(f"{phase} shape or denominator mismatch")
    for key in ("features", "estimate", "scale", "truth", "weight"):
        if not np.isfinite(rows[key]).all():
            raise ValueError(f"{phase} has nonfinite {key}")
    for key in rows.keys() - {"features", "estimate", "scale", "truth", "weight"}:
        if rows[key].dtype.kind != "U" or rows[key].shape != (n,) or not all(rows[key]):
            raise ValueError(f"{phase} has invalid {key}")
    if len(set(rows["case_id"])) != n or len(set(rows["mechanism_id"])) != n:
        raise ValueError(f"{phase} has repeated cases or mechanisms")
    if any(TRAIN_SOURCES.get(source) != group
           for source, group in zip(rows["source_id"], rows["source_group"])):
        raise ValueError(f"{phase} contains a held-out or unknown source")
    if set(rows["source_id"]) != set(TRAIN_SOURCES):
        raise ValueError(f"{phase} does not include all ten training presets")
    if np.any(rows["scale"] <= 0):
        raise ValueError(f"{phase} has nonpositive standardization scale")
    if np.any(rows["weight"] <= 0) or not np.isclose(rows["weight"].sum(), 1):
        raise ValueError(f"{phase} has invalid weights")
    expected_weight = 1 / n if phase == "calibration" else weights_for(rows)
    if not np.allclose(rows["weight"], expected_weight, rtol=2e-6, atol=0):
        raise ValueError(f"{phase} weights do not match its sampling protocol")
    return rows


def verify_disjoint(partitions, floor):
    for key in ("case_id", "mechanism_id", "configuration_sha256",
                "observed_sha256", "observation_seed_sha256"):
        seen = set()
        for phase, rows in partitions.items():
            if key not in rows:
                continue
            current = set(rows[key])
            if len(current) != len(rows[key]) or seen & current:
                raise ValueError(f"{phase} overlaps a case/partition by {key}")
            seen |= current
    if not math.isfinite(floor) or floor <= 0 or any(
        np.any(rows["scale"] < floor) for rows in partitions.values()
    ):
        raise ValueError("Feature-cache scales violate the registered floor")


def open_fitting_caches(directory, counts):
    path = directory / "cache_manifest.json"
    manifest = read_json(path)
    contract = manifest["point_contract"]
    if contract.get("adapter_sha256") != SELECTED_ADAPTER_SHA:
        raise ValueError("Feature caches do not use the selected meeting adapter")
    if contract.get("feature_dimension") != DIMENSION:
        raise ValueError("Feature dimension differs from the frozen V2 geometry")
    if not check_sha(manifest["train_plan_sha256"]):
        raise ValueError("Missing frozen fitting-plan SHA")
    if set(manifest["phase_sha256"]) != set(counts):
        raise ValueError("Expected the five meeting feature partitions only")
    if manifest.get("n", 500) != 500:
        raise ValueError("Expected 500 rows per meeting table")
    partitions = {
        phase: read_partition(directory / f"{phase}.npz", phase, manifest["phase_sha256"][phase], counts)
        for phase in counts if phase != "calibration"
    }
    verify_disjoint(partitions, float(contract["standard_error_floor"]))
    return manifest, partitions, sha256(path)


def tensors(rows, center, scale, device):
    features = (rows["features"] - center) / scale
    response = (rows["estimate"] - rows["truth"]) / rows["scale"]
    x, y, w = (
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in (features, response, rows["weight"])
    )
    if not all(bool(torch.isfinite(value).all()) for value in (x, y, w)):
        raise ValueError("Normalized features, responses or weights are nonfinite")
    return x, y, w


def state_copy(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def risk(model, data):
    x, y, w = data
    model.eval()
    with torch.no_grad():
        result = float((pinball(model(x), y) * w).sum())
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite DEV pinball risk")
    return result


def optimize_step(model, optimizer, loss):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Nonfinite training gradients")
    optimizer.step()
    return float(norm)


def fit_base(train, dev, path, args):
    model = PosttrainedV3Error("head_only").to(args.device)
    if sum(parameter.numel() for parameter in model.parameters()) != CORE_PARAMETERS:
        raise ValueError("Registered V2 error-head architecture changed")
    initial = state_copy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    x, y, w = train
    best_state = None
    best_risk = math.inf
    best_step = None
    with path.open("x", encoding="utf-8") as log:
        for step in range(1, args.head_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            idx = torch.randint(len(x), (args.batch_size,), device=args.device)
            # Uniform-case draws with an unbiased importance factor for the exact balanced risk.
            loss = (pinball(model(x[idx]), y[idx]) * w[idx] * len(x)).mean()
            norm = optimize_step(model, optimizer, loss)
            if step == 1 or step % 50 == 0 or step == args.head_steps:
                current = risk(model, dev)
                row = dict(step=step, train_pinball=float(loss.detach()), dev_pinball=current,
                           gradient_norm=norm)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(dict(phase="head", **row)), flush=True)
                if current < best_risk:
                    best_state, best_risk, best_step = state_copy(model), current, step
    if best_state is None or all(torch.equal(initial[k], best_state[k]) for k in initial):
        raise RuntimeError("Base head did not train")
    model.load_state_dict(best_state)
    return model, dict(selected_step=best_step, selected_dev_pinball=best_risk)


def fit_sft(model, head_train, head_dev, train, dev, path, args):
    """50/50 balanced replay and fresh synthetic meeting tasks; DEV-only selection."""
    sx, sy, sw = head_train
    fx, fy, fw = train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    baseline = (risk(model, head_dev), risk(model, dev))
    best = sum(baseline) / 2
    best_step, best_state = 0, state_copy(model)
    with path.open("x", encoding="utf-8") as log:
        log.write(json.dumps(dict(step=0, replay_dev_pinball=baseline[0],
                                  fresh_dev_pinball=baseline[1], joint_dev_pinball=best,
                                  selected=True)) + "\n")
        log.flush()
        for step in range(1, args.sft_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            si = torch.multinomial(sw, args.batch_size, replacement=True)
            fi = torch.multinomial(fw, args.batch_size, replacement=True)
            loss = .5 * (pinball(model(sx[si]), sy[si]).mean() +
                         pinball(model(fx[fi]), fy[fi]).mean())
            norm = optimize_step(model, optimizer, loss)
            if step % 25 == 0 or step == args.sft_steps:
                r, f = risk(model, head_dev), risk(model, dev)
                current = .5 * (r + f)
                selected = current < best
                row = dict(step=step, train_pinball=float(loss.detach()), replay_dev_pinball=r,
                           fresh_dev_pinball=f, joint_dev_pinball=current, selected=selected,
                           gradient_norm=norm)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(dict(phase="sft", **row)), flush=True)
                if selected:
                    best_state, best, best_step = state_copy(model), current, step
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return dict(selected_step=best_step, selected_joint_dev_pinball=best,
                baseline_joint_dev_pinball=sum(baseline) / 2,
                baseline_replay_dev_pinball=baseline[0], baseline_fresh_dev_pinball=baseline[1])


def calibrate(model, data):
    x, response, _ = data
    model.eval()
    with torch.no_grad():
        pairs = model(x).cpu().numpy()
    check_pairs(pairs)
    # Same score-SE units as V2. The symmetric radius is scale*max(abs(q_lo), abs(q_hi)).
    scores = np.abs(response.cpu().numpy())[:, None] - np.abs(pairs).max(-1)
    ranks = [math.ceil((len(x) + 1) * level) for level in LEVELS]
    if max(ranks) > len(x):
        raise ValueError("Calibration count too small for the requested nominal levels")
    raw = [max(0., float(np.partition(scores[:, j], rank - 1)[rank - 1]))
           for j, rank in enumerate(ranks)]
    return dict(ranks=ranks, raw_expansions=raw, expansions=np.maximum.accumulate(raw).tolist())


def load_prediction_head(sft_checkpoint, calibration, point_contract, device="cpu"):
    """Load only matched post-training assets and return an observed-feature predictor."""
    calibration = read_json(calibration)
    if (calibration.get("status") != CAL_STATUS or calibration.get("geometry") != "sym"
            or calibration.get("levels") != list(LEVELS)
            or calibration.get("checkpoint_sha256") != sha256(sft_checkpoint)):
        raise ValueError("Selected calibration does not bind this SFT checkpoint")
    model, doc = load_model(sft_checkpoint, device)
    metadata = doc["metadata"]
    if metadata.get("stage") != SFT_STAGE or model.variant != "head_only":
        raise ValueError("Checkpoint is not the meeting SFT head")
    assert_contract(metadata["contract"], point_contract)
    if calibration.get("point_contract_sha256") != digest(point_contract):
        raise ValueError("Calibration point contract mismatch")
    if calibration.get("train_plan_sha256") != metadata["train_plan_sha256"]:
        raise ValueError("Calibration fitting plan mismatch")
    schema = metadata["feature_schema"]
    if digest(schema) != metadata["feature_schema_sha256"]:
        raise ValueError("Feature schema hash mismatch")
    center, denom = np.asarray(schema["center"]), np.asarray(schema["scale"])
    if (center.shape != (DIMENSION,) or denom.shape != (DIMENSION,)
            or not np.isfinite(center).all() or not np.isfinite(denom).all()
            or np.any(denom <= 0)):
        raise ValueError("Invalid training-only feature schema")
    expansions = np.asarray(calibration["expansions"], dtype=float)
    if (expansions.shape != (3,) or not np.isfinite(expansions).all()
            or np.any(expansions < 0) or np.any(np.diff(expansions) < 0)):
        raise ValueError("Invalid symmetric calibration expansion")
    model.eval().requires_grad_(False)

    def predict(features, estimate, scale):
        features = np.atleast_2d(np.asarray(features))
        estimate, scale = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (estimate, scale))
        n = len(estimate)
        if (features.shape != (n, DIMENSION) or estimate.shape != (n,) or scale.shape != (n,)
                or not all(np.isfinite(v).all() for v in (features, estimate, scale))
                or np.any(scale <= 0)):
            raise ValueError("Invalid observed-only prediction features or scale")
        with torch.no_grad():
            x = torch.as_tensor((features - center) / denom, dtype=torch.float32, device=device)
            pairs = model(x).cpu().numpy()
        check_pairs(pairs)
        radius = scale[:, None] * (np.abs(pairs).max(-1) + expansions)
        ends = np.stack((estimate[:, None] - radius, estimate[:, None] + radius), axis=-1)
        check_pairs(ends)
        return ends

    return predict


def run(args):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    manifest, partitions, manifest_sha = open_fitting_caches(args.cache_dir, COUNTS)
    train = partitions["head_train"]["features"]
    center = train.mean(0, dtype=np.float64)
    denom = np.maximum(train.std(0, dtype=np.float64), 1e-4)
    schema = dict(input_dimension=DIMENSION, center=center.tolist(), scale=denom.tolist(),
                  floor=1e-4, fit_partition="head_train only",
                  fit_cache_sha256=manifest["phase_sha256"]["head_train"], fit_cases=len(train))
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    head_dir, sft_dir = args.output / "head", args.output / "sft"
    head_dir.mkdir()
    sft_dir.mkdir()
    data = {phase: tensors(rows, center, denom, args.device)
            for phase, rows in partitions.items()}
    model, base_report = fit_base(data["head_train"], data["head_dev"],
                                  head_dir / "training.jsonl", args)
    source_sha = {"fit_arena.py": sha256(__file__),
                  "heads.py": sha256(load_model.__code__.co_filename),
                  "common.py": sha256(assert_contract.__code__.co_filename)}
    provenance = dict(
        route=manifest["point_contract"]["route"], contract=manifest["point_contract"],
        feature_schema=schema, feature_schema_sha256=digest(schema),
        source_sha256=source_sha, cache_manifest_sha256=manifest_sha,
        train_plan_sha256=manifest["train_plan_sha256"],
        train_cache_sha256=manifest["phase_sha256"], n=500,
        counts=COUNTS, no_test_used=True, parameter_count=CORE_PARAMETERS,
    )
    base_checkpoint, sft_checkpoint = head_dir / "best.pt", sft_dir / "best.pt"
    save_model(base_checkpoint, model, dict(provenance, stage="arena_n500_base_head", **base_report))
    torch.manual_seed(args.seed + 1)
    sft_report = fit_sft(model, data["head_train"], data["head_dev"], data["sft_train"],
                         data["sft_dev"], sft_dir / "training.jsonl", args)
    # Final checkpoint is sealed before calibration labels are passed to the prediction model.
    save_model(sft_checkpoint, model, dict(provenance, stage=SFT_STAGE,
                                           base_checkpoint_sha256=sha256(base_checkpoint),
                                           selection="minimum equal-mixture replay/fresh DEV pinball; step 0 eligible",
                                           **sft_report))
    cal_rows = read_partition(args.cache_dir / "calibration.npz", "calibration",
                              manifest["phase_sha256"]["calibration"], COUNTS)
    verify_disjoint(dict(partitions, calibration=cal_rows),
                    float(manifest["point_contract"]["standard_error_floor"]))
    calibration_result = calibrate(model, tensors(cal_rows, center, denom, args.device))
    calibration_path = args.output / "calibration.json"
    write_json_new(calibration_path, dict(
        status=CAL_STATUS, geometry="sym", levels=list(LEVELS), cases=COUNTS["calibration"],
        checkpoint_sha256=sha256(sft_checkpoint), point_contract_sha256=digest(manifest["point_contract"]),
        train_plan_sha256=manifest["train_plan_sha256"],
        calibration_cache_sha256=manifest["phase_sha256"]["calibration"],
        scope="calibration on fresh ID task mixture; held-out source performance is empirical",
        no_test_used=True, **calibration_result,
    ))
    report = dict(
        status="PASS_ARENA_N500_HEAD_SFT_CALIBRATION", n=500, counts=COUNTS,
        adapter_training_tables=8000, unique_optimization_tables=8000 + COUNTS["head_train"] + COUNTS["sft_train"],
        adapter_sha256=SELECTED_ADAPTER_SHA, base_checkpoint_sha256=sha256(base_checkpoint),
        sft_checkpoint_sha256=sha256(sft_checkpoint), calibration_sha256=sha256(calibration_path),
        cache_manifest_sha256=manifest_sha, train_plan_sha256=manifest["train_plan_sha256"],
        point_contract_sha256=digest(manifest["point_contract"]), no_test_used=True,
        base=base_report, sft=sft_report, calibration=calibration_result,
        head_steps=args.head_steps, sft_steps=args.sft_steps, seconds=time.perf_counter() - start,
    )
    write_json_new(args.output / "completion.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026110901)
    parser.add_argument("--head-steps", type=int, default=6000)
    parser.add_argument("--sft-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    options = parser.parse_args()
    if min(options.cpu_threads, options.head_steps, options.sft_steps, options.batch_size) < 1:
        parser.error("thread, step and batch counts must be positive")
    run(options)
