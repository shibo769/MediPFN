"""Factual-only pretraining of the custom MediPFN v2 nuisance network.

Fresh mechanisms/rows are drawn from the frozen train split. Development uses
disjoint seeds from the same source groups; no held-out-source outcomes select
weights. Synthetic NIE truth is deliberately unused by this stage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from model import MediationPFNV2, ModelConfig, nested_contexts, save_checkpoint, subset


def derived_seed(*parts):
    raw = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def collate(tables, config, device="cpu"):
    b, n = len(tables), len(tables[0]["a"])
    arrays = dict(x=np.zeros((b, n, config.max_x), np.float32),
                  m=np.zeros((b, n, config.max_m), np.float32),
                  a=np.zeros((b, n), np.float32), y=np.zeros((b, n), np.float32),
                  x_mask=np.zeros((b, config.max_x), np.float32),
                  m_mask=np.zeros((b, config.max_m), np.float32),
                  x_binary=np.zeros((b, config.max_x), np.float32),
                  m_binary=np.zeros((b, config.max_m), np.float32))
    for i, t in enumerate(tables):
        x, m = np.asarray(t["x"]), np.asarray(t["m"])
        if x.ndim == 1:
            x = x.reshape(n, -1)
        if m.ndim == 1:
            m = m[:, None]
        p, q = x.shape[1], m.shape[1]
        if p > config.max_x or q > config.max_m or q < 1:
            raise ValueError("Task dimensions exceed configured model interface")
        arrays["x"][i, :, :p] = x
        arrays["m"][i, :, :q] = m
        arrays["a"][i] = np.asarray(t["a"]).reshape(n)
        arrays["y"][i] = np.asarray(t["y"]).reshape(n)
        arrays["x_mask"][i, :p] = 1
        arrays["m_mask"][i, :q] = 1
        for key, length in (("x", p), ("m", q)):
            types = t.get("x_types" if key == "x" else "mediator_types", ["continuous"] * length)
            if len(types) != length:
                raise ValueError("Feature type metadata does not match feature count")
            arrays[key + "_binary"][i, :length] = [str(v) in ("binary", "categorical") for v in types]
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError("Nonfinite observed task data")
    if not np.isin(arrays["a"], [0, 1]).all():
        raise ValueError("This v2 treatment contrast requires A in {0,1}")
    return {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}


def make_batch(seed, step, batch_size, config, n_values, query_budget, dev=False, workers=1):
    from dgp import sample_task, registry
    domain = "development" if dev else "nuisance_train"
    rng = np.random.default_rng(derived_seed(seed, domain, step))
    n = int(rng.choice(n_values))
    records = [r for r in registry() if r["default_split"] == "train"]
    families = sorted(set(r["source_group"] for r in records))
    def draw(index):
        # Retry only invalid observation splits, never based on model predictions.
        for retry in range(20):
            task_rng = np.random.default_rng(derived_seed(seed, domain, step, index, retry))
            family = families[index % len(families)] if dev else str(task_rng.choice(families))
            candidates = [r["source_id"] for r in records if r["source_group"] == family]
            source = str(task_rng.choice(candidates))
            # Even/odd seed namespaces are disjoint for train/DEV mechanisms and rows.
            mechanism_seed = 2*derived_seed(seed, domain, "mechanism", step, index, retry)+int(dev)
            table_seed = 2*derived_seed(seed, domain, "observations", step, index, retry)+int(dev)
            t = sample_task(source, n, table_seed, mechanism_seed=mechanism_seed, compute_truth=False)
            order = np.random.default_rng(derived_seed(seed, domain, "order", step, index, retry)).permutation(n)
            for k in ("x", "a", "m", "y"):
                t[k] = np.asarray(t[k])[order]
            counts = np.bincount(np.asarray(t["a"][:n//2], int), minlength=2)
            if counts.min() >= 2:
                return t
        raise RuntimeError("Repeated task support splits lacked both treatment groups")
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tables = list(pool.map(draw, range(batch_size)))
    else:
        tables = [draw(i) for i in range(batch_size)]
    packed = collate(tables, config)
    cut, q = n//2, min(n-n//2, query_budget)
    return dict(support=subset(packed, slice(None, cut)), query=subset(packed, slice(cut, cut+q)),
                provenance=[t["metadata"] for t in tables],
                drawn_rows=n, query_rows=q)


def factual_loss(model, batch, device, eta_weight=1.):
    support = {k: v.to(device, non_blocking=True) for k, v in batch["support"].items()}
    query = {k: v.to(device, non_blocking=True) for k, v in batch["query"].items()}
    context = model.encode(support)
    pi = model.predict(context, query["x"], query["m"], "pi")
    g = model.predict(context, query["x"], query["m"], "g")
    mu = model.predict(context, query["x"], query["m"], "mu", query["a"])
    loss_pi = F.binary_cross_entropy_with_logits(pi, query["a"])
    loss_g = F.binary_cross_entropy_with_logits(g, query["a"])
    loss_mu = ((mu - query["y"]) / context["y_scale"]).square().mean()
    loss_eta = mu.new_zeros(())
    if eta_weight > 0:
        eta0, eta1 = nested_contexts(model, support)
        with torch.no_grad():
            pseudo = model.predict(context, query["x"], query["m"], "mu", torch.ones_like(query["a"]))
        for a, eta in ((0, eta0), (1, eta1)):
            prediction = model.predict(eta, query["x"], query["m"], "eta"+str(a))
            mask = (query["a"] == a).to(prediction.dtype)
            loss_eta = loss_eta + (((prediction-pseudo) / context["y_scale"]).square()*mask).sum()/mask.sum().clamp_min(1)
        loss_eta = loss_eta / 2
    loss = loss_pi + loss_g + loss_mu + eta_weight*loss_eta
    return loss, {"bce_pi": loss_pi.detach(), "bce_g": loss_g.detach(),
                  "scaled_mse_mu": loss_mu.detach(), "scaled_mse_eta": loss_eta.detach()}


def source_bindings():
    root = Path(__file__).parent
    paths = [root/"model.py", root/"train.py", root/"dgp"/"__init__.py",
             root/"dgp"/"generators.py", root/"dgp"/"source_manifest.json"]
    paths += sorted((root/"dgp"/"vendor").glob("*.py"))
    paths += sorted((root/"dgp"/"vendor").glob("*.json"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def train(args):
    from dgp import available_sources
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"best.pt").exists() and not args.resume:
        raise FileExistsError("Existing run requires explicit --resume, never overwrite")
    threads = args.cpu_threads or int(os.environ.get("NSLOTS", os.cpu_count() or 1))
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    config = ModelConfig(width=args.width, heads=args.heads, depth=args.depth,
                         inducing_tokens=args.inducing_tokens, max_x=args.max_x, max_m=args.max_m)
    model = MediationPFNV2(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = args.device.startswith("cuda") and not args.no_amp
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = (torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
              if hasattr(torch.amp, "GradScaler") else
              torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16))
    start_step, best, best_step, history = 0, float("inf"), None, []
    bindings = source_bindings()
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        if previous["config"] != asdict(config) or previous["source_sha256"] != bindings:
            raise ValueError("Resume config/source differs; start an isolated new run")
        allowed_changes = {"output", "resume", "steps", "device", "cpu_threads", "data_workers"}
        old_args = previous["run_metadata"]["arguments"]
        if any(old_args.get(k) != v for k,v in vars(args).items() if k not in allowed_changes):
            raise ValueError("Resume changes a statistical/training argument; start an isolated new run")
        model.load_state_dict(previous["state_dict"])
        optimizer.load_state_dict(previous["optimizer_state"])
        scaler.load_state_dict(previous["scaler_state"])
        start_step, best, best_step, history = previous["step"], previous["best_loss"], previous["best_step"], previous["history"]
    elif args.init_checkpoint:
        previous = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if previous["config"] != asdict(config):
            raise ValueError("Initialization checkpoint architecture differs")
        model.load_state_dict(previous["state_dict"])
    development = [make_batch(args.data_seed, i, args.dev_batch_size, config, [n], args.query_budget,
                              dev=True, workers=args.data_workers) for i,n in enumerate(args.n_values)]
    dev_mechanisms = {r["mechanism_id"] for b in development for r in b["provenance"]}
    stage = "factual_and_nested_nuisance" if args.eta_weight > 0 else "factual_nuisance_only"
    metadata = dict(arguments=vars(args), source_sha256=bindings, train_sources=available_sources("train"),
                    ood_source_ids=available_sources("ood_source"), ood_composition_ids=available_sources("ood_composition"),
                    objective="BCE(A,pi(X))+BCE(A,g(X,M))+scaled factual outcome MSE" +
                              ("+inner-OOF nested eta regression" if args.eta_weight > 0 else ""),
                    truth_used=False, selection="same-source independent-mechanism DEV factual loss only (pi,g,mu)",
                    stage=stage, ready_for_nie_score=args.eta_weight > 0,
                    dev_provenance=[r for b in development for r in b["provenance"]],
                    created_utc=datetime.now(timezone.utc).isoformat())
    (output/"run_config.json").write_text(json.dumps(metadata, indent=2, default=str)+"\n", encoding="utf-8")
    started = time.perf_counter()
    last_parts = {}
    group_counts, source_counts, unique_mechanisms = Counter(), Counter(), set()
    actual_tables, actual_rows, actual_query_rows = 0, 0, 0
    if args.resume:
        accounting = previous.get("accounting", {})
        group_counts.update(accounting.get("source_group_counts", {}))
        source_counts.update(accounting.get("source_record_counts", {}))
        unique_mechanisms.update(accounting.get("mechanism_ids", []))
        actual_tables, actual_rows, actual_query_rows = [accounting.get(k,0) for k in ("actual_tables","drawn_observation_rows","used_query_rows")]
    prefetcher = ThreadPoolExecutor(max_workers=1)
    def schedule(step):
        return prefetcher.submit(make_batch, args.data_seed, step, args.batch_size, config,
                                 args.n_values, args.query_budget, False, args.data_workers)
    next_batch = schedule(start_step+1) if start_step < args.steps else None
    for step in range(start_step, args.steps+1):
        if step > start_step:
            model.train()
            batch = next_batch.result()
            next_batch = schedule(step+1) if step < args.steps else None
            for record in batch["provenance"]:
                if record["mechanism_id"] in dev_mechanisms:
                    raise ValueError("Training mechanism fingerprint overlaps the frozen DEV set")
                group_counts[record["source_group"]] += 1
                source_counts[record["source_id"]] += 1
                unique_mechanisms.add(record["mechanism_id"])
            actual_tables += len(batch["provenance"])
            actual_rows += len(batch["provenance"])*batch["drawn_rows"]
            actual_query_rows += len(batch["provenance"])*batch["query_rows"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                loss, last_parts = factual_loss(model, batch, args.device, args.eta_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite nuisance loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite nuisance gradient")
            scaler.step(optimizer)
            scaler.update()
        if step == start_step or step % args.dev_every == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                values = [float(factual_loss(model, b, args.device, 0.)[0]) for b in development]
            dev = float(np.mean(values))
            if not math.isfinite(dev):
                raise FloatingPointError("Nonfinite development loss")
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            entry = dict(step=step, dev_loss=dev, dev_by_n=dict(zip(args.n_values, values)),
                         seconds=time.perf_counter()-started, actual_training_tables=actual_tables,
                         unique_training_mechanisms=len(unique_mechanisms), source_group_counts=dict(group_counts),
                         source_record_counts=dict(source_counts), drawn_observation_rows=actual_rows,
                         used_query_rows=actual_query_rows, **{k: float(v) for k,v in last_parts.items()})
            accounting = dict(actual_tables=actual_tables, drawn_observation_rows=actual_rows,
                              used_query_rows=actual_query_rows, source_group_counts=dict(group_counts),
                              source_record_counts=dict(source_counts), mechanism_ids=sorted(unique_mechanisms))
            history.append(entry)
            if dev < best:
                best, best_step = dev, step
                save_checkpoint(output/"best.pt", model, stage=stage, selected_step=step,
                                selected_dev_loss=dev, source_sha256=bindings, run_metadata=metadata, accounting=accounting)
            save_checkpoint(output/"last.pt", model, stage="factual_nuisance_resume", step=step,
                            optimizer_state=optimizer.state_dict(), scaler_state=scaler.state_dict(),
                            history=history, best_loss=best, best_step=best_step, source_sha256=bindings,
                            run_metadata=metadata, accounting=accounting)
            entry.update(best_loss=best, best_step=best_step, parameters=model.parameter_count)
            with (output/"training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry)+"\n")
            print(json.dumps(entry), flush=True)
    prefetcher.shutdown(wait=True)
    return output/"best.pt"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026110301)
    p.add_argument("--data-seed", type=int, default=2026110302)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dev-batch-size", type=int, default=16)
    p.add_argument("--dev-every", type=int, default=100)
    p.add_argument("--n-values", type=int, nargs="+", default=[200,600,1000])
    p.add_argument("--query-budget", type=int, default=64)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--inducing-tokens", type=int, default=32)
    p.add_argument("--max-x", type=int, default=32)
    p.add_argument("--max-m", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.)
    p.add_argument("--eta-weight", type=float, default=0., help="0 launches factual pretraining only; nested eta is a subsequent stage")
    p.add_argument("--cpu-threads", type=int)
    p.add_argument("--data-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--init-checkpoint", help="Initialize weights for a separate new stage/run without optimizer state")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.steps < 1 or args.dev_every < 1 or min(args.n_values) < 16:
        raise ValueError("Invalid training dimensions/budget")
    train(args)
