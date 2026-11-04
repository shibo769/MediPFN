"""Fit eta0/eta1 with a frozen factual teacher on independent training tasks.

The fixed teacher produces inner-OOF mu1 labels for outer support and predicts
outer-query mu1 without its outcomes. A separate copied network learns eta;
the factual teacher's weights never change. DEV targets are thus fixed across
eta checkpoints. No true NIE, calibration, or OOD data enter this stage.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from model import MediationPFNV2, load_checkpoint, nested_contexts, save_checkpoint
from train import make_batch, source_bindings


def nested_loss(student, teacher, batch, device):
    support = {k:v.to(device) for k,v in batch["support"].items()}
    query = {k:v.to(device) for k,v in batch["query"].items()}
    with torch.no_grad():
        context = teacher.encode(support)
        pseudo = teacher.predict(context, query["x"], query["m"], "mu", torch.ones_like(query["a"]))
    eta_contexts = nested_contexts(student, support, teacher=teacher)
    loss = pseudo.new_zeros(())
    for a, eta in enumerate(eta_contexts):
        prediction = student.predict(eta, query["x"], query["m"], "eta"+str(a))
        valid = (query["a"] == a).to(prediction.dtype)
        loss = loss + (((prediction-pseudo)/context["y_scale"]).square()*valid).sum()/valid.sum().clamp_min(1)
    return loss/2


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"best.pt").exists():
        raise FileExistsError("Start an isolated eta run; existing checkpoints are protected")
    torch.set_num_threads(args.cpu_threads or int(os.environ.get("NSLOTS", os.cpu_count() or 1)))
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    teacher, teacher_metadata = load_checkpoint(args.teacher, args.device)
    teacher.eval().requires_grad_(False)
    student = MediationPFNV2(teacher.config).to(args.device)
    student.load_state_dict(teacher.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    teacher_sha = hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest()
    bindings = source_bindings()
    bindings["train_nested.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    amp = args.device.startswith("cuda") and not args.no_amp
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = (torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
              if hasattr(torch.amp,"GradScaler") else torch.cuda.amp.GradScaler(enabled=amp and dtype == torch.float16))
    development = [make_batch(args.data_seed, i, args.dev_batch_size, teacher.config, [n], args.query_budget,
                              dev=True, workers=args.data_workers) for i,n in enumerate(args.n_values)]
    dev_ids = {p["mechanism_id"] for b in development for p in b["provenance"]}
    metadata = dict(arguments=vars(args), source_sha256=bindings, frozen_teacher_sha256=teacher_sha,
                    teacher_selected_step=teacher_metadata.get("selected_step"),
                    stage="nested_eta_regression", ready_for_nie_score=True,
                    selection="fixed frozen-teacher inner-OOF DEV pseudo-outcome loss; train-source mechanisms only",
                    truth_used=False, dev_provenance=[p for b in development for p in b["provenance"]],
                    created_utc=datetime.now(timezone.utc).isoformat())
    (output/"run_config.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    best, best_step, start, history = float("inf"),None,time.perf_counter(),[]
    for step in range(args.steps+1):
        if step:
            batch = make_batch(args.data_seed, step, args.batch_size, teacher.config, args.n_values,
                               args.query_budget, workers=args.data_workers)
            if any(p["mechanism_id"] in dev_ids for p in batch["provenance"]):
                raise ValueError("Eta train/DEV mechanism fingerprints overlap")
            student.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=dtype,enabled=amp):
                loss = nested_loss(student, teacher, batch, args.device)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite nested-regression loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
            if not bool(torch.isfinite(grad)):
                raise FloatingPointError("Nonfinite nested-regression gradient")
            scaler.step(optimizer)
            scaler.update()
        if step == 0 or step%args.dev_every == 0 or step == args.steps:
            student.eval()
            with torch.inference_mode():
                values = [float(nested_loss(student,teacher,b,args.device)) for b in development]
            dev = float(np.mean(values))
            if not math.isfinite(dev):
                raise FloatingPointError("Nonfinite fixed-target eta DEV loss")
            entry = dict(step=step,dev_loss=dev,seconds=time.perf_counter()-start,actual_training_tables=step*args.batch_size)
            history.append(entry)
            if dev < best:
                best,best_step = dev,step
                save_checkpoint(output/"best.pt",student,stage="nested_eta_regression",selected_step=step,
                                selected_dev_loss=dev,frozen_teacher_sha256=teacher_sha,source_sha256=bindings,
                                run_metadata=metadata)
            save_checkpoint(output/"last.pt",student,stage="nested_eta_resume",step=step,optimizer_state=optimizer.state_dict(),
                            scaler_state=scaler.state_dict(),frozen_teacher_sha256=teacher_sha,source_sha256=bindings,
                            run_metadata=metadata,history=history,best_step=best_step,best_loss=best)
            print(json.dumps(dict(entry,best_loss=best,best_step=best_step)),flush=True)
            with (output/"training.jsonl").open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(entry)+"\n")
    return output/"best.pt"


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=2026110311)
    p.add_argument("--data-seed",type=int,default=2026110312)
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--dev-batch-size",type=int,default=16)
    p.add_argument("--dev-every",type=int,default=250)
    p.add_argument("--n-values",type=int,nargs="+",default=[200,600,1000])
    p.add_argument("--query-budget",type=int,default=64)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--cpu-threads",type=int)
    p.add_argument("--data-workers",type=int,default=4)
    p.add_argument("--no-amp",action="store_true")
    train(p.parse_args())
