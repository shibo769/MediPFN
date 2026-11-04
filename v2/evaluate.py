"""Nested cross-fitted NIE1 scores; factual queries never supply outcome labels.

The emitted intervals here are uncalibrated score-Wald intervals. They are not
the learned-error/post-training interval and do not assert distribution-free or
OOD coverage. A factual-only checkpoint is rejected unless a separately fitted
nested-regression checkpoint is supplied.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import time
import numpy as np
import torch
from model import load_checkpoint, nested_contexts, subset
from train import collate


def bayes_ratio(pi, g):
    """p(M|A=0,X)/p(M|A=1,X), valid for scalar/vector M on common support."""
    return (1-g)/g * pi/(1-pi)


def nie_score(a, y, pi, g, mu1, eta0, eta1):
    psi11 = a/pi*(y-eta1)+eta1
    psi10 = a/pi*bayes_ratio(pi,g)*(y-mu1)+(1-a)/(1-pi)*(mu1-eta0)+eta0
    return psi11-psi10


@torch.inference_mode()
def estimate_table(model, table, eta_model=None, device="cpu", seed=123, folds=2,
                   probability_floor=.01, return_scores=False):
    if not 0 < probability_floor < .5:
        raise ValueError("Probability floor must lie in (0,.5)")
    if folds < 2:
        raise ValueError("At least two outer folds are required")
    eta_model = model if eta_model is None else eta_model
    model.eval()
    eta_model.eval()
    packed = collate([table], model.config, device)
    a = np.asarray(table["a"]).reshape(-1)
    n = len(a)
    rng = np.random.default_rng(seed)
    assignment = np.empty(n, int)
    for group in (0,1):
        index = np.flatnonzero(a == group)
        rng.shuffle(index)
        assignment[index] = np.arange(len(index)) % folds
    score = np.full(n, np.nan)
    diagnostics, pooled = [], []
    started = time.perf_counter()
    for fold in range(folds):
        support_index = np.flatnonzero(assignment != fold)
        rng.shuffle(support_index)
        query_index = np.flatnonzero(assignment == fold)
        support = subset(packed, torch.as_tensor(support_index, device=device))
        query = subset(packed, torch.as_tensor(query_index, device=device))
        context = model.encode(support)
        eta0_context, eta1_context = nested_contexts(eta_model, support, teacher=model)
        raw_pi = model.predict(context, query["x"], query["m"], "pi").sigmoid()
        raw_g = model.predict(context, query["x"], query["m"], "g").sigmoid()
        pi = raw_pi.clamp(probability_floor,1-probability_floor)
        g = raw_g.clamp(probability_floor,1-probability_floor)
        mu1 = model.predict(context, query["x"], query["m"], "mu", torch.ones_like(query["a"]))
        eta0 = eta_model.predict(eta0_context, query["x"], query["m"], "eta0")
        eta1 = eta_model.predict(eta1_context, query["x"], query["m"], "eta1")
        values = nie_score(query["a"], query["y"], pi, g, mu1, eta0, eta1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Nonfinite cross-fitted NIE scores")
        score[query_index] = values.cpu().numpy().reshape(-1)
        diagnostics.append(dict(fold=fold, support_rows=len(support_index), query_rows=len(query_index),
                                pi_clipped=float((raw_pi != pi).float().mean()),
                                g_clipped=float((raw_g != g).float().mean()),
                                ratio_max=float(bayes_ratio(pi,g).max())))
        pooled.append(context["pooled"].cpu().numpy().reshape(-1))
    if not np.isfinite(score).all():
        raise FloatingPointError("Incomplete outer fold assignment")
    estimate = float(score.mean())
    se = float(score.std(ddof=1)/np.sqrt(n))
    result = dict(estimate=estimate, standard_error=se, n=n, outer_folds=folds,
                  probability_floor=probability_floor, diagnostics=diagnostics,
                  seconds=time.perf_counter()-started, interval_type="score-Wald; uncalibrated",
                  pooled_context=np.mean(pooled, axis=0).tolist())
    for nominal in (.90,.95,.99):
        radius = NormalDist().inv_cdf((1+nominal)/2)*se
        result["interval_"+str(round(nominal*100))] = [estimate-radius, estimate+radius]
    if return_scores:
        result["scores"] = score.tolist()
        result["fold_assignment"] = assignment.tolist()
    return result


def load_nuisances(checkpoint, eta_checkpoint=None, device="cpu"):
    model, meta = load_checkpoint(checkpoint, device)
    if eta_checkpoint:
        eta_model, eta_meta = load_checkpoint(eta_checkpoint, device)
        actual = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if eta_meta.get("frozen_teacher_sha256") != actual:
            raise ValueError("Nested regression was fitted with a different factual teacher")
        if eta_model.config != model.config:
            raise ValueError("Nuisance/eta model configs differ")
    else:
        if not meta.get("run_metadata", {}).get("ready_for_nie_score", False):
            raise ValueError("Factual-only checkpoint needs a trained --eta-checkpoint")
        eta_model = model
    return model, eta_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eta-checkpoint")
    parser.add_argument("--source", required=True)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026110391)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probability-floor", type=float, default=.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from dgp import sample_task
    task = sample_task(args.source, args.n, args.seed)
    model, eta_model = load_nuisances(args.checkpoint, args.eta_checkpoint, args.device)
    result = estimate_table(model, task, eta_model, args.device, args.seed, probability_floor=args.probability_floor)
    result.update(truth=float(task["truth"]), metadata=task["metadata"])
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
