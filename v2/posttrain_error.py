"""Confidence-guided scaling and group-balanced quantile post-training for v2.

Frozen base quantiles first receive an observable-feature coverage-confidence
scale. A separately fitted correction then changes their central location and
positive tail widths. Source/mechanism IDs weight training risk only; no such
ID is an inference input. All fitting/selection precedes fresh calibration.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from error_model import SignedErrorQuantiles,load_error_checkpoint,pinball_loss
from train_error import read_cache,sha256
from error_binding import assert_same_feature_contract

POST_FORMAT="medipfn-v2-confidence-balanced-quantiles-1"
SCALE_GRID=(.4,.6,.8,1.,1.25,1.6,2.,2.5,3.5)


def balanced_weights(provenance):
    """Equal groups, equal mechanisms within groups, equal rows within each."""
    grouped=defaultdict(set)
    repeats=Counter()
    for p in provenance:
        key=(p["source_group"],p["mechanism_id"])
        grouped[key[0]].add(key[1])
        repeats[key]+=1
    count=len(grouped)
    if not count:
        raise ValueError("No post-training mechanism groups")
    weights=[1/(count*len(grouped[p["source_group"]])*repeats[(p["source_group"],p["mechanism_id"])]) for p in provenance]
    return np.asarray(weights,np.float32)


def base_geometry(base,features):
    normalized=((features-base.feature_center)/base.feature_scale).clamp(-20,20)
    raw=base.output(base.body(normalized))
    levels=len(base.config.alphas)
    center=raw[:,0,None]
    left=F.softplus(raw[:,1:1+levels]+base.width_offset).cumsum(-1)
    right=F.softplus(raw[:,1+levels:]+base.width_offset).cumsum(-1)
    return normalized,center,left,right


class CoverageConfidence(nn.Module):
    def __init__(self,feature_count,width=32,scale_grid=SCALE_GRID):
        super().__init__()
        self.width=width
        if any(s<=0 for s in scale_grid) or any(a>=b for a,b in zip(scale_grid,scale_grid[1:])):
            raise ValueError("Confidence scale grid must be positive and increasing")
        self.register_buffer("scale_grid",torch.tensor(scale_grid,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(feature_count+4,width),nn.Tanh(),nn.Linear(width,2))
        self.register_buffer("temperature",torch.tensor(1.))

    def logits(self,normalized,center,left,right,alphas):
        b,levels=left.shape
        geometry=torch.stack((center.expand(-1,levels),left.clamp_min(1e-5).log(),
                              right.clamp_min(1e-5).log(),left.new_tensor(alphas).log()[None,:].expand(b,-1)), -1)
        inputs=torch.cat((normalized[:,None,:].expand(-1,levels,-1),geometry.clamp(-20,20)),-1)
        raw=self.net(inputs)
        slope=F.softplus(raw[:,:,1])+1e-4
        return raw[:,:,0,None]+slope[:,:,None]*self.scale_grid.log()[None,None,:]

    def forward(self,normalized,center,left,right,alphas):
        logits=self.logits(normalized,center,left,right,alphas)/self.temperature
        probabilities=logits.sigmoid()
        target=probabilities.new_tensor([1-a for a in alphas])[None,:,None]
        qualifies=probabilities>=target
        # argmax returns zero if all fail; replace those with the largest scale.
        chosen=qualifies.to(torch.int64).argmax(-1)
        chosen=torch.where(qualifies.any(-1),chosen,torch.full_like(chosen,len(self.scale_grid)-1))
        return self.scale_grid[chosen],probabilities


class QuantileCorrection(nn.Module):
    def __init__(self,feature_count,levels,width=16,log_scale_clip=3.):
        super().__init__()
        self.width=width
        self.log_scale_clip=log_scale_clip
        self.net=nn.Sequential(nn.Linear(feature_count,width),nn.Tanh(),nn.Linear(width,1+2*levels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,normalized,center,left,right):
        levels=left.shape[1]
        raw=self.net(normalized)
        offset=raw[:,0,None]
        lo_adjust=raw[:,1:1+levels].clamp(-self.log_scale_clip,self.log_scale_clip)
        hi_adjust=raw[:,1+levels:].clamp(-self.log_scale_clip,self.log_scale_clip)
        lo_width=(left.clamp_min(1e-5)*lo_adjust.exp()).cummax(1).values
        hi_width=(right.clamp_min(1e-5)*hi_adjust.exp()).cummax(1).values
        pairs=torch.stack((center+offset-lo_width,center+offset+hi_width),-1)
        return pairs,raw


class PosttrainedError(nn.Module):
    def __init__(self,base,confidence_width=32,correction_width=16,scale_grid=SCALE_GRID):
        super().__init__()
        self.base=base.eval().requires_grad_(False)
        self.config=base.config
        self.confidence=CoverageConfidence(self.config.feature_count,confidence_width,scale_grid)
        self.correction=QuantileCorrection(self.config.feature_count,len(self.config.alphas),correction_width)

    def confidence_geometry(self,features):
        normalized,center,left,right=base_geometry(self.base,features)
        scale,probability=self.confidence(normalized,center,left,right,self.config.alphas)
        left=(left*scale).cummax(1).values
        right=(right*scale).cummax(1).values
        return normalized,center,left,right,scale,probability

    def reference(self,features):
        _,center,left,right,_,_=self.confidence_geometry(features)
        return torch.stack((center-left,center+right),-1)

    def forward(self,features):
        normalized,center,left,right,_,_=self.confidence_geometry(features)
        return self.correction(normalized,center,left,right)[0]


def save_post_checkpoint(path,model,**metadata):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save(dict(format=POST_FORMAT,config=asdict(model.config),state_dict=model.state_dict(),
                    confidence_width=model.confidence.width,correction_width=model.correction.width,
                    scale_grid=model.confidence.scale_grid.cpu().tolist(),**metadata),temporary)
    temporary.replace(path)


def load_post_checkpoint(path,device="cpu"):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint.get("format")!=POST_FORMAT:
        raise ValueError("Not a confidence/group-balanced v2 checkpoint")
    model=PosttrainedError(SignedErrorQuantiles(checkpoint["config"]),checkpoint["confidence_width"],
                          checkpoint["correction_width"],checkpoint["scale_grid"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(),checkpoint


def confidence_targets(center,left,right,response,scale_grid):
    low=center[:,:,None]-left[:,:,None]*scale_grid[None,None,:]
    high=center[:,:,None]+right[:,:,None]*scale_grid[None,None,:]
    return ((response[:,None,None]>=low)&(response[:,None,None]<=high)).to(left.dtype)


def coverage_and_length(pairs,response,scales,provenance,alphas):
    values=pairs.detach().cpu().numpy()
    target=response.detach().cpu().numpy()
    observed_scale=np.asarray(scales)
    covered=(target[:,None]>=values[:,:,0])&(target[:,None]<=values[:,:,1])
    length=(values[:,:,1]-values[:,:,0])*observed_scale[:,None]
    groups=sorted(set(p["source_group"] for p in provenance))
    by_group={g:covered[np.asarray([p["source_group"]==g for p in provenance])].mean(0).tolist() for g in groups}
    minimum_margin=min(by_group[g][i]-(1-a) for g in groups for i,a in enumerate(alphas))
    return dict(coverage=covered.mean(0).tolist(),mean_length=length.mean(0).tolist(),
                source_group_coverage=by_group,minimum_source_coverage_margin=float(minimum_margin))


def fit(args):
    train,train_meta=read_cache(args.train_cache,"post_train")
    dev,dev_meta=read_cache(args.dev_cache,"post_dev")
    base,base_checkpoint=load_error_checkpoint(args.error_checkpoint,args.device)
    previous_metadata=base_checkpoint["run_metadata"]
    for metadata in (train_meta,dev_meta):
        assert_same_feature_contract(metadata,previous_metadata)
        if metadata["relative_scale_floor"]!=base.config.relative_scale_floor:
            raise ValueError("Post-training scale differs from error fitting")
    old_train_ids=set(previous_metadata["train_mechanism_ids"])
    old_dev_ids=set(previous_metadata["dev_mechanism_ids"])
    old_ids=old_train_ids|old_dev_ids
    train_ids={p["mechanism_id"] for p in train_meta["provenance"]}
    dev_ids={p["mechanism_id"] for p in dev_meta["provenance"]}
    if train_ids&dev_ids or old_ids&(train_ids|dev_ids):
        raise ValueError("Post-training mechanisms overlap prior stages or one another")
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"best.pt").exists():
        raise FileExistsError("Existing post-training checkpoint is protected")
    torch.manual_seed(args.seed)
    model=PosttrainedError(base,args.confidence_width,args.correction_width).to(args.device)
    features=torch.from_numpy(train["features"]).to(args.device)
    response=torch.from_numpy(train["response"]).to(args.device,dtype=torch.float32)
    weights=torch.from_numpy(balanced_weights(train_meta["provenance"])).to(args.device)
    dev_features=torch.from_numpy(dev["features"]).to(args.device)
    dev_response=torch.from_numpy(dev["response"]).to(args.device,dtype=torch.float32)
    dev_weights=torch.from_numpy(balanced_weights(dev_meta["provenance"])).to(args.device)
    with torch.no_grad():
        geometry=base_geometry(base,features)
        dev_geometry=base_geometry(base,dev_features)
        targets=confidence_targets(*geometry[1:],response,model.confidence.scale_grid)
        dev_targets=confidence_targets(*dev_geometry[1:],dev_response,model.confidence.scale_grid)
    optimizer=torch.optim.AdamW(model.confidence.parameters(),lr=args.lr,weight_decay=.01)
    best,best_state,best_epoch=float("inf"),None,0
    confidence_history=[]
    started=time.perf_counter()
    for epoch in range(args.confidence_epochs+1):
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            logits=model.confidence.logits(*geometry,base.config.alphas)
            loss=(F.binary_cross_entropy_with_logits(logits,targets,reduction="none").mean((1,2))*weights).sum()
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.confidence.parameters(),1.)
            if not bool(torch.isfinite(loss) and torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite coverage-confidence fit")
            optimizer.step()
        with torch.no_grad():
            logits=model.confidence.logits(*dev_geometry,base.config.alphas)
            risk=float((F.binary_cross_entropy_with_logits(logits,dev_targets,reduction="none").mean((1,2))*dev_weights).sum())
        if not math.isfinite(risk):
            raise FloatingPointError("Nonfinite confidence DEV risk")
        confidence_history.append(dict(epoch=epoch,dev_balanced_bce=risk))
        if risk<best:
            best,best_epoch=risk,epoch
            best_state={k:v.detach().clone() for k,v in model.confidence.state_dict().items()}
    model.confidence.load_state_dict(best_state)
    with torch.no_grad():
        logits=model.confidence.logits(*dev_geometry,base.config.alphas)
        temperatures=(.5,.75,1.,1.5,2.,3.)
        risks=[float((F.binary_cross_entropy_with_logits(logits/t,dev_targets,reduction="none").mean((1,2))*dev_weights).sum()) for t in temperatures]
        model.confidence.temperature.fill_(temperatures[int(np.argmin(risks))])
        reference_pairs=model.reference(dev_features)
        reference=coverage_and_length(reference_pairs,dev_response,dev["scale"],dev_meta["provenance"],base.config.alphas)
    model.confidence.requires_grad_(False)
    metadata=dict(previous_metadata)
    metadata.update(stage="confidence_guided_group_balanced_posttraining",base_error_checkpoint_sha256=sha256(args.error_checkpoint),
                    post_train_cache_sha256=sha256(args.train_cache),post_dev_cache_sha256=sha256(args.dev_cache),
                    post_train_cache_metadata_sha256=sha256(Path(args.train_cache).with_suffix(".json")),
                    post_dev_cache_metadata_sha256=sha256(Path(args.dev_cache).with_suffix(".json")),
                    posttrain_source_sha256=sha256(__file__),posttrain_arguments=vars(args),
                    train_mechanism_ids=sorted(old_train_ids|train_ids),dev_mechanism_ids=sorted(old_dev_ids|dev_ids),
                    post_training="coverage-confidence scaling, temperature selection, and source/mechanism-balanced quantile correction on fresh post-training tasks",
                    confidence_selected_epoch=best_epoch,confidence_temperature=float(model.confidence.temperature),
                    confidence_dev_bce=best,posttrain_grouping="equal source groups, equal mechanisms, equal repeats; IDs used only in training risk",
                    selection="maximum minimum source-group coverage margin subject to each-level ID coverage loss <= .05 and width <= 1.5x confidence reference; tie by mean relative width",
                    reference_dev=reference,created_utc=datetime.now(timezone.utc).isoformat())
    (output/"run_config.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    optimizer=torch.optim.AdamW(model.correction.parameters(),lr=args.lr,weight_decay=.01)
    best_key=None
    correction_history=[]
    for epoch in range(args.correction_epochs+1):
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            normalized,center,left,right,_,_=model.confidence_geometry(features)
            pairs,raw=model.correction(normalized,center,left,right)
            tau=pairs.new_tensor([[a/2,1-a/2] for a in base.config.alphas])
            residual=response[:,None,None]-pairs
            per_case=torch.maximum(tau*residual,(tau-1)*residual).mean((1,2))
            loss=(per_case*weights).sum()+args.output_penalty*raw.square().mean()
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.correction.parameters(),1.)
            if not bool(torch.isfinite(loss) and torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite group-balanced quantile fit")
            optimizer.step()
        with torch.no_grad():
            pairs=model(dev_features)
            metrics=coverage_and_length(pairs,dev_response,dev["scale"],dev_meta["provenance"],base.config.alphas)
        eligible=all(cp>=ref_cp-.05 and length<=1.5*ref_length+1e-12
                     for cp,ref_cp,length,ref_length in zip(metrics["coverage"],reference["coverage"],metrics["mean_length"],reference["mean_length"]))
        relative_width=float(np.mean(np.asarray(metrics["mean_length"])/np.maximum(reference["mean_length"],1e-12)))
        key=(metrics["minimum_source_coverage_margin"],-relative_width)
        entry=dict(epoch=epoch,eligible=eligible,relative_reference_width=relative_width,**metrics)
        correction_history.append(entry)
        if eligible and (best_key is None or key>best_key):
            best_key=key
            save_post_checkpoint(output/"best.pt",model,run_metadata=metadata,selected_epoch=epoch,selected_dev_metrics=metrics)
        save_post_checkpoint(output/"last.pt",model,run_metadata=metadata,epoch=epoch,
                             optimizer_state=optimizer.state_dict(),confidence_history=confidence_history,correction_history=correction_history)
        if epoch%20==0 or epoch==args.correction_epochs:
            print(json.dumps(dict(stage="posttrain",seconds=time.perf_counter()-started,**entry)),flush=True)
    if best_key is None:
        raise RuntimeError("Even the confidence reference was ineligible; numerical inconsistency")
    (output/"history.json").write_text(json.dumps(dict(confidence=confidence_history,correction=correction_history),indent=2)+"\n",encoding="utf-8")
    return output/"best.pt"


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--error-checkpoint",required=True)
    p.add_argument("--train-cache",required=True)
    p.add_argument("--dev-cache",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=2026110331)
    p.add_argument("--confidence-epochs",type=int,default=100)
    p.add_argument("--correction-epochs",type=int,default=160)
    p.add_argument("--confidence-width",type=int,default=32)
    p.add_argument("--correction-width",type=int,default=16)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--output-penalty",type=float,default=1e-4)
    fit(p.parse_args())
