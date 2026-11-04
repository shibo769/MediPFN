"""Independent task-level calibration after selecting the signed-error model.

The finite-sample rank is justified for exchangeable calibration and future
tasks from the registered ID mixture, conditional on the frozen predictor. It
does not imply conditional/per-source coverage or arbitrary OOD validity.
Changing the nuisance/eta/error model or post-training invalidates this asset.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from error_model import interval_from_pair,load_error_checkpoint,uncertainty_features
from train_error import read_cache,sha256
from error_binding import assert_same_feature_contract,assert_bound_prediction,feature_contract

CALIBRATION_FORMAT="medipfn-v2-independent-task-calibration-1"


def load_selected_error_checkpoint(path,device="cpu"):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint.get("format")=="medipfn-v2-confidence-balanced-quantiles-1":
        from posttrain_error import load_post_checkpoint
        model,metadata=load_post_checkpoint(path,device)
    else:
        model,metadata=load_error_checkpoint(path,device)
    model._deployment_contract=feature_contract(metadata["run_metadata"])
    model._checkpoint_sha256=sha256(path)
    return model,metadata


def conformal_expansions(pairs,response,alphas):
    pairs=np.asarray(pairs,float)
    response=np.asarray(response,float).reshape(-1)
    if pairs.shape!=(len(response),len(alphas),2):
        raise ValueError("Calibration quantile/response shape mismatch")
    if not np.isfinite(pairs).all() or not np.isfinite(response).all():
        raise FloatingPointError("Nonfinite calibration inputs")
    if np.any(pairs[:,:,0]>pairs[:,:,1]):
        raise ValueError("Crossing error quantile predictions")
    scores=np.maximum(pairs[:,:,0]-response[:,None],response[:,None]-pairs[:,:,1])
    m=len(response)
    raw,ranks=[],[]
    for level,alpha in enumerate(alphas):
        rank=math.ceil((m+1)*(1-alpha))
        if rank>m:
            raise ValueError("Too few calibration tasks for a finite interval at alpha="+str(alpha))
        quantile=float(np.partition(scores[:,level],rank-1)[rank-1])
        raw.append(max(0.,quantile))
        ranks.append(rank)
    # Extra expansion preserves the nested confidence-level presentation and
    # can only make each separately calibrated interval more conservative.
    used=np.maximum.accumulate(raw)
    return dict(raw_expansions=raw,expansions=used.tolist(),ranks=ranks,tasks=m)


def calibrate(args):
    output=Path(args.output)
    if output.exists():
        raise FileExistsError("Calibration assets are immutable; choose a new output path")
    arrays,cache_meta=read_cache(args.calibration_cache,"calibration")
    if cache_meta["draw_design"]!="iid uniform source-group and sample-size mixture":
        raise ValueError("Calibration cache must use the registered iid task mixture")
    model,checkpoint=load_selected_error_checkpoint(args.error_checkpoint,args.device)
    metadata=checkpoint["run_metadata"]
    if metadata.get("stage")!="confidence_guided_group_balanced_posttraining" and not getattr(args,"allow_unposttrained",False):
        raise ValueError("Final calibration requires the selected post-trained model; an explicit ablation may use --allow-unposttrained")
    assert_same_feature_contract(cache_meta,metadata)
    if cache_meta["relative_scale_floor"]!=model.config.relative_scale_floor:
        raise ValueError("Calibration interval scale differs from error-training scale")
    if cache_meta["source_sha256"]!=metadata["source_sha256"]:
        raise ValueError("Feature-construction source changed after error fitting")
    trained_ids=set(metadata["train_mechanism_ids"])|set(metadata["dev_mechanism_ids"])
    calibration_ids={p["mechanism_id"] for p in cache_meta["provenance"]}
    if trained_ids&calibration_ids:
        raise ValueError("Calibration mechanism overlaps uncertainty train/DEV")
    with torch.inference_mode():
        features=torch.from_numpy(arrays["features"]).to(args.device)
        pairs=model(features).cpu().numpy()
    result=conformal_expansions(pairs,arrays["response"],model.config.alphas)
    result.update(format=CALIBRATION_FORMAT,alphas=list(model.config.alphas),
                  error_checkpoint_sha256=sha256(args.error_checkpoint),
                  selected_stage=metadata.get("stage"),
                  factual_checkpoint_sha256=metadata["factual_checkpoint_sha256"],
                  eta_checkpoint_sha256=metadata["eta_checkpoint_sha256"],
                  calibration_cache_sha256=sha256(args.calibration_cache),
                  calibration_metadata_sha256=sha256(Path(args.calibration_cache).with_suffix(".json")),
                  calibration_source_sha256=sha256(__file__),
                  calibration_mechanism_ids=sorted(calibration_ids),
                  relative_scale_floor=model.config.relative_scale_floor,
                  probability_floor=metadata["probability_floor"],outer_folds=metadata["outer_folds"],
                  feature_names=metadata["feature_names"],source_sha256=metadata["source_sha256"],
                  registered_source_groups=metadata["registered_source_groups"],
                  sample_sizes=metadata["sample_sizes"],truth_target=metadata["truth_target"],
                  source_group_counts=cache_meta["source_group_counts"],
                  target_task_mixture=dict(source_groups=cache_meta["registered_source_groups"],
                                           sample_sizes=cache_meta["arguments"]["n_values"],
                                           weights="uniform source group; uniform sample size; iid mechanisms/rows"),
                  interval_rule="[theta_hat-scale*(q_hi+expansion), theta_hat-scale*(q_lo-expansion)]",
                  assumption="Calibration and future tasks exchangeable from the registered ID mixture, conditional on all frozen models",
                  scope="marginal task coverage; no per-source/conditional or arbitrary OOD guarantee",
                  created_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def load_calibrated_error(error_checkpoint,calibration_path,device="cpu"):
    model,metadata=load_selected_error_checkpoint(error_checkpoint,device)
    calibration=json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if calibration.get("format")!=CALIBRATION_FORMAT:
        raise ValueError("Not a v2 task calibration asset")
    if calibration["error_checkpoint_sha256"]!=sha256(error_checkpoint):
        raise ValueError("Calibration belongs to a different uncertainty model")
    if calibration["alphas"]!=list(model.config.alphas):
        raise ValueError("Calibration alpha levels differ")
    assert_same_feature_contract(calibration,metadata["run_metadata"])
    for key in ("factual_checkpoint_sha256","eta_checkpoint_sha256"):
        if calibration[key]!=metadata["run_metadata"][key]:
            raise ValueError("Calibration nuisance checkpoint binding differs")
    return model,metadata,calibration


@torch.inference_mode()
def predict_intervals(result,table,error_model,calibration=None):
    contract=getattr(error_model,"_deployment_contract",None)
    if contract is None:
        raise ValueError("Load a checkpoint-bound uncertainty model before prediction")
    assert_bound_prediction(result,table,contract)
    if error_model.config.relative_scale_floor!=contract["relative_scale_floor"]:
        raise ValueError("Uncertainty model scale floor differs from its construction contract")
    if calibration is not None:
        assert_same_feature_contract(calibration,contract)
        if getattr(error_model,"_checkpoint_sha256",None)!=calibration["error_checkpoint_sha256"]:
            raise ValueError("Prediction calibration belongs to a different uncertainty checkpoint")
    features,scale=uncertainty_features(result,table,error_model.config.relative_scale_floor)
    device=next(error_model.parameters()).device
    pairs=error_model(torch.from_numpy(features[None,:]).to(device)).cpu().numpy()[0]
    intervals={}
    if calibration is not None and calibration["alphas"]!=list(error_model.config.alphas):
        raise ValueError("Prediction and calibration alpha levels differ")
    for i,alpha in enumerate(error_model.config.alphas):
        expansion=0. if calibration is None else calibration["expansions"][i]
        intervals[str(round(100*(1-alpha)))]=dict(
            endpoints=interval_from_pair(result["estimate"],scale,pairs[i],expansion),
            alpha=alpha,signed_error_quantiles=pairs[i].tolist(),expansion=expansion)
    return dict(estimate=result["estimate"],scale=scale,raw_standard_error=result["standard_error"],
                intervals=intervals,calibrated=calibration is not None)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--error-checkpoint",required=True)
    p.add_argument("--calibration-cache",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--allow-unposttrained",action="store_true",help="Explicit pre-post-training ablation only")
    args=p.parse_args()
    result=calibrate(args)
    print(json.dumps(dict(output=args.output,sha256=sha256(args.output),tasks=result["tasks"],
                          alphas=result["alphas"],expansions=result["expansions"])))
