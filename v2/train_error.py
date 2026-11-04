"""Cache frozen-model signed errors, then fit v2 uncertainty quantiles.

Usage has two phases so expensive cross-fitted feature extraction is performed
once per table. Training/DEV truth is a response only. Calibration caches must
not be passed to fitting; OOD sources are rejected from every fitting cache.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from error_model import (DEFAULT_ALPHAS, SUMMARY_FEATURES, ErrorConfig,
                         SignedErrorQuantiles,pinball_loss,save_error_checkpoint,uncertainty_features)
from train import source_bindings
from error_binding import (BoundNIEEstimator,assert_same_feature_contract,
                           assert_bound_prediction,validate_feature_metadata)

SPLIT_CODES={"train":10,"dev":11,"calibration":12,"post_train":13,"post_dev":14}
CACHE_FORMAT="medipfn-v2-observed-error-features-1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_seed(seed,split,index,kind):
    """New >160-bit namespaces cannot equal the factual <=65-bit seed values."""
    if split not in SPLIT_CODES:
        raise ValueError("Only independent train/DEV/calibration feature caches are supported")
    digest=hashlib.sha256((str(seed)+":"+split+":"+str(index)+":"+kind).encode()).digest()[:16]
    return (SPLIT_CODES[split]<<160)|int.from_bytes(digest,"big")


def error_bindings():
    root=Path(__file__).parent
    bindings=source_bindings()
    for name in ("evaluate.py","error_model.py","error_binding.py","train_error.py"):
        bindings[name]=sha256(root/name)
    return bindings


def cache_paths(path):
    path=Path(path)
    if path.suffix != ".npz":
        raise ValueError("Feature cache filename must end in .npz")
    return path,path.with_suffix(".json")


def build_cache(args):
    from dgp import registry,sample_task
    path,metadata_path=cache_paths(args.output)
    if path.exists() or metadata_path.exists() or path.with_suffix(".failure.json").exists():
        raise FileExistsError("Feature cache assets are immutable; choose a new output path")
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.cpu_threads or int(os.environ.get("NSLOTS",os.cpu_count() or 1)))
    estimator=BoundNIEEstimator(args.checkpoint,args.eta_checkpoint,args.device,
                                probability_floor=args.probability_floor,outer_folds=2)
    model=estimator.factual
    teacher_meta=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    forbidden_ids=set(teacher_meta.get("accounting",{}).get("mechanism_ids",[]))
    forbidden_ids.update(p["mechanism_id"] for p in teacher_meta.get("run_metadata",{}).get("dev_provenance",[]))
    eta_meta=torch.load(args.eta_checkpoint,map_location="cpu",weights_only=False)
    forbidden_ids.update(p["mechanism_id"] for p in eta_meta.get("run_metadata",{}).get("dev_provenance",[]))
    records=[r for r in registry() if r["default_split"]=="train"]
    groups=sorted(set(r["source_group"] for r in records))
    feature_names=["context_"+str(i) for i in range(model.config.width)]+list(SUMMARY_FEATURES)
    metadata=dict(format=CACHE_FORMAT,split=args.split,arguments=vars(args),
                  factual_checkpoint_sha256=sha256(args.checkpoint),eta_checkpoint_sha256=sha256(args.eta_checkpoint),
                  source_sha256=error_bindings(),feature_names=feature_names,
                  relative_scale_floor=args.relative_scale_floor,probability_floor=args.probability_floor,
                  outer_folds=2,truth_target="superpopulation_NIE_1",
                  registered_source_groups=groups,
                  sample_sizes=list(args.n_values),
                  draw_design="iid uniform source-group and sample-size mixture" if args.split=="calibration" else
                              "balanced source groups/sample sizes with independently seeded mechanisms",
                  created_utc=datetime.now(timezone.utc).isoformat())
    feature_rows,response_rows,estimate_rows,scale_rows,truth_rows,provenance=[],[],[],[],[],[]
    counts=Counter()
    started=time.perf_counter()
    for index in range(args.cases):
        case_rng=np.random.default_rng(task_seed(args.seed,args.split,index,"selection"))
        if args.split=="calibration":
            # Exchangeable calibration tasks are drawn iid from this registered mixture.
            group=str(case_rng.choice(groups))
            n=int(case_rng.choice(args.n_values))
        else:
            group=groups[index%len(groups)]
            n=int(args.n_values[(index//len(groups))%len(args.n_values)])
        candidates=[r["source_id"] for r in records if r["source_group"]==group]
        source=str(case_rng.choice(candidates))
        mechanism_seed=task_seed(args.seed,args.split,index,"mechanism")
        observation_seed=task_seed(args.seed,args.split,index,"observations")
        try:
            table=sample_task(source,n,observation_seed,mechanism_seed=mechanism_seed,compute_truth=True)
            if table["metadata"]["mechanism_id"] in forbidden_ids:
                raise ValueError("Error-stage mechanism overlaps factual training")
            result=estimator.estimate(table,seed=task_seed(args.seed,args.split,index,"folds"),return_scores=True)
            assert_bound_prediction(result,table,metadata)
            features,scale=uncertainty_features(result,table,args.relative_scale_floor)
            target=(float(result["estimate"])-float(table["truth"]))/scale
            if not math.isfinite(target):
                raise FloatingPointError("Nonfinite standardized signed-error response")
        except Exception as exc:
            failure=dict(index=index,source_id=source,source_group=group,n=n,
                         mechanism_seed=mechanism_seed,observation_seed=observation_seed,
                         exception=type(exc).__name__,message=str(exc),completed_cases=len(feature_rows))
            failure_path=path.with_suffix(".failure.json")
            failure_path.write_text(json.dumps(dict(metadata=metadata,failure=failure),indent=2)+"\n",encoding="utf-8")
            raise RuntimeError("Feature generation failed; details retained at "+str(failure_path)) from exc
        feature_rows.append(features)
        response_rows.append(target)
        estimate_rows.append(result["estimate"])
        scale_rows.append(scale)
        truth_rows.append(table["truth"])
        provenance.append(dict(table["metadata"],n=n,fold_seed=task_seed(args.seed,args.split,index,"folds")))
        counts[group]+=1
        if (index+1)%args.log_every==0 or index+1==args.cases:
            print(json.dumps(dict(phase="error_features",split=args.split,completed=index+1,total=args.cases,
                                  seconds=time.perf_counter()-started,source_group_counts=dict(counts))),flush=True)
    arrays=dict(features=np.stack(feature_rows).astype(np.float32),response=np.asarray(response_rows,np.float64),
                estimate=np.asarray(estimate_rows,np.float64),scale=np.asarray(scale_rows,np.float64),
                truth=np.asarray(truth_rows,np.float64))
    temporary=path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle,**arrays)
    temporary.replace(path)
    metadata.update(cases=args.cases,completed_cases=len(feature_rows),failures=0,
                    source_group_counts=dict(counts),seconds=time.perf_counter()-started,
                    cache_sha256=sha256(path),provenance=provenance)
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    return path


def read_cache(path,expected_split=None):
    path,metadata_path=cache_paths(path)
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format")!=CACHE_FORMAT or metadata.get("cache_sha256")!=sha256(path):
        raise ValueError("Feature cache format/hash mismatch")
    if expected_split is not None and metadata["split"]!=expected_split:
        raise ValueError("Wrong data split: expected "+expected_split)
    validate_feature_metadata(metadata)
    if metadata["source_sha256"]!=error_bindings():
        raise ValueError("Feature cache source/config bindings differ from the active frozen release")
    if metadata.get("split") not in SPLIT_CODES:
        raise ValueError("Unknown feature cache split")
    cases=metadata.get("cases")
    if type(cases) is not int or cases<1 or metadata.get("completed_cases")!=cases:
        raise ValueError("Incomplete feature cache case denominator")
    if len(metadata.get("provenance",[]))!=cases:
        raise ValueError("Feature cache provenance denominator differs")
    arguments=metadata.get("arguments",{})
    for key in ("split","cases","probability_floor","relative_scale_floor"):
        if arguments.get(key)!=metadata[key]:
            raise ValueError("Feature cache arguments disagree with declared setting: "+key)
    if arguments.get("n_values")!=metadata["sample_sizes"]:
        raise ValueError("Feature cache sample-size settings disagree")
    from dgp import registry,SOURCE_COMMIT
    records={r["source_id"]:r for r in registry() if r["default_split"]=="train"}
    groups=sorted(set(r["source_group"] for r in records.values()))
    if metadata["registered_source_groups"]!=groups:
        raise ValueError("Feature cache source mixture differs from registered train groups")
    actual_counts=Counter()
    for index,p in enumerate(metadata["provenance"]):
        source=p.get("source_id")
        if source not in records or p.get("source_group")!=records[source]["source_group"]:
            raise ValueError("Feature cache contains an OOD or misidentified source")
        if p.get("registry_id")!=records[source]["registry_id"] or p.get("source_commit")!=SOURCE_COMMIT:
            raise ValueError("Feature cache source provenance differs from the frozen registry")
        if p.get("n") not in metadata["sample_sizes"] or p.get("target")!=metadata["truth_target"]:
            raise ValueError("Feature cache task size/estimand differs from registration")
        config=json.dumps(p.get("config"),sort_keys=True,separators=(",",":"))
        mechanism_id=hashlib.sha256((source+":"+config).encode()).hexdigest()
        if mechanism_id!=p.get("mechanism_id"):
            raise ValueError("Feature cache mechanism parameter hash differs")
        for key,kind in (("mechanism_seed","mechanism"),("table_seed","observations"),("fold_seed","folds")):
            if p.get(key)!=task_seed(arguments["seed"],metadata["split"],index,kind):
                raise ValueError("Feature cache stage seed binding differs: "+key)
        actual_counts[p["source_group"]]+=1
    if dict(actual_counts)!=metadata.get("source_group_counts"):
        raise ValueError("Feature cache group counts do not match its provenance")
    with np.load(path,allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    if metadata.get("failures")!=0 or len(arrays["features"])!=metadata["cases"]:
        raise ValueError("Incomplete feature cache cannot fit uncertainty model")
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError("Nonfinite feature cache")
    if arrays["features"].shape!=(metadata["cases"],len(metadata["feature_names"])):
        raise ValueError("Feature cache dimensions differ from declared feature names")
    if any(arrays[k].shape!=(metadata["cases"],) for k in ("response","estimate","scale","truth")):
        raise ValueError("Feature response/endpoint denominators differ")
    if not np.all(arrays["scale"]>0):
        raise ValueError("Feature cache contains a nonpositive uncertainty scale")
    recomputed=(arrays["estimate"]-arrays["truth"])/arrays["scale"]
    if not np.allclose(arrays["response"],recomputed,rtol=1e-12,atol=1e-12):
        raise ValueError("Feature cache signed-error response is inconsistent with estimate/truth/scale")
    return arrays,metadata


def fit(args):
    train,train_meta=read_cache(args.train_cache,"train")
    dev,dev_meta=read_cache(args.dev_cache,"dev")
    assert_same_feature_contract(train_meta,dev_meta)
    train_ids={p["mechanism_id"] for p in train_meta["provenance"]}
    dev_ids={p["mechanism_id"] for p in dev_meta["provenance"]}
    if train_ids&dev_ids:
        raise ValueError("Error-stage train and DEV mechanisms overlap")
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"best.pt").exists():
        raise FileExistsError("An uncertainty checkpoint already exists; choose a new run directory")
    torch.set_num_threads(args.cpu_threads or int(os.environ.get("NSLOTS",os.cpu_count() or 1)))
    torch.manual_seed(args.seed)
    config=ErrorConfig(feature_count=train["features"].shape[1],width=args.width,
                       relative_scale_floor=train_meta["relative_scale_floor"])
    model=SignedErrorQuantiles(config).to(args.device)
    features=torch.from_numpy(train["features"]).to(args.device)
    response=torch.from_numpy(train["response"]).to(args.device,dtype=torch.float32)
    dev_features=torch.from_numpy(dev["features"]).to(args.device)
    dev_response=torch.from_numpy(dev["response"]).to(args.device,dtype=torch.float32)
    model.set_normalization(features)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    metadata=dict(arguments=vars(args),stage="signed_error_quantile_training",source_sha256=error_bindings(),
                  factual_checkpoint_sha256=train_meta["factual_checkpoint_sha256"],
                  eta_checkpoint_sha256=train_meta["eta_checkpoint_sha256"],
                  train_cache_sha256=sha256(args.train_cache),dev_cache_sha256=sha256(args.dev_cache),
                  train_cache_metadata_sha256=sha256(Path(args.train_cache).with_suffix(".json")),
                  dev_cache_metadata_sha256=sha256(Path(args.dev_cache).with_suffix(".json")),
                  probability_floor=train_meta["probability_floor"],outer_folds=train_meta["outer_folds"],
                  relative_scale_floor=train_meta["relative_scale_floor"],
                  registered_source_groups=train_meta["registered_source_groups"],
                  sample_sizes=train_meta["sample_sizes"],truth_target=train_meta["truth_target"],
                  feature_names=train_meta["feature_names"],train_cases=len(features),dev_cases=len(dev_features),
                  train_mechanism_ids=sorted(train_ids),dev_mechanism_ids=sorted(dev_ids),
                  objective="average pinball risk for signed-error quantiles at registered alpha/2 and 1-alpha/2",
                  selection="independent same-source DEV pinball risk; calibration and OOD unused",
                  post_training="none beyond quantile fitting; any later post-training requires fresh calibration",
                  created_utc=datetime.now(timezone.utc).isoformat())
    (output/"run_config.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    best,best_epoch,history,started=float("inf"),0,[],time.perf_counter()
    for epoch in range(args.epochs+1):
        if epoch:
            model.train()
            for index in torch.randperm(len(features),device=args.device).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss=pinball_loss(model(features[index]),response[index],model.config.alphas)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite signed-error quantile loss")
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite uncertainty gradient")
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_loss=float(pinball_loss(model(dev_features),dev_response,model.config.alphas))
        entry=dict(epoch=epoch,dev_pinball_loss=dev_loss,seconds=time.perf_counter()-started)
        history.append(entry)
        if dev_loss<best:
            best,best_epoch=dev_loss,epoch
            save_error_checkpoint(output/"best.pt",model,run_metadata=metadata,selected_epoch=epoch,selected_dev_loss=best)
        save_error_checkpoint(output/"last.pt",model,run_metadata=metadata,epoch=epoch,
                              optimizer_state=optimizer.state_dict(),history=history,best_epoch=best_epoch,best_loss=best)
        print(json.dumps(dict(entry,best_loss=best,best_epoch=best_epoch)),flush=True)
        with (output/"training.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(entry)+"\n")
        if epoch-best_epoch>=args.patience:
            break
    return output/"best.pt"


def parser():
    root=argparse.ArgumentParser(description=__doc__)
    sub=root.add_subparsers(dest="phase",required=True)
    cache=sub.add_parser("cache")
    cache.add_argument("--checkpoint",required=True)
    cache.add_argument("--eta-checkpoint",required=True)
    cache.add_argument("--split",choices=tuple(SPLIT_CODES),required=True)
    cache.add_argument("--cases",type=int,required=True)
    cache.add_argument("--output",required=True)
    cache.add_argument("--device",default="cuda")
    cache.add_argument("--seed",type=int,default=2026110321)
    cache.add_argument("--n-values",nargs="+",type=int,default=[200,600,1000])
    cache.add_argument("--relative-scale-floor",type=float,default=1e-3)
    cache.add_argument("--probability-floor",type=float,default=.01)
    cache.add_argument("--log-every",type=int,default=50)
    cache.add_argument("--cpu-threads",type=int)
    fitting=sub.add_parser("fit")
    fitting.add_argument("--train-cache",required=True)
    fitting.add_argument("--dev-cache",required=True)
    fitting.add_argument("--output",required=True)
    fitting.add_argument("--device",default="cuda")
    fitting.add_argument("--seed",type=int,default=2026110322)
    fitting.add_argument("--epochs",type=int,default=100)
    fitting.add_argument("--patience",type=int,default=15)
    fitting.add_argument("--batch-size",type=int,default=256)
    fitting.add_argument("--width",type=int,default=128)
    fitting.add_argument("--lr",type=float,default=1e-3)
    fitting.add_argument("--cpu-threads",type=int)
    return root


if __name__=="__main__":
    args=parser().parse_args()
    if args.phase=="cache":
        if args.cases<1 or min(args.n_values)<16 or args.log_every<1:
            raise ValueError("Invalid feature cache budget")
        build_cache(args)
    else:
        if args.epochs<1 or args.batch_size<1 or args.patience<1:
            raise ValueError("Invalid signed-error fit budget")
        fit(args)
