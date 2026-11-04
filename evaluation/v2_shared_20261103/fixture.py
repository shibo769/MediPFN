"""Real GPU execution/binding/parity fixture using additional ID-only tables."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from evaluate_shard import run as evaluate_shard,file_sha,synchronize
from routes import configure_imports,load_route,ASSET_HASHES


def run(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    report=evaluate_shard(argparse.Namespace(bank=args.bank,bank_manifest_sha=args.bank_manifest_sha,
        source_manifest=args.source_manifest,source_manifest_sha=args.source_manifest_sha,models_root=args.models_root,
        output=str(output/"fixture_shard.json"),shard_index=1,shard_size=4,routes=["custom_v2","frozen_v3","adapted_v3"],
        device=args.device,cpu_threads=args.cpu_threads,require_gpu=args.require_gpu,purpose="fixture"))
    if report["attempts"]!=12 or report["failures"]!=0:
        raise RuntimeError("Not every real GPU route/dimension fixture succeeded")
    configure_imports(args.models_root)
    from dgp import sample_task
    from tabular_backbones.train_uncertainty import calibrated_interval
    task=sample_task("he_song_xu",64,2026110397,mechanism_seed=2026110398,compute_truth=False)
    observed={k:task[k] for k in ("x","a","m","y","x_types","mediator_types")}
    parity={}
    for route_name in ("frozen_v3","adapted_v3"):
        route=load_route(route_name,args.models_root,args.device)
        current=route.predict(observed,18031)
        native=calibrated_interval(observed,route.config,route.head_path,route.calibration_path,
                                  factory=route.factory,adapter_checkpoint=route.adapter_path)
        differences=[]
        for key in ("90","95","99"):
            np.testing.assert_allclose(current["intervals"][key],native["intervals"][key],rtol=1e-5,atol=1e-6)
            differences.extend(np.abs(np.asarray(current["intervals"][key])-native["intervals"][key]).tolist())
        parity[route_name]=dict(status="PASS",default_native_fold_seed=18031,
                                max_interval_difference=max(differences))
        del route
        gc.collect()
        if args.device.startswith("cuda"):torch.cuda.empty_cache()
    result=dict(status="PASS_REAL_SHARED_THREE_ROUTE_GPU_FIXTURE",purpose="ID-only execution fixture; not scientific performance",
                fixture_bank_manifest_sha256=args.bank_manifest_sha,evaluation_source_manifest_sha256=args.source_manifest_sha,
                fixture_output_sha256=file_sha(output/"fixture_shard.json"),actual_attempts=report["attempts"],
                failures=report["failures"],hardware=report["hardware"],peak_gpu_allocated_bytes=report["route_peak_gpu_allocated_bytes"],
                native_v3_interval_parity=parity,assets=ASSET_HASHES,
                source_nuisance_calibration_guards="All unchanged model/source/calibration SHA checks and actual shared outer folds verified",
                created_utc=datetime.now(timezone.utc).isoformat())
    (output/"fixture_report.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank",required=True);p.add_argument("--bank-manifest-sha",required=True)
    p.add_argument("--source-manifest",required=True);p.add_argument("--source-manifest-sha",required=True)
    p.add_argument("--models-root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--cpu-threads",type=int,default=8)
    p.add_argument("--require-gpu",default="L40S")
    run(p.parse_args())
