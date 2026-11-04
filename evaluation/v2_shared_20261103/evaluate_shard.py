"""Evaluate all three frozen routes on shared observed tables, without updates."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

PROCESS_STARTED=time.perf_counter()


class EvaluationBindingError(RuntimeError):
    pass


def finite_prediction(prediction,expected_folds,n):
    import numpy as np
    if not np.array_equal(np.asarray(prediction["fold_assignment"],dtype=np.int8),expected_folds):
        raise EvaluationBindingError("Native outer folds differ from the frozen shared table folds")
    for name in ("estimate","standard_error","interval_scale"):
        if not np.isfinite(prediction[name]):
            raise FloatingPointError("Nonfinite prediction: "+name)
    if prediction["standard_error"]<0 or prediction["interval_scale"]<=0:
        raise ValueError("Invalid point/interval scale")
    endpoints=np.asarray([prediction["intervals"][str(level)] for level in (90,95,99)],float)
    if endpoints.shape!=(3,2) or not np.isfinite(endpoints).all() or np.any(endpoints[:,0]>endpoints[:,1]):
        raise ValueError("Nonfinite, missing, or reversed interval endpoints")
    if np.any(np.diff(endpoints[:,0])>1e-7) or np.any(np.diff(endpoints[:,1])< -1e-7):
        raise ValueError("Confidence intervals are not nested")
    if len(expected_folds)!=n:
        raise ValueError("Shared outer-fold denominator differs from observed rows")


def synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8388608),b""):
            h.update(chunk)
    return h.hexdigest()


def source_guard(manifest_path,expected_sha):
    manifest_path=Path(manifest_path)
    if file_sha(manifest_path)!=expected_sha:
        raise ValueError("Evaluation source manifest SHA differs")
    document=json.loads(manifest_path.read_text())
    for name,expected in document["files"].items():
        path=(manifest_path.parent/name).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()) or file_sha(path)!=expected:
            raise ValueError("Evaluation source changed: "+name)
    return document


def run(args):
    import_started=time.perf_counter()
    import numpy as np
    import torch
    from bank import load_manifest,load_case
    from routes import ROUTES,configure_imports,load_route,verify_assets
    import_seconds=time.perf_counter()-import_started
    if args.shard_size<1 or args.shard_index<1:
        raise ValueError("Positive shard index/size required")
    source_guard(args.source_manifest,args.source_manifest_sha)
    bank=load_manifest(args.bank,expected_sha256=args.bank_manifest_sha)
    fixture_bank=bank.get("status")=="FROZEN_ID_ONLY_VALIDATION_FIXTURE"
    if (getattr(args,"purpose","scientific")=="fixture")!=fixture_bank:
        raise ValueError("Fixture/scientific bank purpose differs from the requested evaluation")
    selected=bank["cases"][(args.shard_index-1)*args.shard_size:args.shard_index*args.shard_size]
    if not selected:
        raise ValueError("Shard index has no cases in the frozen bank")
    all_ids=[c["case_id"] for c in bank["cases"]]
    if len(all_ids)!=len(set(all_ids)):
        raise ValueError("Duplicate case identities in the frozen manifest")
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    progress_path=output.with_suffix(".jsonl")
    if output.exists() or progress_path.exists():
        raise FileExistsError("Shard outputs are immutable; choose a new attempt path")
    assets=verify_assets(args.models_root)
    configure_imports(args.models_root)
    torch.set_num_threads(args.cpu_threads)
    device=args.device
    gpu_name=torch.cuda.get_device_name() if str(device).startswith("cuda") else None
    if str(device).startswith("cuda") and args.require_gpu and args.require_gpu.lower() not in gpu_name.lower():
        raise ValueError("Actual GPU differs from registered timing hardware")
    # Bank I/O and truth extraction happen outside all prediction timers. Truth
    # remains here in the evaluator and is never passed into any route.
    loaded=[(case,load_case(args.bank,case,verify=True)) for case in selected]
    route_names=list(ROUTES)
    rotation=(args.shard_index-1)%len(route_names)
    route_names=route_names[rotation:]+route_names[:rotation]
    if args.routes:
        if tuple(args.routes)!=ROUTES:
            raise ValueError("Scientific shards must evaluate all three registered routes")
    hardware=dict(hostname=socket.gethostname(),gpu=gpu_name,cpu_threads=torch.get_num_threads(),
                  torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
                  job_id=os.environ.get("JOB_ID"),task_id=os.environ.get("SGE_TASK_ID"))
    records=[];startups={};route_peak_bytes={}
    process_ready=time.perf_counter()
    with progress_path.open("x",encoding="utf-8") as progress:
        for route_name in route_names:
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            synchronize(device)
            began=time.perf_counter()
            # A loading/import/identity failure is infrastructure failure and
            # aborts the shard; it is not counted as scientific undercoverage.
            route=load_route(route_name,args.models_root,device)
            synchronize(device)
            asset_load_seconds=time.perf_counter()-began
            from dgp import sample_task
            bootstrap=sample_task("he_song_xu",64,2026110397,mechanism_seed=2026110398,compute_truth=False)
            observed={k:bootstrap[k] for k in ("x","a","m","y","x_types","mediator_types")}
            synchronize(device);began=time.perf_counter()
            boot=route.predict(observed,18031)
            synchronize(device)
            bootstrap_seconds=time.perf_counter()-began
            if not all(np.isfinite(v) for pair in boot["intervals"].values() for v in pair):
                raise ValueError("Frozen bootstrap restore failed")
            startups[route_name]=dict(asset_load_seconds=asset_load_seconds,
                                      fixed_bootstrap_seconds=bootstrap_seconds,
                                      initial_model_and_bootstrap_seconds=asset_load_seconds+bootstrap_seconds,
                                      bootstrap_source="fixed non-benchmark he_song_xu observed context, n64",
                                      process_first_prediction_seconds=time.perf_counter()-PROCESS_STARTED)
            if str(device).startswith("cuda"):
                startups[route_name]["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            for case,data in loaded:
                table=data["observed"]
                row=dict(case_id=case["case_id"],route=route_name,case_npz_sha256=case["npz_sha256"],
                         observed_sha256=case["observed_sha256"],fold_seed=case["fold_seed"],n=case["n"],
                         cluster_id=case["cluster_id"],source_group=case["source_group"],
                         source_id=case["source_id"],stratum=case["stratum"],rep=case["rep"])
                synchronize(device);began=time.perf_counter()
                try:
                    prediction=route.predict(table,int(case["fold_seed"]))
                    synchronize(device)
                    elapsed=time.perf_counter()-began
                    finite_prediction(prediction,data["outer_fold_ids"],len(table["a"]))
                    truth=float(data["truth"])
                    row.update(status="success",seconds=elapsed,estimate=float(prediction["estimate"]),
                               standard_error=float(prediction["standard_error"]),interval_scale=float(prediction["interval_scale"]),
                               intervals=prediction["intervals"],truth=truth,
                               error=float(prediction["estimate"])-truth,
                               fold_assignment_sha256=hashlib.sha256(np.asarray(prediction["fold_assignment"],np.int8).tobytes()).hexdigest(),
                               covered={str(level):bool(prediction["intervals"][str(level)][0]<=truth<=prediction["intervals"][str(level)][1]) for level in (90,95,99)},
                               length={str(level):float(np.diff(prediction["intervals"][str(level)])[0]) for level in (90,95,99)},
                               diagnostics=prediction["diagnostics"])
                except Exception as exc:
                    try:synchronize(device)
                    except Exception:pass
                    row.update(status="failure",seconds=time.perf_counter()-began,
                               exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                               truth=float(data["truth"]),estimate=None,standard_error=None,interval_scale=None,
                               intervals=None,error=None,fold_assignment_sha256=None,
                               covered={str(level):False for level in (90,95,99)},length={str(level):None for level in (90,95,99)})
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                    if isinstance(exc,EvaluationBindingError):
                        progress.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n");progress.flush()
                        raise
                records.append(row)
                progress.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n");progress.flush()
                print(json.dumps(dict(route=route_name,case_id=case["case_id"],status=row["status"],seconds=row["seconds"])),flush=True)
            if str(device).startswith("cuda"):
                route_peak_bytes[route_name]=torch.cuda.max_memory_allocated()
            del route
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    expected=len(selected)*3
    if len(records)!=expected or len({(r["route"],r["case_id"]) for r in records})!=expected:
        raise ValueError("Incomplete/duplicate route-case attempts")
    report=dict(format="medipfn-v2-shared-three-route-evaluation-1",status="COMPLETE",
                purpose=getattr(args,"purpose","scientific"),bank_status=bank.get("status"),
                shard_index=args.shard_index,shard_size=args.shard_size,case_ids=[c["case_id"] for c in selected],
                bank_manifest_sha256=args.bank_manifest_sha,evaluation_source_manifest_sha256=args.source_manifest_sha,
                frozen_assets=assets,hardware=hardware,route_order=route_names,
                shared_package_import_seconds=import_seconds,shared_preparation_seconds=process_ready-PROCESS_STARTED,
                route_startup=startups,route_peak_gpu_allocated_bytes=route_peak_bytes,
                timing_scope="Synchronized per observed table: preprocessing/context fits, cross-fitting, score, error head and fixed calibration; excludes bank I/O, truth/metrics, model load and fixed bootstrap",
                cold_start_scope="Per-job shared import/preparation and per-route asset load/fixed non-test bootstrap are recorded separately",
                attempts=expected,successes=sum(r["status"]=="success" for r in records),
                failures=sum(r["status"]!="success" for r in records),records=records,
                elapsed_process_seconds=time.perf_counter()-PROCESS_STARTED,
                progress_sha256=file_sha(progress_path),created_utc=datetime.now(timezone.utc).isoformat())
    temporary=output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(output)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank",required=True)
    p.add_argument("--bank-manifest-sha",required=True)
    p.add_argument("--source-manifest",required=True)
    p.add_argument("--source-manifest-sha",required=True)
    p.add_argument("--models-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--shard-index",type=int,default=int(os.environ.get("SGE_TASK_ID","1")))
    p.add_argument("--shard-size",type=int,default=12)
    p.add_argument("--routes",nargs="+",default=["custom_v2","frozen_v3","adapted_v3"])
    p.add_argument("--device",default="cuda")
    p.add_argument("--cpu-threads",type=int,default=int(os.environ.get("NSLOTS","8")))
    p.add_argument("--require-gpu",default="L40S")
    p.add_argument("--purpose",choices=("scientific","fixture"),default="scientific")
    result=run(p.parse_args())
    print(json.dumps(dict(status=result["status"],attempts=result["attempts"],failures=result["failures"])),flush=True)
