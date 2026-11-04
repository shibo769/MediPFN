"""Real-checkpoint integration fixture for v2 error/post-training/calibration.

All CUDA operations run in this one process, avoiding SCC exclusive-process
conflicts from a CUDA-owning parent launching CUDA children. This fixture does
not provide scientific performance or select a production configuration.
"""
from argparse import ArgumentParser,Namespace
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from calibrate import calibrate,load_calibrated_error,predict_intervals
from error_binding import BoundNIEEstimator
from posttrain_error import fit as fit_post
from train_error import build_cache,fit as fit_error,sha256


def main(args):
    output=Path(args.output)
    if output.exists():
        raise FileExistsError("Integration fixture output must be a new directory")
    output.mkdir(parents=True)
    for split in ("train","dev","post_train","post_dev"):
        count=args.cases
        build_cache(Namespace(checkpoint=args.checkpoint,eta_checkpoint=args.eta_checkpoint,
                              split=split,cases=count,output=str(output/(split+".npz")),device=args.device,
                              seed=2026110341,n_values=[64],relative_scale_floor=1e-3,
                              probability_floor=.01,log_every=count,cpu_threads=args.cpu_threads))
    base=fit_error(Namespace(train_cache=str(output/"train.npz"),dev_cache=str(output/"dev.npz"),
                             output=str(output/"error_fit"),device=args.device,seed=2026110342,
                             epochs=2,patience=2,batch_size=16,width=16,lr=1e-3,cpu_threads=args.cpu_threads))
    post=fit_post(Namespace(error_checkpoint=str(base),train_cache=str(output/"post_train.npz"),
                            dev_cache=str(output/"post_dev.npz"),output=str(output/"post_fit"),
                            device=args.device,seed=2026110343,confidence_epochs=2,correction_epochs=2,
                            confidence_width=8,correction_width=8,lr=1e-3,output_penalty=1e-4))
    build_cache(Namespace(checkpoint=args.checkpoint,eta_checkpoint=args.eta_checkpoint,
                          split="calibration",cases=args.calibration_cases,output=str(output/"calibration.npz"),
                          device=args.device,seed=2026110341,n_values=[64],relative_scale_floor=1e-3,
                          probability_floor=.01,log_every=args.calibration_cases,cpu_threads=args.cpu_threads))
    calibration_path=output/"final_calibration.json"
    calibration=calibrate(Namespace(error_checkpoint=str(post),calibration_cache=str(output/"calibration.npz"),
                                    output=str(calibration_path),device=args.device,allow_unposttrained=False))
    error_model,_,calibration=load_calibrated_error(post,calibration_path,args.device)
    estimator=BoundNIEEstimator(args.checkpoint,args.eta_checkpoint,args.device)
    from dgp import sample_task
    task=sample_task("he_song_xu",64,2026110344,mechanism_seed=2026110345)
    estimate=estimator.estimate(task,seed=2026110346,return_scores=True)
    predicted=predict_intervals(estimate,task,error_model,calibration)
    endpoints=np.asarray([predicted["intervals"][str(level)]["endpoints"] for level in (90,95,99)])
    if not np.isfinite(endpoints).all() or not np.all(endpoints[:,0]<endpoints[:,1]):
        raise AssertionError("Nonfinite or reversed learned/calibrated endpoints")
    if not np.all(np.diff(endpoints[:,0])<=0) or not np.all(np.diff(endpoints[:,1])>=0):
        raise AssertionError("Confidence levels are not nested after post-training/calibration")
    paths=list(output.glob("*.npz"))+list(output.glob("*.json"))+[
        output/"error_fit"/"best.pt",output/"post_fit"/"best.pt"]
    report=dict(status="PASS",fixture_only=True,not_scientific_performance=True,
                factual_checkpoint_sha256=sha256(args.checkpoint),eta_checkpoint_sha256=sha256(args.eta_checkpoint),
                assets={p.relative_to(output).as_posix():sha256(p) for p in paths},
                trained_error_cases=args.cases,post_training_cases=args.cases,calibration_cases=args.calibration_cases,
                observed_prediction=predicted,created_utc=datetime.now(timezone.utc).isoformat())
    (output/"smoke_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="PASS",report=str(output/"smoke_report.json"),sha256=sha256(output/"smoke_report.json"))),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--eta-checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--cases",type=int,default=24)
    p.add_argument("--calibration-cases",type=int,default=104)
    p.add_argument("--cpu-threads",type=int,default=8)
    main(p.parse_args())
