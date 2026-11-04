"""Production signed-error learning, post-training and fresh task calibration.

One CUDA process owns this sequential workflow. No ID/OOD test bank is read.
"""
from argparse import ArgumentParser,Namespace
from pathlib import Path
import json,os,time
from train_error import build_cache,fit as fit_error,sha256
from posttrain_error import fit as fit_post
from calibrate import calibrate,load_calibrated_error,predict_intervals
from error_binding import BoundNIEEstimator

def run(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    counts={'train':10000,'dev':1024,'post_train':1024,'post_dev':512,'calibration':1024}
    config=dict(checkpoint=args.checkpoint,eta_checkpoint=args.eta_checkpoint,device='cuda',seed=2026110321,
                n_values=[200,600,1000],relative_scale_floor=.001,probability_floor=.01,
                log_every=100,cpu_threads=args.cpu_threads)
    (out/'workflow_config.json').write_text(json.dumps(dict(counts=counts,cache=config,selection='DEV only; no scientific test data read'),indent=2)+'\n')
    for split in ('train','dev'):
        build_cache(Namespace(**config,split=split,cases=counts[split],output=str(out/(split+'.npz'))))
    error=fit_error(Namespace(train_cache=str(out/'train.npz'),dev_cache=str(out/'dev.npz'),output=str(out/'error_fit'),
                             device='cuda',seed=2026110322,epochs=100,patience=15,batch_size=256,width=128,lr=.001,cpu_threads=args.cpu_threads))
    for split in ('post_train','post_dev'):
        build_cache(Namespace(**config,split=split,cases=counts[split],output=str(out/(split+'.npz'))))
    post=fit_post(Namespace(error_checkpoint=str(error),train_cache=str(out/'post_train.npz'),dev_cache=str(out/'post_dev.npz'),
                            output=str(out/'post_fit'),device='cuda',seed=2026110331,confidence_epochs=100,correction_epochs=160,
                            confidence_width=32,correction_width=16,lr=.001,output_penalty=.0001))
    split='calibration'
    build_cache(Namespace(**config,split=split,cases=counts[split],output=str(out/(split+'.npz'))))
    calibration_path=out/'calibration.json'
    calibrate(Namespace(error_checkpoint=str(post),calibration_cache=str(out/'calibration.npz'),output=str(calibration_path),
                        device='cuda',allow_unposttrained=False))
    model,_,cal=load_calibrated_error(post,calibration_path,'cuda')
    estimator=BoundNIEEstimator(args.checkpoint,args.eta_checkpoint,'cuda')
    from dgp import sample_task
    table=sample_task('he_song_xu',200,2026110397,mechanism_seed=2026110398,compute_truth=False)
    prediction=predict_intervals(estimator.estimate(table,seed=2026110399),table,model,cal)
    report=dict(status='TRAINED_AND_POSTTRAINED_AND_FRESH_ID_TASK_CALIBRATION_COMPLETE',job_id=os.environ.get('JOB_ID'),
                seconds=time.perf_counter()-start,counts=counts,
                factual_checkpoint_sha256=sha256(args.checkpoint),eta_checkpoint_sha256=sha256(args.eta_checkpoint),
                error_checkpoint_sha256=sha256(error),post_checkpoint_sha256=sha256(post),calibration_sha256=sha256(calibration_path),
                inference_restore_fixture=prediction,scientific_test_evaluation='not performed',
                scope='marginal ID task-mixture calibration; source-OOD is a separate frozen evaluation')
    (out/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--eta-checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cpu-threads',type=int,default=int(os.environ.get('NSLOTS','8')))
    run(p.parse_args())
