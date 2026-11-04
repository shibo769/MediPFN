"""Independent loaded-model single-table timing, never cached predictions.

Each PFN arm recomputes the prescribed three support views, full-n score and
observable features before its CI callback. Calibration/checkpoint loading,
raw-file I/O, synthetic generation and oracle labels are outside the timer.
The first 13 tables in each predeclared test-linear n/p cell supply 3 warmups
and 10 measured repetitions, with cyclically rotated method order.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from statistics import NormalDist
import argparse,hashlib,json,math,os,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from model import load_checkpoint
from teacher_model import load_teacher
from inference import crossfit_views
from baselines import crossfit_nuisance
from evaluate import EvaluationGuard,canonical_score,interval_stats,failed_stats
import posttrain as pt
import posttrain_heuristics as heuristic
import apply_posttrain as core_apply
from merge_cache import sha,read

CORE_ARMS=core_apply.ARMS
HEURISTIC_ARMS=heuristic.APPLICATION_ARMS
CLASSICAL_ARMS=('linear_dml','gbm_dml','rf_dml')
TEACHER_ARMS=('teacher_latent_theta','teacher_FSP_oracle_score')

def clean(value):return pt.clean(value)
def atomic_json(path,value):
    from build_cache import atomic_json as write
    write(path,value)
def source_hashes():
    folder=Path(__file__).resolve().parent
    names=('run_timing.py','model.py','inference.py','posttrain.py','posttrain_heuristics.py','apply_posttrain.py','merge_cache.py','evaluate.py','baselines.py','teacher_model.py')
    return {name:sha(folder/name) for name in names}

def require_bound(gate,key,value,model_id):
    """Scalar and per-model authorization fields share the evaluation schema."""
    def flatten(raw):
        if isinstance(raw,str):return [raw]
        if isinstance(raw,dict):return [item for v in raw.values() for item in flatten(v)]
        if isinstance(raw,(list,tuple)):return [item for v in raw for item in flatten(v)]
        return []
    declared=gate.get(key)
    allowed=flatten(declared.get(model_id)) if isinstance(declared,dict) and model_id in declared else flatten(declared)
    models=gate.get('models',{})
    candidates=[models[model_id]] if model_id in models else list(models.values())
    for entry in candidates:
        if isinstance(entry,dict) and key in entry:allowed.extend(flatten(entry[key]))
    if value not in allowed:raise ValueError('Timing release must bind '+key+' for '+model_id)

def eligible_groups(protocol,manifest):
    settings=protocol['timing'];ns=set(map(int,settings['n']));ps=set(map(int,settings['p']))
    groups=[g for g in manifest['groups'] if g['split']=='test' and g['bank']=='synthetic' and g.get('population_kind','synthetic')=='synthetic' and g['family']=='linear' and int(g['n']) in ns and int(g['p']) in ps]
    pairs=[(int(g['n']),int(g['p'])) for g in groups]
    if len(pairs)!=len(set(pairs)) or set(pairs)!={(n,p) for n in ns for p in ps}:raise ValueError('Timing manifest must contain exactly one prespecified linear cell per n/p pair')
    return sorted(groups,key=lambda g:(int(g['p']),int(g['n'])))

def load_bundle(args,gate):
    protocol_hash=sha(args.protocol);checkpoint_hash=sha(args.checkpoint)
    require_bound(gate,'checkpoint_sha256',checkpoint_hash,args.model_id)
    core_art,blocked_ids,blocked_seeds,identity=core_apply.verify_posttrain(args.posttrain_dir)
    if identity!={'checkpoint_sha256':checkpoint_hash,'protocol_sha256':protocol_hash}:raise ValueError('Core bundle belongs to another backbone/protocol')
    require_bound(gate,'posttrain_artifacts_sha256',sha(Path(args.posttrain_dir)/'artifacts.json'),args.model_id)
    inputs=dict(checkpoint_sha256=checkpoint_hash,core_artifacts_sha256=sha(Path(args.posttrain_dir)/'artifacts.json'))
    heads=pt.load_heads(args.posttrain_dir,'cpu');policies={name:pt.load_policy(Path(args.posttrain_dir)/(file+'.pt'),'cpu') for name,file in core_apply.POLICIES.items()}
    with np.load(Path(args.posttrain_dir)/'normalization.npz',allow_pickle=False) as z:norm={k:(z[k+'_center'],z[k+'_scale']) for k in ('x_single','x_multi')}
    hheads={};hpolicies={};pool={}
    if args.heuristics_dir:
        _,ids,seeds=heuristic.verify_candidate(args.heuristics_dir,args.posttrain_dir);blocked_ids|=ids;blocked_seeds|=seeds
        folder=Path(args.heuristics_dir);require_bound(gate,'heuristics_artifacts_sha256',sha(folder/'artifacts.json'),args.model_id)
        hheads={name:heuristic.load_head(folder/(name+'.pt'),'cpu') for name in heuristic.HEAD_ARMS}
        hpolicies={name:heuristic.load_expanded_policy(folder/(name+'.pt'),'cpu') for name in heuristic.EXPANDED_ARMS}
        pool=read(folder/'expanded_pool.json')['selection'];inputs['heuristics_artifacts_sha256']=sha(folder/'artifacts.json')
    calibration,binding=core_apply.calibration_arrays(args.calibration_cache,protocol_hash,checkpoint_hash)
    calibrators=core_apply.fit_calibrators(calibration,blocked_ids,blocked_seeds);inputs['calibration_binding']=binding
    teacher_models={};teacher_inputs={}
    for name,path in ((TEACHER_ARMS[0],args.teacher_latent),(TEACHER_ARMS[1],args.teacher_fsp)):
        if path:
            digest=sha(path)
            # Teacher hashes may be a mode->hash mapping or live in models.
            require_bound(gate,'teacher_checkpoint_sha256',digest,args.model_id)
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            expected='latent_theta' if name==TEACHER_ARMS[0] else 'FSP_oracle_score'
            if checkpoint.get('mode')!=expected:raise ValueError('Teacher mode/checkpoint mismatch')
            for source,digest_source in checkpoint.get('training_metadata',{}).get('source_hashes',{}).items():
                if sha(Path(__file__).with_name(source))!=digest_source:raise ValueError('Teacher source drift '+source)
            teacher_models[name]=load_teacher(path,args.device);teacher_inputs[name]=dict(path=str(Path(path).resolve()),sha256=digest)
    inputs['teachers']=teacher_inputs
    backbone=load_checkpoint(args.checkpoint,args.device)
    for module in [backbone,*heads,*policies.values(),*hheads.values(),*hpolicies.values(),*teacher_models.values()]:
        module.eval()
        for parameter in module.parameters():parameter.requires_grad_(False)
    if not args.allow_partial and (not args.heuristics_dir or set(teacher_models)!=set(TEACHER_ARMS)):raise ValueError('Full timing requires core13, heuristic20 and both teacher checkpoints; --allow-partial is explicitly diagnostic')
    return SimpleNamespace(backbone=backbone,heads=heads,policies=policies,norm=norm,calibrators=calibrators,heuristic_heads=hheads,
        heuristic_policies=hpolicies,pool_selection=pool,teachers=teacher_models,blocked_ids=blocked_ids,blocked_seeds=blocked_seeds,inputs=inputs)

def stats_for(summary,alpha,endpoints=None,point=None):
    row=interval_stats(dict(estimate=summary['estimate'],se=summary['se'],clipping_rate=summary['clipping_rate'],failed=False),alpha,endpoints=endpoints)
    if point is not None and not row[5]:
        if not np.isfinite(point):return failed_stats()
        row[0]=point
    return row

@torch.inference_mode()
def core_ci(method,result,bundle,alpha):
    summaries=(result['summaries'][0],result['summaries'][-1]);est=result['estimates'][None];se=result['ses'][None]
    xs=result['x_single'][None];xm=result['x_multi'][None]
    if method in ('single_wald','multi_wald'):
        point=0 if method=='single_wald' else 1
        return stats_for(summaries[point],alpha)
    if method in ('empirical_single','empirical_multi'):
        point=0 if method=='empirical_single' else 1;qlo,qhi=bundle.calibrators[point].quantiles(alpha)
        endpoints=(est[0,point]-se[0,point]*qhi,est[0,point]-se[0,point]*qlo)
        return stats_for(summaries[point],alpha,endpoints)
    if method in ('generic_error','mediation_error'):
        point=0 if method=='generic_error' else 1;key='x_single' if point==0 else 'x_multi';values=xs if point==0 else xm
        x=pt.normalized({key:values},key,bundle.norm,'cpu');density=bundle.heads[point](x)
        qlo=float(pt.quantile(density,alpha/2)[0]);qhi=float(pt.quantile(density,1-alpha/2)[0])
        return stats_for(summaries[point],alpha,(est[0,point]-se[0,point]*qhi,est[0,point]-se[0,point]*qlo))
    prediction=pt.predict(bundle.heads,bundle.policies[method],bundle.norm,xs,xm,est,se,alpha,'cpu');point=int(prediction['point_index'][0])
    return stats_for(summaries[point],alpha,prediction['intervals'][0])

@torch.inference_mode()
def heuristic_ci(method,result,bundle,alpha):
    summaries=(result['summaries'][0],result['summaries'][-1]);est=result['estimates'][None];se=result['ses'][None]
    xs=result['x_single'][None];xm=result['x_multi'][None]
    if method in heuristic.CONTROL_ARMS:
        point=0 if method==heuristic.CONTROL_ARMS[0] else 1;key='x_single' if point==0 else 'x_multi';values=xs if point==0 else xm
        x=pt.normalized({key:values},key,bundle.norm,'cpu');density=bundle.heads[point](x)
        qlo=float(pt.quantile(density,alpha/2)[0]);qhi=float(pt.quantile(density,1-alpha/2)[0]);w,m,_=density
        mean=float((w.double()*m.double()).sum(-1)[0]);corrected=est[0,point]-se[0,point]*mean
        return stats_for(summaries[point],alpha,(est[0,point]-se[0,point]*qhi,est[0,point]-se[0,point]*qlo),corrected)
    if method in heuristic.HEAD_ARMS:
        prediction=heuristic.predict_error_head(bundle.heuristic_heads[method],bundle.norm,xm,est[:,1],se[:,1],alpha,'cpu')
        return stats_for(summaries[1],alpha,prediction['intervals'][0],prediction['estimate'][0])
    extra=[bundle.heuristic_heads[bundle.pool_selection[name]] for name in ('moment','consistency')]
    prediction=heuristic.predict_expanded(bundle.heuristic_policies[method],bundle.heads,extra,bundle.norm,xs,xm,est,se,alpha,'cpu');point=int(prediction['point_index'][0])
    return stats_for(summaries[point],alpha,prediction['intervals'][0],prediction['estimate'][0])

def make_callbacks(bundle,args):
    callbacks={}
    for method in CORE_ARMS+(HEURISTIC_ARMS if bundle.heuristic_heads else ()):
        def callback(table,fold_ids,seed,method=method):
            # This call is intentionally INSIDE each method's timer. There is
            # no cross-method reuse of nuisances, contexts, features or CIs.
            rows=np.column_stack((table['x'],table['a'],table['m'],table['y'])).astype(np.float64)
            result=crossfit_views(bundle.backbone,rows,fold_ids,seed=int(seed),context_size=args.context_size,device=args.device,query_batch=args.query_batch)
            row=core_ci(method,result,bundle,args.alpha) if method in CORE_ARMS else heuristic_ci(method,result,bundle,args.alpha)
            return dict(stats=row,error=None)
        callbacks[method]=callback
    for method,kind in zip(CLASSICAL_ARMS,('linear','gbm','rf')):
        def callback(table,fold_ids,seed,kind=kind):
            result=crossfit_nuisance(table,fold_ids,kind=kind,seed=int(seed)%(2**31-1),n_jobs=args.estimator_jobs)
            summary=canonical_score(table,result['nuisance']);row=interval_stats(summary,args.alpha)
            return dict(stats=row,error=result['error'] or summary.get('error'))
        callbacks[method]=callback
    for method,model in bundle.teachers.items():
        @torch.inference_mode()
        def callback(table,fold_ids,seed,model=model):
            rows=np.column_stack((table['x'],table['a'],table['m'],table['y'])).astype(np.float32)
            prediction=model(torch.as_tensor(rows[None],device=args.device));estimate=float(prediction['estimate'][0].cpu());logV=float(prediction['log_variance_coefficient'][0].cpu())
            se=math.exp(logV/2)/math.sqrt(len(rows));z=NormalDist().inv_cdf(1-args.alpha/2)
            row=np.array([estimate,se,estimate-z*se,estimate+z*se,0.,0.])
            if not np.isfinite(row).all() or not se>0:row=failed_stats()
            return dict(stats=row,error=None)
        callbacks[method]=callback
    return callbacks

def benchmark(tables,fold_ids,seeds,callbacks,warmups=3,repeats=10,synchronize=None):
    if len(tables)<warmups+repeats:raise ValueError('Need all first13 independent timing tables')
    names=tuple(callbacks);sync=synchronize or (lambda:None);samples={name:[] for name in names};errors={name:[] for name in names};stats={name:[] for name in names}
    order_records=[]
    for repetition in range(warmups+repeats):
        order=names[repetition%len(names):]+names[:repetition%len(names)];order_records.append(order)
        for name in order:
            sync();tick=time.perf_counter();error=None;row=failed_stats()
            try:
                # No truth, mechanism object/coefficients/family, or law ID is
                # passed into a timed callback. Only observed table/folds/seed.
                result=callbacks[name](tables[repetition],fold_ids[repetition],int(seeds[repetition]));row=np.asarray(result['stats'],float);error=result.get('error')
                if row.shape!=(6,):raise ValueError('Timing callback stats schema')
                if row[5] not in (0,1) or (not row[5] and (not np.isfinite(row[:5]).all() or row[1]<=0 or row[2]>row[3])):raise ValueError('Invalid successful timing callback interval')
            except Exception as exc:error=repr(exc);row=failed_stats()
            sync();elapsed=1000*(time.perf_counter()-tick)
            samples[name].append(elapsed);stats[name].append(row.tolist());errors[name].append(error)
    measurements=[]
    for name in names:
        ms=samples[name][warmups:];flags=[bool(row[5]) for row in stats[name]]
        measurements.append(dict(method=name,raw_ms=ms,warmup_ms=samples[name][:warmups],median_ms=float(np.median(ms)),mean_ms=float(np.mean(ms)),
            p10_ms=float(np.quantile(ms,.1)),p90_ms=float(np.quantile(ms,.9)),failure_flags=flags,warmup_failures=sum(flags[:warmups]),timed_failures=sum(flags[warmups:]),
            errors=errors[name],stats=stats[name]))
    return dict(warmups=warmups,repeats=repeats,method_order=names,rotated_orders=order_records,measurements=measurements)

def run(args):
    if not 0<=args.task_index<args.task_count:raise ValueError('Invalid timing shard')
    protocol=read(args.protocol);guard=EvaluationGuard(args.protocol,args.authorization);guard.check('test');gate=read(args.authorization)
    # Missing runtime libraries are an environment/setup failure, not a whole
    # bank of scientific fitting failures or meaningful timing measurements.
    import sklearn
    import sklearn.ensemble
    import sklearn.linear_model
    require_bound(gate,'timing_source_sha256',sha(__file__),args.model_id)
    settings=protocol['timing'];warmups=int(settings['warmups']);repeats=int(settings['repetitions'])
    if warmups!=3 or repeats!=10 or not settings['cache_predictions_excluded'] or not settings['include_all_context_fits_and_CI']:raise ValueError('Timing protocol differs from registered independent run')
    if args.context_size!=int(protocol['context_size']):raise ValueError('Context budget must equal frozen deployment protocol')
    if args.alpha not in protocol['alphas']:raise ValueError('Unregistered nominal interval level')
    manifest=read(args.manifest)
    if manifest['protocol_sha256']!=sha(args.protocol):raise ValueError('Raw timing manifest protocol mismatch')
    groups=eligible_groups(protocol,manifest)[args.task_index::args.task_count]
    threads=args.cpu_threads or int(os.environ.get('NSLOTS',os.environ.get('SLURM_CPUS_PER_TASK',os.cpu_count() or 1)))
    if threads<1:raise ValueError('Positive allocated CPU budget required')
    torch.set_num_threads(threads);torch.set_float32_matmul_precision('high')
    if args.estimator_jobs is None:args.estimator_jobs=threads
    if args.estimator_jobs<1 or args.estimator_jobs>threads:raise ValueError('Per-estimator jobs must fit allocated timing CPU budget')
    source=source_hashes();bundle=load_bundle(args,gate);callbacks=make_callbacks(bundle,args)
    sync=(lambda:torch.cuda.synchronize(torch.device(args.device))) if str(args.device).startswith('cuda') else (lambda:None)
    outputs=[];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=threads):
        for group in groups:
            raw_path=Path(group['raw_path']);path=out/(group['name']+'.timing.json');ledger=path.with_suffix('.sha256.json')
            binding=dict(protocol_sha256=sha(args.protocol),raw_sha256=group['raw_sha256'],raw_metadata_sha256=group['metadata_sha256'],raw_manifest_sha256=sha(args.manifest),
                authorization_sha256=sha(args.authorization),source_hashes=source,model_id=args.model_id,inputs=bundle.inputs,
                alpha=args.alpha,context_size=args.context_size,query_batch=args.query_batch,cpu_threads=threads,estimator_jobs=args.estimator_jobs,
                methods=list(callbacks),table_indices=list(range(warmups+repeats)))
            if path.exists() or ledger.exists():
                if not path.exists() or not ledger.exists() or read(ledger)['sha256']!=sha(path) or read(path)['binding']!=clean(binding):raise ValueError('Interrupted/changed timing output requires explicit recovery')
                outputs.append(dict(path=str(path.resolve()),sha256=sha(path),cell=group['name']));continue
            if sha(raw_path)!=group['raw_sha256'] or sha(group['metadata_path'])!=group['metadata_sha256']:raise ValueError('Raw timing table bank changed')
            with np.load(raw_path,allow_pickle=False) as z:
                # Materialize each raw key once, and NEVER read z['truth'].
                raw={key:z[key][:warmups+repeats] for key in ('x','a','m','y','fold_ids','fold_seed','observation_seed','mechanism_id')}
            if len(raw['x'])!=warmups+repeats:raise ValueError('Timing bank has fewer than13 tables')
            if set(raw['mechanism_id'].astype(str))&bundle.blocked_ids or (set(map(int,raw['observation_seed']))|set(map(int,raw['fold_seed'])))&bundle.blocked_seeds:raise ValueError('Timing test/training identity overlap')
            tables=[{key:raw[key][i] for key in ('x','a','m','y')} for i in range(warmups+repeats)]
            measurements=benchmark(tables,raw['fold_ids'],raw['fold_seed'],callbacks,warmups,repeats,sync)
            record=dict(cell=group['name'],n=int(group['n']),p=int(group['p']),binding=binding,**measurements,
                observation_seeds=raw['observation_seed'],fold_seeds=raw['fold_seed'],created_utc=datetime.now(timezone.utc).isoformat(),
                scope='Single-table complete fresh pipeline for each arm. PFN arms all compute three views and full-n score/features before CI. No prediction sharing/caching across arms. Classic fits and teacher transfers/CI included; loads/calibrator fitting/raw I/O excluded.',
                point_note='Core13 retain canonical primary points; new learned heads/actions use corrected density means. Canonical scoreSE is not proved corrected-point SE.',
                runtime=dict(torch=torch.__version__,device=str(args.device),cuda_device=torch.cuda.get_device_name() if str(args.device).startswith('cuda') else None,
                    numpy=np.__version__,scikit_learn=sklearn.__version__,torch_cpu_threads=torch.get_num_threads(),allocated_SGE_slots=os.environ.get('NSLOTS'),allocated_SLURM_slots=os.environ.get('SLURM_CPUS_PER_TASK')))
            atomic_json(path,record);atomic_json(ledger,dict(sha256=sha(path),source_sha256=sha(__file__)));outputs.append(dict(path=str(path.resolve()),sha256=sha(path),cell=group['name']))
            print('TIMING_CELL',group['name'],flush=True)
    if source!=source_hashes():raise ValueError('Timing source changed during execution')
    suffix='' if args.task_count==1 else f'.part-{args.task_index}-of-{args.task_count}'
    atomic_json(out/('manifest'+suffix+'.json'),dict(protocol_sha256=sha(args.protocol),source_hashes=source,model_id=args.model_id,groups=outputs,methods=list(callbacks),
        raw_predictions_reused=False,truth_in_timed_callbacks=False,warmups=3,repeats=10));return outputs

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','manifest','checkpoint','posttrain-dir','calibration-cache','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--heuristics-dir');p.add_argument('--teacher-latent');p.add_argument('--teacher-fsp');p.add_argument('--authorization');p.add_argument('--model-id',default='medium')
    p.add_argument('--device',default='cuda');p.add_argument('--alpha',type=float,default=.05);p.add_argument('--context-size',type=int,default=256);p.add_argument('--query-batch',type=int,default=128)
    p.add_argument('--cpu-threads',type=int);p.add_argument('--estimator-jobs',type=int);p.add_argument('--allow-partial',action='store_true');p.add_argument('--task-index',type=int,default=int(os.environ.get('SGE_TASK_ID','1'))-1);p.add_argument('--task-count',type=int,default=1)
    return p

if __name__=='__main__':
    args=parser().parse_args()
    if args.authorization is None:args.authorization=str(Path(args.protocol).parent/'audit/test_authorization.json')
    run(args)
