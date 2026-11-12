"""Learn and freshly calibrate NIE-error quantiles on a frozen real V3 estimator.

This is a mediation-specific frozen-backbone/error-head route, not original
TabCon and not the complete custom MediPFN v2 post-training pipeline. Independent
training/DEV tasks fit/select the head; fresh ID-mixture calibration follows
selection. No source-OOD/composition-OOD observations are read or generated.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from .frozen import BackboneConfig,estimate_table,sha256,make_shared_frozen_v3_factory
from .error_head import NIEErrorHead,LEVELS,save_head,load_head
from .run_state import allocation_threads,initialize_output

VARIANT='frozen-v3-mediation-NIE-error-head'
SUMMARY_NAMES=('log_n','log_p_plus_one','log_q','estimate_over_y_scale','se_over_y_scale',
               'treated_fraction','mean_y_over_scale','log_y_scale','score_q05_over_scale',
               'score_q50_over_scale','score_q95_over_scale','pi_clip_fraction','g_clip_fraction','log_max_ratio')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def feature_vector(result,table):
    """Only observed quantities and estimated scores/embeddings enter the head."""
    embedding=np.asarray(result['pooled_context'],dtype=np.float64)
    if embedding.ndim!=1 or not len(embedding):raise ValueError('Missing pooled V3 embeddings')
    y=np.asarray(table['y'],float);scale=max(float(y.std()),1e-3)
    scores=np.asarray(result['scores'],float)
    diagnostics=result['diagnostics']
    clip_pi=np.mean([v['pi_clipped'] for v in diagnostics]);clip_g=np.mean([v['g_clipped'] for v in diagnostics])
    ratio=max(v['ratio_max'] for v in diagnostics)
    q=np.quantile(scores,[.05,.5,.95])/scale
    summaries=np.array([np.log(len(y)),np.log1p(table['x'].shape[1]),np.log(table['m'].shape[1]),
             result['estimate']/scale,result['standard_error']/scale,np.mean(table['a']),
             y.mean()/scale,np.log(scale),*q,clip_pi,clip_g,np.log1p(ratio)])
    out=np.concatenate([embedding,summaries]).astype(np.float32)
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite error-head feature')
    return out


def task_for_case(index,n_values,seed,partition):
    from dgp import registry,available_sources,sample_task
    rows=registry();ids=set(available_sources('train'))
    groups=sorted(set(v['source_group'] for v in rows if v['source_id'] in ids))
    offset={'train':100000000,'dev':200000000,'calibration':300000000}[partition]
    r=np.random.default_rng(seed+offset+index*1009)
    # Exact family balance for train/DEV; IID uniform-family ID mixture for fresh
    # calibration, which is the law to which an ID-mixture calibration refers.
    family=groups[int(r.integers(len(groups))) if partition=='calibration' else index%len(groups)]
    candidates=[v['source_id'] for v in rows if v['source_id'] in ids and v['source_group']==family]
    source=candidates[int(r.integers(len(candidates)))];n=int(r.choice(n_values))
    mechanism_seed=seed+offset+index*1009+701;table_seed=seed+offset+index*1009+17
    t=sample_task(source,n,table_seed,mechanism_seed=mechanism_seed,compute_truth=True)
    if source not in ids:raise RuntimeError('Held-out source entered uncertainty fitting')
    return t,table_seed+31


def cache_partition(name,count,args,config,factory,out,seen_ids,expected_dim=None):
    vectors=[];targets=[];records=[];path=out/(name+'_features.jsonl')
    with path.open('x',encoding='utf-8') as stream:
        for i in range(count):
            t=None
            try:
                t,fold_seed=task_for_case(i,args.n_values,args.seed,name)
                mid=t['mechanism_id']
                if mid in seen_ids:raise RuntimeError('Mechanism reused across independent partitions')
                result=estimate_table(t,config,seed=fold_seed,factory=factory,
                                      return_scores=True,collect_embeddings=True)
                features=feature_vector(result,t)
                if expected_dim is None:expected_dim=len(features)
                if len(features)!=expected_dim:raise RuntimeError('Backbone feature dimension changed')
                scale=max(float(result['standard_error']),args.standard_error_floor)
                error=(float(result['estimate'])-float(t['truth']))/scale
                if not np.isfinite(error):raise FloatingPointError('Nonfinite synthetic error target')
                rec=dict(index=i,partition=name,features=features.tolist(),standardized_error=error,
                    estimate=result['estimate'],truth=float(t['truth']),interval_scale=scale,
                    standard_error=result['standard_error'],seconds=result['seconds'],metadata=t['metadata'])
                stream.write(json.dumps(rec,separators=(',',':'))+'\n');stream.flush()
                vectors.append(features);targets.append(error);records.append(rec)
                seen_ids.add(mid)
                if i==0 or (i+1)%16==0:print(json.dumps(dict(stage='features',partition=name,completed=i+1,total=count)),flush=True)
            except Exception as e:
                failure=dict(status='FAILED_FEATURE_EXTRACTION',partition=name,index=i,error=repr(e),
                     traceback=traceback.format_exc(),metadata=t['metadata'] if t is not None else None,
                     completed=len(records),attempted=i+1)
                (out/('feature_failure_'+name+'_'+str(i)+'.json')).write_text(json.dumps(failure,indent=2)+'\n')
                raise
    manifest=dict(partition=name,cases=len(records),unique_mechanisms=len(set(v['metadata']['mechanism_id'] for v in records)),
                  families=dict(Counter(v['metadata']['source_group'] for v in records)),
                  dimensions=dict(Counter(str(len(v['features'])) for v in records)),sha256=sha256(path),
                  failures=0,sampling='IID uniform-family ID mixture' if name=='calibration' else 'exact family-balanced source mixture')
    (out/(name+'_cache_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    return np.stack(vectors),np.asarray(targets,dtype=np.float32),records


def quantile_indices(level):
    return {0.90:(2,3),0.95:(1,4),0.99:(0,5)}[level]


def calibrate(errors,quantiles):
    """Additive corrections in standardized-error units, nested across levels."""
    result={};previous=0.;n=len(errors)
    for level in (.90,.95,.99):
        lo,hi=quantile_indices(level)
        score=np.maximum(errors-quantiles[:,hi],quantiles[:,lo]-errors)
        rank=int(np.ceil((n+1)*level))
        if rank>n:raise ValueError('Too few independent calibration tasks for requested nominal level')
        raw=float(np.partition(score,rank-1)[rank-1]);correction=max(0.,previous,raw);previous=correction
        result[str(round(level*100))]=dict(nominal=level,rank=rank,raw_correction=raw,correction=correction,
               calibration_tasks=n,calibration_coverage=float(np.mean((errors>=quantiles[:,lo]-correction)&(errors<=quantiles[:,hi]+correction))))
    return result


def calibrated_interval(table,config,head_checkpoint,calibration_path,factory=None,adapter_checkpoint=None):
    """Deploy a frozen head/calibration pair; no labels or optimization accepted."""
    cal=json.loads(Path(calibration_path).read_text())
    if cal.get('status')!='CALIBRATED_ID_MIXTURE' or sha256(head_checkpoint)!=cal['head_checkpoint_sha256']:
        raise ValueError('Wrong/unfrozen uncertainty checkpoint')
    binding=cal['backbone_binding']
    for key in ('model_version','package_version','classifier_sha256','regressor_sha256','n_estimators','probability_floor'):
        if binding['base_config'][key]!=getattr(config,key):raise ValueError('Different point-estimator configuration: '+key)
    if binding.get('adapter_sha256'):
        if not adapter_checkpoint or sha256(adapter_checkpoint)!=binding['adapter_sha256']:
            raise ValueError('Calibrated adapter route needs its matching frozen adapter checkpoint')
        if factory is None:
            from .adapter import load_adapter,adapted_factory
            adapter,_=load_adapter(adapter_checkpoint,config);factory=adapted_factory(config,adapter)
    point=estimate_table(table,config,return_scores=True,collect_embeddings=True,factory=factory)
    head,doc=load_head(head_checkpoint,binding,map_location=config.device);head.to(config.device)
    schema=doc['feature_schema']
    if canonical_hash(schema)!=cal['feature_schema_sha256']:raise ValueError('Different feature normalization')
    features=(feature_vector(point,table)-np.asarray(schema['center']))/np.asarray(schema['scale'])
    with torch.no_grad():q=head(torch.tensor(features[None],dtype=torch.float32,device=config.device))[0].cpu().numpy()
    scale=max(point['standard_error'],cal['standard_error_floor']);intervals={}
    for level in (.90,.95,.99):
        lower,upper=quantile_indices(level);correction=cal['corrections'][str(round(level*100))]['correction']
        intervals[str(round(level*100))]=[float(point['estimate']-scale*(q[upper]+correction)),
                                        float(point['estimate']-scale*(q[lower]-correction))]
    return dict(estimate=point['estimate'],standard_error=point['standard_error'],interval_scale=scale,
                intervals=intervals,variant=cal['variant'],head_checkpoint_sha256=cal['head_checkpoint_sha256'],
                calibration_sha256=sha256(calibration_path),point_seconds=point['seconds'])


def run(args):
    from dgp import SOURCE_COMMIT
    torch.set_num_threads(allocation_threads());torch.manual_seed(args.seed)
    config=BackboneConfig(**json.loads(Path(args.config).read_text()))
    if config.model_version!='v3':raise ValueError('Actual TabPFN-v3 required; no implicit fallback')
    fixture=json.loads(Path(args.fixture_report).read_text())
    if fixture.get('status')!='PASS':raise ValueError('Real GPU fixture has not passed')
    for key in ('model_version','package_version','classifier_sha256','regressor_sha256','n_estimators','probability_floor'):
        if fixture.get('config',{}).get(key)!=getattr(config,key):raise ValueError('Different real fixture: '+key)
    binding=dict(base_config=asdict(config),source_checkpoint_sha256=dict(classifier=config.classifier_sha256,
                  regressor=config.regressor_sha256),source_commit=SOURCE_COMMIT,
                  fixture_report_sha256=sha256(args.fixture_report),adapter_sha256=None,
                  feature_program_sha256=sha256(__file__),frozen_program_sha256=sha256(Path(__file__).with_name('frozen.py')))
    variant=VARIANT
    factory=make_shared_frozen_v3_factory(config)
    if args.adapter_checkpoint:
        from .adapter import load_adapter,adapted_factory
        adapter,_=load_adapter(args.adapter_checkpoint,config)
        binding['adapter_sha256']=sha256(args.adapter_checkpoint)
        factory=adapted_factory(config,adapter);variant='frozen-v3-offline-mediation-adapter-NIE-error-head'
    out=initialize_output(args.output,dict(variant=variant,arguments=vars(args),binding=binding),__file__)
    started=time.perf_counter();seen=set()
    train_x,train_y,train_records=cache_partition('train',args.cases,args,config,factory,out,seen)
    dev_x,dev_y,dev_records=cache_partition('dev',args.dev_cases,args,config,factory,out,seen,train_x.shape[1])
    center=train_x.mean(0);scale=np.maximum(train_x.std(0),1e-4)
    schema=dict(input_dimension=train_x.shape[1],pooled_embedding_dimension=train_x.shape[1]-len(SUMMARY_NAMES),
                summary_names=SUMMARY_NAMES,center=center.tolist(),scale=scale.tolist(),
                feature_scale_floor=1e-4,standard_error_floor=args.standard_error_floor)
    tx=torch.tensor((train_x-center)/scale,device=config.device);ty=torch.tensor(train_y,device=config.device)
    dx=torch.tensor((dev_x-center)/scale,device=config.device);dy=torch.tensor(dev_y,device=config.device)
    head=NIEErrorHead(tx.shape[1],width=args.width).to(config.device)
    opt=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=1e-4)
    best=float('inf');best_step=0;history=[]
    for step in range(1,args.head_steps+1):
        head.train();opt.zero_grad(set_to_none=True)
        ix=torch.randint(len(tx),(min(args.batch_size,len(tx)),),device=config.device)
        loss=head.loss(tx[ix],ty[ix]);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.))
        if not np.isfinite(float(loss.detach())) or not np.isfinite(norm):raise FloatingPointError('Nonfinite head optimization')
        opt.step()
        if step==1 or step%args.dev_every==0 or step==args.head_steps:
            head.eval()
            with torch.no_grad():value=float(head.loss(dx,dy))
            rec=dict(step=step,loss=float(loss.detach()),dev_pinball=value);history.append(rec);print(json.dumps(rec),flush=True)
            if value<best:
                best=value;best_step=step
                save_head(out/'best.pt',head,binding,schema,dict(variant=variant,selection='independent family-balanced DEV pinball',
                      dev_loss=value,step=step,train_cases=args.cases,dev_cases=args.dev_cases,
                      ready_for_interval=False,calibration_required=True))
    save_head(out/'last.pt',head,binding,schema,dict(variant=variant,step=args.head_steps,ready_for_interval=False,calibration_required=True))
    (out/'head_selection.json').write_text(json.dumps(dict(best_step=best_step,best_dev=best,history=history),indent=2)+'\n')
    # Calibration labels are generated and inspected only after model selection.
    best_head,_=load_head(out/'best.pt',binding,map_location=config.device);best_head.to(config.device);best_head.eval()
    cal_x,cal_y,cal_records=cache_partition('calibration',args.cal_cases,args,config,factory,out,seen,train_x.shape[1])
    with torch.no_grad():q=best_head(torch.tensor((cal_x-center)/scale,device=config.device)).cpu().numpy()
    corrections=calibrate(cal_y,q)
    calibrated=dict(status='CALIBRATED_ID_MIXTURE',variant=variant,head_checkpoint_sha256=sha256(out/'best.pt'),
        backbone_binding=binding,feature_schema_sha256=canonical_hash(schema),source_checkpoint_sha256=binding['source_checkpoint_sha256'],
        standard_error_floor=args.standard_error_floor,scale_definition='max(estimated score standard error, recorded floor); same scale for targets and endpoints',
        interval_definition='[estimate-scale*(q_upper+correction), estimate-scale*(q_lower-correction)]',
        corrections=corrections,calibration_cache_sha256=sha256(out/'calibration_features.jsonl'),
        calibration_law='fresh IID uniform mixture over the eight training provenance families and configured sample sizes',
        applicability='Same ID task-mixture evaluation only; no OOD coverage guarantee; no original TabCon reproduction',
        training_cases=args.cases,dev_cases=args.dev_cases,calibration_cases=args.cal_cases)
    (out/'calibration.json').write_text(json.dumps(calibrated,indent=2)+'\n')
    completion=dict(status='TRAINED_HEAD_AND_FRESH_ID_CALIBRATION_COMPLETE',variant=variant,
             best_checkpoint_sha256=sha256(out/'best.pt'),last_checkpoint_sha256=sha256(out/'last.pt'),
             calibration_sha256=sha256(out/'calibration.json'),seconds=time.perf_counter()-started,
             binding=binding,feature_dimension=train_x.shape[1],failures=0,
             evaluation_status='No ID/OOD scientific test evaluation performed')
    (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n');print(json.dumps(completion),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--fixture-report',required=True);p.add_argument('--output',required=True)
    p.add_argument('--adapter-checkpoint');p.add_argument('--cases',type=int,default=1024);p.add_argument('--dev-cases',type=int,default=384)
    p.add_argument('--cal-cases',type=int,default=512);p.add_argument('--n-values',default='200,600,1000');p.add_argument('--seed',type=int,default=93061103)
    p.add_argument('--head-steps',type=int,default=2000);p.add_argument('--width',type=int,default=256);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--dev-every',type=int,default=50);p.add_argument('--standard-error-floor',type=float,default=1e-3)
    a=p.parse_args();a.n_values=[int(v) for v in a.n_values.split(',')]
    if min(a.cases,a.dev_cases,a.cal_cases,a.head_steps,a.batch_size,a.dev_every)<1 or min(a.n_values)<40 or a.standard_error_floor<=0:p.error('Invalid training/calibration budget')
    run(a)
