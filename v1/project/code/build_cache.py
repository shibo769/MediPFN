"""Hash-bound per-cell nuisance/feature caches; all table attempts retained.

Raw manifest groups come from generate_data.py. This module never trains a
model or merges role banks. merge_cache.py owns the training aggregation.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time,uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from model import load_checkpoint
from inference import crossfit_views,VIEWS,select_indices,raw_to_nuisance,canonical_nie,generic_features,mediation_features
from evaluate import EvaluationGuard

SCHEMA='v9-nuisance-feature-cache-1'
SAFE_SPLITS=('train','dev','calibration','cal','capacity_train','capacity_dev')
VIEW_NAMES=VIEWS+('consensus',)
STAT_COLUMNS=('estimate','se','lo','hi','clipping_rate')

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return str(x)
    return x
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()
def resolve(path,manifest,protocol):
    path=Path(path)
    if path.is_absolute():return path.resolve()
    options=[Path(manifest).parent/path,Path(protocol).parent/path]
    return next((p.resolve() for p in options if p.exists()),options[0].resolve())
def source_hashes():
    folder=Path(__file__).resolve().parent
    return {n:sha(folder/n) for n in ('build_cache.py','model.py','inference.py','evaluate.py','baselines.py')}
def widths(model):
    single=22+2*model.config.width
    return single,single+74
def ensure_guard(guard,split):return guard.check(split if split in SAFE_SPLITS else 'test')
def check_checkpoint_release(authorization,digest):
    """Accept the shared release schema's scalar/per-model checkpoint hashes."""
    def hashes(value):
        if isinstance(value,str):return {value} if len(value)==64 else set()
        if isinstance(value,dict):return set().union(*(hashes(v) for v in value.values())) if value else set()
        if isinstance(value,(list,tuple)):return set().union(*(hashes(v) for v in value)) if value else set()
        return set()
    declared=set()
    has_spec=any(key in authorization for key in ('checkpoint_sha256','checkpoint_hashes'))
    for key in ('checkpoint_sha256','checkpoint_hashes'):
        if key in authorization:declared|=hashes(authorization[key])
    for model in authorization.get('models',{}).values():
        if isinstance(model,dict):
            has_spec|=any(key in model for key in ('checkpoint_sha256','checkpoint_hashes'))
            declared|=hashes(model.get('checkpoint_sha256',model.get('checkpoint_hashes',[])))
    if has_spec and not declared:raise ValueError('Malformed checkpoint release binding')
    if declared and digest not in declared:raise ValueError('Checkpoint not in test release set')

@torch.inference_mode()
def batched_views(model,rows,fold_ids,seeds,context_size,query_batch,device,workers=1):
    """Exactly the same per-table context/normalization/score semantics.

    Only independent support contexts and query rows are stacked across tables.
    Context selection runs on CPU threads; labels never enter that selection.
    """
    rows=np.asarray(rows,np.float64);ff=np.asarray(fold_ids);b,n,d=rows.shape;p=d-3;started=time.perf_counter()
    if ff.shape!=(b,n) or p<1 or not np.isfinite(rows).all():raise ValueError('Invalid batched rows/folds')
    if any(set(np.unique(v))!={0,1} for v in ff):raise ValueError('Two folds required per table')
    jobs=[(i,vi,view,fold) for i in range(b) for vi,view in enumerate(VIEWS) for fold in (0,1)]
    def plan(job):
        i,vi,view,fold=job;ids=np.flatnonzero(ff[i]!=fold);qq=np.flatnonzero(ff[i]==fold)
        selected=select_indices(rows[i,ids,:p],context_size,int(seeds[i])+1009*vi+37*fold,view);chosen=ids[selected]
        sx=rows[i,ids,:p].astype(float);mean=sx.mean(0);scale=np.maximum(sx.std(0),1e-6);distance=np.sqrt(np.sum(((rows[i,chosen,:p]-mean)/scale)**2,1))
        metadata=dict(view=view,fold=fold,support_fraction=len(chosen)/len(ids),selected_x_distance_mean=float(distance.mean()),selected_x_distance_max=float(distance.max()))
        return (i,vi,fold),dict(chosen=chosen,query=qq,metadata=metadata,selection=dict(view=view,fold=fold,indices=chosen.copy()))
    if workers>1:
        with ThreadPoolExecutor(max_workers=workers) as pool:plans=dict(pool.map(plan,jobs))
    else:plans=dict(map(plan,jobs))
    allnu=np.full((b,3,n,5),np.nan,np.float64);pooled=[[[] for _ in VIEWS] for _ in range(b)]
    for vi,view in enumerate(VIEWS):
        for fold in (0,1):
            contexts=[plans[i,vi,fold] for i in range(b)]
            if len({len(c['chosen']) for c in contexts})!=1 or len({len(c['query']) for c in contexts})!=1:raise ValueError('Batch requires matched support/query lengths')
            support=np.stack([rows[i,c['chosen']] for i,c in enumerate(contexts)]);query=np.stack([rows[i,c['query'],:p] for i,c in enumerate(contexts)])
            context=model.encode_support(torch.as_tensor(support,dtype=torch.float32,device=device));embedding=context['pooled'].detach().cpu().numpy()
            for i in range(b):pooled[i][vi].append(embedding[i])
            for start in range(0,query.shape[1],query_batch):
                prediction=model.predict_nuisance(context,torch.as_tensor(query[:,start:start+query_batch],dtype=torch.float32,device=device)).detach().cpu().numpy()
                for i,c in enumerate(contexts):allnu[i,vi,c['query'][start:start+query_batch]]=raw_to_nuisance(prediction[i])
    results=[]
    for i in range(b):
        consensus=allnu[i].mean(0);summaries=[canonical_nie(rows[i],nu) for nu in allnu[i]]+[canonical_nie(rows[i],consensus)]
        metadata=[plans[i,vi,fold]['metadata'] for vi in range(3) for fold in (0,1)]
        selections=[plans[i,vi,fold]['selection'] for vi in range(3) for fold in (0,1)]
        generic=generic_features(rows[i],summaries[0],np.asarray(pooled[i][0]));multi=mediation_features(rows[i],consensus,summaries,metadata,generic)
        results.append(dict(nuisance=consensus,views=allnu[i],summaries=summaries,x_single=generic,x_multi=multi,
            estimates=np.array([summaries[0]['estimate'],summaries[-1]['estimate']]),ses=np.array([summaries[0]['se'],summaries[-1]['se']]),
            metadata=metadata,support_indices=selections,attempted_rows=n,used_rows=n,error=''))
    seconds=(time.perf_counter()-started)/b
    for result in results:result['seconds']=seconds
    return results

def raw_group(group,manifest_path,protocol_path,protocol_hash):
    raw_path=resolve(group['raw_path'],manifest_path,protocol_path);metadata_path=resolve(group['metadata_path'],manifest_path,protocol_path)
    if sha(raw_path)!=group['raw_sha256'] or sha(metadata_path)!=group['metadata_sha256']:raise ValueError('Raw group/metadata hash mismatch: '+group['name'])
    metadata=read(metadata_path)
    declared=metadata.get('protocol_sha256',metadata.get('protocol_hash'))
    if declared is not None and declared!=protocol_hash:raise ValueError('Raw metadata protocol mismatch')
    if metadata.get('raw_sha256')!=group['raw_sha256']:raise ValueError('Raw metadata data-hash mismatch')
    with np.load(raw_path,allow_pickle=False) as z:raw={key:z[key] for key in z.files}
    count,n,p=(int(group[k]) for k in ('count','n','p'))
    required=('x','a','m','y','fold_ids','truth','mechanism_id','observation_seed','fold_seed')
    if any(k not in raw for k in required):raise ValueError('Raw bank missing required fields')
    if raw['x'].shape!=(count,n,p) or any(raw[k].shape!=(count,n) for k in ('a','m','y','fold_ids')):raise ValueError('Raw covariate/label/fold shape mismatch')
    if any(raw[k].shape!=(count,) for k in ('truth','mechanism_id','observation_seed','fold_seed')):raise ValueError('Raw provenance row count mismatch')
    if not np.isfinite(raw['truth']).all():raise ValueError('Raw exact truth must be finite for every attempt')
    if len(metadata.get('tables',[]))!=count:raise ValueError('Raw metadata attempt count mismatch')
    for i,table in enumerate(metadata['tables']):
        if str(table['mechanism_id'])!=str(raw['mechanism_id'][i]) or int(table['observation_seed'])!=int(raw['observation_seed'][i]) or int(table['fold_seed'])!=int(raw['fold_seed'][i]):raise ValueError('Raw table provenance mismatch')
    return raw,dict(raw_path=str(raw_path),raw_sha256=group['raw_sha256'],raw_metadata_path=str(metadata_path),raw_metadata_sha256=group['metadata_sha256'])

def make_binding(group,args,manifest_hash,checkpoint_hash,source):
    return dict(schema=SCHEMA,protocol_sha256=sha(args.protocol),manifest_sha256=manifest_hash,checkpoint_sha256=checkpoint_hash,
        group=group,context_size=args.context_size,query_batch=args.query_batch,batch_size=args.batch_size,workers=args.workers,
        model_seed_index=args.model_seed_index,view_names=VIEW_NAMES,include_pooled=True,source_hashes=source)

def process_group(group,args,model,guard,manifest_hash,checkpoint_hash,source,infer=None):
    """infer injection is only for synthetic failure-path contracts."""
    ensure_guard(guard,group['split'])
    name=str(group['name'])
    if Path(name).name!=name or name in ('.','..'):raise ValueError('Group name must be a filename component')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True);path=output/(name+'.npz');ledger=path.with_suffix('.json')
    binding=make_binding(group,args,manifest_hash,checkpoint_hash,source)
    if path.exists() or ledger.exists():
        if not path.exists() or not ledger.exists():raise ValueError('Interrupted cache commit requires explicit recovery: '+name)
        record=read(ledger)
        # JSON encodes tuples as lists, so compare canonical serializations.
        if record['binding']!=clean(binding) or record['sha256']!=sha(path):raise ValueError('Resume binding/hash mismatch: '+name)
        raw_path=resolve(group['raw_path'],args.manifest,args.protocol);metadata_path=resolve(group['metadata_path'],args.manifest,args.protocol)
        if sha(raw_path)!=group['raw_sha256'] or sha(metadata_path)!=group['metadata_sha256']:raise ValueError('Raw data drift on resume')
        return manifest_entry(path,ledger,group,model)
    raw,raw_binding=raw_group(group,args.manifest,args.protocol,binding['protocol_sha256'])
    count,n=int(group['count']),int(group['n']);single_width,multi_width=widths(model)
    nuisance=np.full((count,4,n,5),np.nan,np.float64);stats=np.full((count,4,5),np.nan,np.float64);stats[:,:,2]=np.inf;stats[:,:,3]=-np.inf
    scores=np.full((count,4,n),np.nan,np.float64);x_single=np.full((count,single_width),np.nan,np.float32);x_multi=np.full((count,multi_width),np.nan,np.float32)
    estimates=np.full((count,2),np.nan);ses=np.full((count,2),np.nan);seconds=np.full(count,np.nan);errors=[];types=[];stages=[];metadata_records=[]
    predict=crossfit_views if infer is None else infer;started=time.perf_counter()
    def store_result(result,i):
        ss=np.asarray(result['x_single']);mm=np.asarray(result['x_multi']);nu=np.concatenate((result['views'],result['nuisance'][None]),axis=0)
        if ss.shape!=(single_width,) or mm.shape!=(multi_width,) or nu.shape!=(4,n,5) or len(result['summaries'])!=4:raise ValueError('Predictor feature/view schema mismatch')
        st=np.array([[v[k] for k in STAT_COLUMNS] for v in result['summaries']]);sc=np.stack([v['score'] for v in result['summaries']])
        est=np.asarray(result['estimates']);se=np.asarray(result['ses'])
        if est.shape!=(2,) or se.shape!=(2,) or sc.shape!=(4,n) or (st[:,1]<=0).any():raise ValueError('Nonregular canonical point/SE/score')
        if not all(np.isfinite(v).all() for v in (ss,mm,nu,st,sc,est,se)):raise ValueError('Nonfinite inference/feature result')
        if result['attempted_rows']!=n or result['used_rows']!=n:raise ValueError('Full-n score contract violated')
        nuisance[i]=nu;stats[i]=st;scores[i]=sc;x_single[i]=ss;x_multi[i]=mm;estimates[i]=est;ses[i]=se
        metadata_records.append(dict(index=i,view_metadata=result['metadata'],support_indices=[dict(view=r['view'],fold=r['fold'],indices=np.asarray(r['indices']).tolist()) for r in result['support_indices']]))
        errors.append('');types.append('');stages.append('')
    for start in range(0,count,args.batch_size):
        stop=min(count,start+args.batch_size);rows_batch=np.concatenate((raw['x'][start:stop],raw['a'][start:stop,:,None],raw['m'][start:stop,:,None],raw['y'][start:stop,:,None]),axis=-1).astype(np.float64)
        tick=time.perf_counter();batch_results=None;batch_error=None
        if infer is None and stop-start>1:
            try:batch_results=batched_views(model,rows_batch,raw['fold_ids'][start:stop],raw['fold_seed'][start:stop],args.context_size,args.query_batch,args.device,args.workers)
            except Exception as exc:batch_error=repr(exc) # Isolate failures by rerunning individual tables.
        batch_seconds=(time.perf_counter()-tick)/(stop-start)
        for i in range(start,stop):
            stage='inference';tick=time.perf_counter()
            try:
                if batch_results is not None:result=batch_results[i-start]
                else:result=predict(model,rows_batch[i-start],raw['fold_ids'][i],seed=int(raw['fold_seed'][i]),context_size=args.context_size,device=args.device,
                                    views=VIEWS,query_batch=args.query_batch,include_pooled=True)
                stage='cache_schema'
                store_result(result,i)
            except Exception as exc:
                errors.append(repr(exc));types.append(type(exc).__name__);stages.append(stage);metadata_records.append(dict(index=i,error=repr(exc),failure_type=type(exc).__name__,stage=stage,batch_fallback_reason=batch_error))
            seconds[i]=batch_seconds if batch_results is not None else time.perf_counter()-tick

    arrays=dict(nuisance=nuisance,stats=stats,scores=scores,x_single=x_single,x_multi=x_multi,estimates=estimates,ses=ses,truth=raw['truth'],
        mechanism_id=raw['mechanism_id'].astype(str),seeds=raw['observation_seed'],fold_seeds=raw['fold_seed'],observation_seed=raw['observation_seed'],fold_seed=raw['fold_seed'],
        failed=np.asarray(errors,str),failure_type=np.asarray(types,str),failure_stage=np.asarray(stages,str),success=np.asarray([not e for e in errors]),inference_seconds=seconds,
        view_names=np.asarray(VIEW_NAMES),stat_columns=np.asarray(STAT_COLUMNS))
    for field in ('split','bank','family'):arrays[field]=np.full(count,str(group[field]))
    for field in ('p','n','seed_index'):arrays[field]=np.full(count,int(group.get(field,0)))
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('xb') as f:np.savez_compressed(f,**arrays)
        if path.exists():raise FileExistsError('Group cache already committed by another task')
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()
    record=dict(schema=SCHEMA,utc=now(),binding=binding,sha256=sha(path),attempts=count,successes=sum(not e for e in errors),failures=sum(bool(e) for e in errors),
        feature_widths=dict(x_single=single_width,x_multi=multi_width),view_names=VIEW_NAMES,stat_columns=STAT_COLUMNS,raw_provenance=raw_binding,
        seconds=time.perf_counter()-started,table_evidence=metadata_records,
        timing_scope='Cache generation: batched wall time amortized within successful batches, or isolated table time on fallback; includes context selection/features/canonical score; not independent timing benchmark')
    atomic_json(ledger,record)
    return manifest_entry(path,ledger,group,model)

def manifest_entry(path,ledger,group,model):
    return dict(name=group['name'],path=str(path.resolve()),sha256=sha(path),metadata_path=str(ledger.resolve()),metadata_sha256=sha(ledger),
        split=group['split'],bank=group['bank'],seed_index=int(group.get('seed_index',0)),family=group['family'],p=int(group['p']),n=int(group['n']),count=int(group['count']),model=model.config.size)

def finalize(args,eligible,protocol_hash,manifest_hash,checkpoint_hash,source):
    output=Path(args.output);parts=[]
    for index in range(args.task_count):
        path=output/('manifest.json' if args.task_count==1 else f'manifest.part-{index}-of-{args.task_count}.json')
        if not path.exists():raise FileNotFoundError('Missing cache shard manifest: '+str(path))
        record=read(path)
        if any(record[k]!=expected for k,expected in (('protocol_sha256',protocol_hash),('checkpoint_sha256',checkpoint_hash),('raw_manifest_sha256',manifest_hash),('source_hashes',source))):raise ValueError('Cache shard binding mismatch')
        if record['task_index']!=index or record['task_count']!=args.task_count or record['model_seed_index']!=args.model_seed_index:raise ValueError('Cache shard index/seed mismatch')
        parts.append((path,record))
    groups=[group for _,part in parts for group in part['groups']];names=[g['name'] for g in groups]
    if len(set(names))!=len(names) or set(names)!={g['name'] for g in eligible}:raise ValueError('Cache finalization missing/duplicated/unexpected groups')
    for group in groups:
        if sha(group['path'])!=group['sha256'] or sha(group['metadata_path'])!=group['metadata_sha256']:raise ValueError('Cache output changed before finalization')
        ledger=read(group['metadata_path'])
        if ledger['sha256']!=group['sha256'] or ledger['binding']['checkpoint_sha256']!=checkpoint_hash or ledger['binding']['protocol_sha256']!=protocol_hash:raise ValueError('Cache ledger binding mismatch')
    lookup={g['name']:g for g in groups};record=dict(parts[0][1]);record.update(utc=now(),groups=[lookup[g['name']] for g in eligible],task_index=None,
        attempts=sum(g['count'] for g in groups),shard_manifests={str(path.resolve()):sha(path) for path,_ in parts})
    atomic_json(output/'manifest.json',record);return output/'manifest.json'

def build(args):
    if args.workers is None:args.workers=int(os.environ.get('NSLOTS',os.environ.get('SLURM_CPUS_PER_TASK',os.cpu_count() or 1)))
    if args.task_count<1 or not 0<=args.task_index<args.task_count or args.context_size<2 or args.query_batch<1 or args.batch_size<1 or args.workers<1:raise ValueError('Invalid shard/context/query budget')
    guard=EvaluationGuard(args.protocol,args.authorization);manifest=read(args.manifest);protocol_hash=sha(args.protocol)
    if manifest.get('protocol_sha256',manifest.get('protocol_hash'))!=protocol_hash:raise ValueError('Raw manifest protocol mismatch')
    eligible=[g for g in manifest['groups'] if args.model_seed_index is None or int(g.get('seed_index',0)) in (args.model_seed_index,-1)]
    distinct={int(g.get('seed_index',0)) for g in eligible if int(g.get('seed_index',0))>=0}
    if len(distinct)>1:raise ValueError('Pass --model-seed-index to keep independent model training/calibration banks separate')
    groups=eligible[args.task_index::args.task_count]
    checking_groups=eligible if args.finalize_manifest else groups
    for group in checking_groups:ensure_guard(guard,group['split'])
    manifest_hash=sha(args.manifest);checkpoint_hash=sha(args.checkpoint);source=source_hashes()
    if any(g['split'] not in SAFE_SPLITS for g in checking_groups):
        auth=read(args.authorization)
        check_checkpoint_release(auth,checkpoint_hash)
    if args.finalize_manifest:return finalize(args,eligible,protocol_hash,manifest_hash,checkpoint_hash,source)
    if args.cpu_threads is not None:torch.set_num_threads(args.cpu_threads)
    elif os.environ.get('NSLOTS'):torch.set_num_threads(int(os.environ['NSLOTS']))
    elif os.environ.get('SLURM_CPUS_PER_TASK'):torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.set_float32_matmul_precision('high');model=load_checkpoint(args.checkpoint,args.device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    entries=[]
    for group in groups:
        entry=process_group(group,args,model,guard,manifest_hash,checkpoint_hash,source);entries.append(entry)
        print('CACHE_GROUP',group['name'],flush=True)
    if source!=source_hashes() or checkpoint_hash!=sha(args.checkpoint) or manifest_hash!=sha(args.manifest) or protocol_hash!=sha(args.protocol):raise ValueError('Source/checkpoint/manifest/protocol drift during cache run')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    name='manifest.json' if args.task_count==1 else f'manifest.part-{args.task_index}-of-{args.task_count}.json'
    record=dict(schema=SCHEMA,utc=now(),protocol_sha256=protocol_hash,checkpoint_sha256=checkpoint_hash,checkpoint_path=str(Path(args.checkpoint).resolve()),
        raw_manifest_sha256=manifest_hash,model=model.config.size,parameter_count=model.parameter_count,source_hashes=source,groups=entries,
        task_index=args.task_index,task_count=args.task_count,model_seed_index=args.model_seed_index,attempts=sum(g['count'] for g in groups),role='Frozen backbone inference only; calibration/test remain separate from posttrain training banks')
    atomic_json(output/name,record);return output/name

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--context-size',type=int,default=256);p.add_argument('--query-batch',type=int,default=128)
    p.add_argument('--task-index',type=int,default=int(os.environ.get('SGE_TASK_ID','1'))-1);p.add_argument('--task-count',type=int,default=1);p.add_argument('--authorization',default=None);p.add_argument('--cpu-threads',type=int,default=None)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--workers',type=int,default=None);p.add_argument('--model-seed-index',type=int,default=None)
    p.add_argument('--finalize-manifest',action='store_true')
    return p

if __name__=='__main__':
    args=parser().parse_args()
    if args.authorization is None:args.authorization=str(Path(args.protocol).parent/'audit/test_authorization.json')
    build(args)
