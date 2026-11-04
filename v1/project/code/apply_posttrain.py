"""Apply all frozen v9 CI arms to saved views, with sealed-test authorization.

No model re-encoding, fitting or test-based selection occurs here. All raw
expert intervals and mixture parameters are retained for independent replay.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
import posttrain as pt
from merge_cache import sha,read,entries,load_entry,resolve
from evaluate import EvaluationGuard,EmpiricalErrorCalibration,interval_stats,failed_stats

POLICIES={'sft':'sft','sft_dpo':'sft_dpo','sft_rlvr':'sft_rlvr','full':'sft_dpo_rlvr','extra_sft_dpo':'extra_sft_dpo','extra_sft_rlvr':'extra_sft_rlvr','extra_sft_dpo_rlvr':'extra_sft_dpo_rlvr'}
ARMS=pt.EXPERTS+tuple(POLICIES)+('empirical_single','empirical_multi')
ALPHAS=(.10,.05,.01)

def cache_failed(a,index,view):
    flagged=False
    if 'failed' in a:
        value=np.asarray(a['failed'])[index]
        if np.ndim(value):value=value[view]
        flagged=bool(str(value)) if np.asarray(value).dtype.kind in 'US' else bool(value)
    if 'success' in a:
        value=np.asarray(a['success'])[index]
        if np.ndim(value):value=value[view]
        flagged=flagged or not bool(value)
    return flagged

def verify_posttrain(folder):
    folder=Path(folder);art=read(folder/'artifacts.json')
    if art.get('complete') is not True:raise ValueError('Incomplete post-training artifact bundle')
    for name,digest in art['head_hashes'].items():
        if sha(folder/name)!=digest:raise ValueError('Post-training weights changed '+name)
    if sha(folder/'normalization.npz')!=art['normalization_hash']:raise ValueError('Normalization drift')
    for name in tuple(pt.EXPERTS[2:])+tuple(POLICIES.values()):
        if name+'.pt' not in art['head_hashes']:raise ValueError('Missing fixed arm '+name)
    contract=read(folder/'training_contract.json')
    for name,digest in contract['source_hashes'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Post-training source changed '+name)
    trainpath=Path(contract['args']['cache'])
    candidates=[trainpath] if trainpath.is_absolute() else [Path.cwd()/trainpath,folder/trainpath,folder.parent/trainpath]
    trainpath=next((p for p in candidates if p.exists() and sha(p)==art['cache_hash']),None)
    if trainpath is None:raise ValueError('Training identity bank missing/changed')
    with np.load(trainpath,allow_pickle=False) as z:
        mids=set(z['mechanism_id'].astype(str));seeds=set(map(int,z['seeds'])) if 'seeds' in z else set()
        if 'fold_seeds' in z:seeds.update(map(int,z['fold_seeds']))
        identity=dict(checkpoint_sha256=str(z['checkpoint_sha256'].item()) if 'checkpoint_sha256' in z else None,protocol_sha256=str(z['protocol_sha256'].item()) if 'protocol_sha256' in z else None)
    return art,mids,seeds,identity

def authorize_test(protocol,authorization,posttrain_dir,checkpoint_hash,model):
    EvaluationGuard(protocol,authorization).check('test')
    gate=read(authorization);art_hash=sha(Path(posttrain_dir)/'artifacts.json')
    bound=gate.get('posttrain_artifacts_sha256');models=gate.get('models',{})
    if isinstance(bound,dict):bound=bound.get(model)
    if bound is None and model in models:bound=models[model].get('posttrain_artifacts_sha256')
    if bound!=art_hash:raise ValueError('Test gate must bind fixed post-training artifacts')
    ck=gate.get('checkpoint_sha256')
    if isinstance(ck,dict):ck=ck.get(model)
    if ck is None and model in models:ck=models[model].get('checkpoint_sha256')
    if ck!=checkpoint_hash:raise ValueError('Test gate must bind frozen backbone checkpoint')

def calibration_arrays(path,protocol_hash,checkpoint_hash):
    path=Path(path)
    if path.suffix=='.json':
        header,groups=entries(path);parts=[];bindings=[]
        if not isinstance(header,dict) or header.get('protocol_sha256')!=protocol_hash or header.get('checkpoint_sha256')!=checkpoint_hash:raise ValueError('Calibration manifest protocol/backbone mismatch')
        for entry in groups:
            if str(entry.get('split')) not in ('calibration','calibrate'):raise ValueError('Calibration manifest contains another split')
            p,a=load_entry(entry,path,protocol_hash);parts.append(a);bindings.append(dict(path=str(p),sha256=sha(p)))
        if not parts:raise ValueError('Empty calibration manifest')
        keys=('estimates','ses','truth','mechanism_id','split','seeds','fold_seeds')
        result={k:np.concatenate([a[k] for a in parts]) for k in keys}
        for key,default in (('failed',''),('success',True)):
            if any(key in a for a in parts):result[key]=np.concatenate([a[key] if key in a else np.repeat(default,len(a['truth'])) for a in parts])
        return result,dict(path=str(path),sha256=sha(path),groups=bindings)
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if 'protocol_sha256' not in a or str(a['protocol_sha256'].item())!=protocol_hash:raise ValueError('Calibration protocol drift')
    if 'checkpoint_sha256' not in a or str(a['checkpoint_sha256'].item())!=checkpoint_hash:raise ValueError('Calibration backbone mismatch')
    if not set(a['split'].astype(str)).issubset({'calibration','calibrate'}):raise ValueError('Independent calibration split required')
    return a,dict(path=str(path),sha256=sha(path))

def fit_calibrators(calibration,blocked_ids,blocked_seeds):
    a=calibration;n=len(a['truth']);ids=a['mechanism_id'].astype(str)
    if set(ids)&blocked_ids:raise ValueError('Calibration mechanism overlaps train/dev')
    all_seeds=set(map(int,a.get('seeds',[])))|set(map(int,a.get('fold_seeds',[])))
    if all_seeds&blocked_seeds:raise ValueError('Calibration observation/fold identity overlaps training')
    if 'seeds' in a and len(np.unique(a['seeds']))!=n:raise ValueError('Repeated calibration observations')
    if 'fold_seeds' in a and len(np.unique(a['fold_seeds']))!=n:raise ValueError('Repeated calibration folds')
    if 'seeds' in a and 'fold_seeds' in a and set(map(int,a['seeds']))&set(map(int,a['fold_seeds'])):raise ValueError('Calibration observation/fold seed overlap')
    result=[]
    for point in (0,1):
        records=[]
        for i in range(n):
            est,se=a['estimates'][i,point],a['ses'][i,point];failed=cache_failed(a,i,0 if point==0 else 3) or not(np.isfinite(est) and np.isfinite(se) and se>0)
            records.append(dict(split='calibration',mechanism_id=ids[i],truth=float(a['truth'][i]),summary=dict(estimate=float(est),se=float(se),failed=failed)))
        result.append(EmpiricalErrorCalibration().fit(records,blocked_ids))
    return result

def summaries(a,point):
    stats=np.asarray(a['stats']);view=0 if point==0 else stats.shape[1]-1;out=[]
    for i,row in enumerate(stats[:,view]):
        flagged=cache_failed(a,i,view)
        est,se,lo,hi,clip=map(float,row[:5]);failed=flagged or (len(row)>5 and bool(row[5])) or not(np.isfinite(row[:5]).all() and se>0 and lo<=hi)
        out.append(dict(estimate=est,se=se,clipping_rate=clip,failed=failed))
    return out

@torch.no_grad()
def apply_group(a,heads,policies,norm,calibrators):
    n=len(a['truth']);single=summaries(a,0);multi=summaries(a,1);ss=(single,multi);stats=np.tile(failed_stats(),(n,len(ARMS),len(ALPHAS),1));actions=np.full((n,len(ARMS)),-1,np.int16);point_index=np.full((n,len(ARMS)),-1,np.int8)
    quantiles=np.full((n,len(ARMS),len(ALPHAS),2),np.nan);rawci=np.full((n,4,len(ALPHAS),2),np.nan);rawq=rawci.copy();evidence={};probabilities={}
    xx=dict(x_single=a['x_single'],x_multi=a['x_multi'],estimates=a['estimates'],ses=a['ses'])
    x=pt.normalized(dict(x_multi=a['x_multi']),'x_multi',norm,'cpu')
    for method,policy in policies.items():
        j=ARMS.index(method);prob=policy(x).softmax(-1).cpu().numpy();probabilities[method]=prob;actions[:,j]=prob.argmax(-1);point_index[:,j]=actions[:,j]%2
    for ai,alpha in enumerate(ALPHAS):
        ex=pt.expert_intervals(heads,norm,**xx,alpha=alpha);rawci[:,:,ai]=ex['intervals'];rawq[:,:,ai]=ex['quantiles']
        if ai==0:evidence=ex['mixture_evidence']
        for j in range(4):
            point=j%2;point_index[:,j]=point;actions[:,j]=j;quantiles[:,j,ai]=ex['quantiles'][:,j]
            for i in range(n):stats[i,j,ai]=interval_stats(ss[point][i],alpha,endpoints=ex['intervals'][i,j])
        for method,policy in policies.items():
            j=ARMS.index(method);act=actions[:,j];pi=point_index[:,j];ci=ex['intervals'][np.arange(n),act];quantiles[:,j,ai]=ex['quantiles'][np.arange(n),act]
            for i in range(n):stats[i,j,ai]=interval_stats(ss[int(pi[i])][i],alpha,endpoints=ci[i])
        for point,method in enumerate(('empirical_single','empirical_multi')):
            j=ARMS.index(method);qq=calibrators[point].quantiles(alpha);point_index[:,j]=point;actions[:,j]=point;quantiles[:,j,ai]=qq
            for i in range(n):stats[i,j,ai]=interval_stats(ss[point][i],alpha,quantiles=qq)
    chosen_points=np.full((n,len(ARMS)),np.nan);chosen_ses=chosen_points.copy()
    for j in range(len(ARMS)):
        for i in range(n):chosen_points[i,j]=ss[int(point_index[i,j])][i]['estimate'];chosen_ses[i,j]=ss[int(point_index[i,j])][i]['se']
    output=dict(stats=stats,actions=actions,point_index=point_index,chosen_estimates=chosen_points,chosen_ses=chosen_ses,quantiles=quantiles,raw_expert_intervals=rawci,raw_expert_quantiles=rawq,arm_names=np.array(ARMS),alphas=np.array(ALPHAS),truth=a['truth'],mechanism_id=a['mechanism_id'],seeds=a['seeds'],fold_seeds=a['fold_seeds'],failed=stats[:,:,:,5].astype(bool))
    for arm,p in evidence.items():
        for key,value in p.items():output[arm+'_'+key]=value
    for arm,value in probabilities.items():output[arm+'_probabilities']=value
    return output

def run(args):
    guard=EvaluationGuard(args.protocol,args.authorization);guard.check('calibration');protocol_hash=sha(args.protocol);header,allgroups=entries(args.cache_manifest)
    if not isinstance(header,dict) or header.get('protocol_sha256')!=protocol_hash:raise ValueError('Cache manifest protocol mismatch')
    checkpoint_hash=header.get('checkpoint_sha256');model=header.get('model','unknown');art,train_ids,train_seeds,training_identity=verify_posttrain(args.posttrain_dir)
    if training_identity!={'checkpoint_sha256':checkpoint_hash,'protocol_sha256':protocol_hash}:raise ValueError('Post-training feature bank belongs to another backbone/protocol')
    calibration,binding=calibration_arrays(args.calibration_cache,protocol_hash,checkpoint_hash);calibrators=fit_calibrators(calibration,train_ids,train_seeds);cal_ids=set(calibration['mechanism_id'].astype(str));cal_seeds=set(map(int,calibration['seeds']))|set(map(int,calibration.get('fold_seeds',[])))
    heads=pt.load_heads(args.posttrain_dir,'cpu');policies={name:pt.load_policy(Path(args.posttrain_dir)/(file+'.pt'),'cpu') for name,file in POLICIES.items()}
    with np.load(Path(args.posttrain_dir)/'normalization.npz') as z:norm={key:(z[key+'_center'],z[key+'_scale']) for key in ('x_single','x_multi')}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);completed=[]
    for gi,entry in enumerate(allgroups):
        if gi%args.task_count!=args.task_index:continue
        split=str(entry.get('split',''))
        if split in ('train','calibration','calibrate'):continue
        needs_test_release=not (split.startswith('dev') or split=='validation')
        if needs_test_release:authorize_test(args.protocol,args.authorization,args.posttrain_dir,checkpoint_hash,model)
        else:guard.check(split)
        path,a=load_entry(entry,args.cache_manifest,protocol_hash);target_ids=set(a['mechanism_id'].astype(str));target_seeds=set(map(int,a['seeds']))|set(map(int,a.get('fold_seeds',[])))
        if target_ids&cal_ids or target_seeds&cal_seeds:raise ValueError('Calibration/evaluation overlap')
        if needs_test_release and (target_ids&train_ids or target_seeds&train_seeds):raise ValueError('Test/training identity overlap')
        name=entry.get('name',path.stem);target=output/(name+'.posttrain.npz');ledger=target.with_suffix('.json')
        locks=dict(protocol_sha256=protocol_hash,checkpoint_sha256=checkpoint_hash,posttrain_artifacts_sha256=sha(Path(args.posttrain_dir)/'artifacts.json'),cache_sha256=sha(path),calibration_sha256=binding['sha256'],source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('posttrain.py'),Path(__file__).with_name('evaluate.py'),Path(__file__).with_name('merge_cache.py'))})
        if target.exists():
            record=read(ledger)
            if record['locks']!=locks or record['sha256']!=sha(target):raise ValueError('Existing application artifact changed')
            completed.append(record);continue
        started=time.perf_counter();result=apply_group(a,heads,policies,norm,calibrators);np.savez_compressed(target,**result)
        record=dict(name=name,path=str(target),sha256=sha(target),split=split,tables=len(a['truth']),arms=ARMS,alphas=ALPHAS,locks=locks,calibration=binding,calibration_states=[c.state() for c in calibrators],seconds=time.perf_counter()-started,point_definition='Unadjusted full-n canonical single or consensus nuisance EIF; point_index0/1; no point correction by CI head',policy_cost='All view features already computed; no early-exit speed claim',all_failed_attempts_retained=True)
        pt.save(ledger,record);completed.append(record);print('APPLIED',name,len(a['truth']),len(ARMS),flush=True)
    suffix='' if args.task_count==1 else f'.part-{args.task_index}-of-{args.task_count}'
    pt.save(output/('manifest'+suffix+'.json'),dict(protocol_sha256=protocol_hash,checkpoint_sha256=checkpoint_hash,posttrain_artifacts_sha256=sha(Path(args.posttrain_dir)/'artifacts.json'),groups=completed))
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--cache-manifest',required=True);p.add_argument('--posttrain-dir',required=True);p.add_argument('--calibration-cache',required=True);p.add_argument('--output',required=True);p.add_argument('--authorization');p.add_argument('--task-index',type=int,default=0);p.add_argument('--task-count',type=int,default=1);args=p.parse_args()
    if not 0<=args.task_index<args.task_count:raise ValueError('Invalid shard')
    run(args)
if __name__=='__main__':main()
