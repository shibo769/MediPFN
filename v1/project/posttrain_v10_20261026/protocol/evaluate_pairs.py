"""Failure-inclusive, paired frozen-model evaluation on v10 DEV or released TEST.

All prediction functions consume observables only; truth is used only after all
intervals have been computed. Multiple backbone variants must share identities.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
V10=HERE.parent
ROOT=V10.parent
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(V10/'tabcon'))
import mixture
import posttrain_heuristics as heur
import apply_posttrain as old
from evaluate import EvaluationGuard,EmpiricalErrorCalibration,clean
from merge_cache import entries,load_entry,sha

ALPHA=.05
Z=1.959963984540054
FIELDS=('x_single','x_multi','estimates','ses','truth','mechanism_id','seeds','fold_seeds','split','bank','family')
METHODS=('global','sft','matched_sft','dpo','rlvr_only','rlvr')

def read_manifest(path,protocol,split):
    h,groups=entries(path)
    if h.get('protocol_sha256')!=sha(protocol):raise ValueError('cache protocol mismatch')
    if not groups or any(g['split']!=split for g in groups):raise ValueError('cache split mismatch')
    values={k:[] for k in FIELDS}; flags=[]
    for group in groups:
        _,a=load_entry(group,path,sha(protocol));n=len(a['truth'])
        for k in FIELDS:
            if k not in a:raise ValueError('Missing cache field '+k)
            values[k].append(a[k])
        failed=np.asarray(a.get('failed',np.full(n,''))).astype(str)!=''
        if 'success' in a:failed|=~np.asarray(a['success'],bool)
        flags.append(failed)
    return {**{k:np.concatenate(v) for k,v in values.items()},'cache_failed':np.concatenate(flags)},h

def stats_from_intervals(est,se,ci,valid):
    n=len(est);out=np.full((n,6),np.nan);out[:,2]=np.inf;out[:,3]=-np.inf;out[:,5]=1
    valid=np.asarray(valid,bool)&np.isfinite(est)&np.isfinite(se)&(se>0)&np.isfinite(ci).all(1)&(ci[:,0]<=ci[:,1])
    out[valid,0]=est[valid];out[valid,1]=se[valid];out[valid,2:4]=ci[valid];out[valid,4]=0;out[valid,5]=0
    return out

def valid_input(a):
    return (~a['cache_failed'] & np.isfinite(a['x_multi']).all(1) &
            np.isfinite(a['estimates'][:,1]) & np.isfinite(a['ses'][:,1]) & (a['ses'][:,1]>0))

@torch.no_grad()
def predict(a,heads,routers,mean,std,top2,old_head=None,old_norm=None,calibration_quantiles=None,device='cpu',batch=512):
    """Prediction takes no truth, law ID, family, seed, or simulation labels."""
    n=len(a['estimates']);valid=valid_input(a);ix=np.flatnonzero(valid)
    est=np.asarray(a['estimates'][:,1],float);se=np.asarray(a['ses'][:,1],float)
    predictions={}
    ci=np.stack((est-Z*se,est+Z*se),-1);predictions['wald']=stats_from_intervals(est,se,ci,valid)
    if calibration_quantiles is not None:
        qlo,qhi=calibration_quantiles
        ci=np.stack((est-se*qhi,est-se*qlo),-1)
        predictions['wald_calibrated']=stats_from_intervals(est,se,ci,valid)
    if old_head is not None:
        p=heur.predict_error_head(old_head,old_norm,a['x_multi'],est,se,ALPHA,device)
        predictions['old_consistency']=stats_from_intervals(p['estimate'],se,p['intervals'],valid&p['valid'])
    result={k:np.full((n,6),np.nan) for k in METHODS}
    for k in METHODS:result[k][:,2]=np.inf;result[k][:,3]=-np.inf;result[k][:,5]=1
    for start in range(0,len(ix),batch):
        ids=ix[start:start+batch];x=torch.as_tensor((a['x_multi'][ids]-mean)/std,dtype=torch.float32,device=device)
        parts=mixture.parts_at(heads,x)
        for name in METHODS:
            part=parts[0] if name=='global' else mixture.mixture(parts,routers[name](x),top2)
            intervals=mixture.intervals(part,est[ids],se[ids]);w,m,_=part
            corrected=est[ids]-se[ids]*(w*m).sum(-1).cpu().numpy()
            result[name][ids]=stats_from_intervals(corrected,se[ids],intervals,np.ones(len(ids),bool))
    predictions.update(result)
    return predictions

def metric(stats,truth,ids):
    s=stats[ids];y=truth[ids];valid=(s[:,5]==0)&np.isfinite(s[:,:4]).all(1)&(s[:,1]>0)&(s[:,2]<=s[:,3])
    cover=valid&(s[:,2]<=y)&(y<=s[:,3]);length=np.full(len(y),np.inf);score=length.copy()
    length[valid]=s[valid,3]-s[valid,2]
    score[valid]=length[valid]+2/ALPHA*(np.maximum(s[valid,2]-y[valid],0)+np.maximum(y[valid]-s[valid,3],0))
    finite_total=float(length.sum()); k=max(1,int(np.ceil(.01*len(y))))
    top_share=float(np.sort(length)[-k:].sum()/finite_total) if np.isfinite(finite_total) and finite_total>0 else None
    return dict(attempts=len(y),failures=int((~valid).sum()),coverage=float(cover.mean()),
                mean_length=float(length.mean()),interval_score=float(score.mean()),
                length_p95=float(np.quantile(length,.95,method='higher')),
                length_p99=float(np.quantile(length,.99,method='higher')),length_top_1pct_share=top_share,
                valid_mean_length=float(length[valid].mean()) if valid.any() else None,
                valid_interval_score=float(score[valid].mean()) if valid.any() else None)

def paired_coverage(left,right,truth,ids,law):
    a=left[ids];b=right[ids];y=truth[ids]
    ca=(a[:,5]==0)&(a[:,2]<=y)&(y<=a[:,3]);cb=(b[:,5]==0)&(b[:,2]<=y)&(y<=b[:,3])
    d=ca.astype(float)-cb.astype(float);unique,inv=np.unique(law[ids],return_inverse=True);g=len(unique)
    residual=np.bincount(inv,weights=d-d.mean());se=float(np.sqrt(g/(g-1)*np.sum(residual**2))/len(d)) if g>1 else None
    return dict(difference=float(d.mean()),cluster_mc_se=se,independent_laws=g)

def evaluate(variants,old_head=None,old_norm=None,calibration_quantiles=None,device='cpu'):
    base=variants[0][1];n=len(base['truth'])
    for name,a,_,_,_,_,_ in variants[1:]:
        for key in ('truth','mechanism_id','seeds','fold_seeds','family','bank'):
            if not np.array_equal(base[key],a[key]):raise ValueError('Paired variants differ on '+key+': '+name)
    outputs={}
    for index,(name,a,heads,routers,mean,std,top2) in enumerate(variants):
        obs={k:a[k] for k in ('x_multi','estimates','ses','cache_failed')}
        p=predict(obs,heads,routers,mean,std,top2,old_head if index==0 else None,old_norm if index==0 else None,
                  calibration_quantiles if index==0 else None,device)
        outputs.update({name+'_'+k:v for k,v in p.items()})
    truth=base['truth'];family=base['family'].astype(str);bank=base['bank'].astype(str);law=base['mechanism_id'].astype(str)
    groups={'all':np.ones(n,bool),'ID':np.isin(family,['linear','smooth','heterosked','weak_overlap','covariate_shift']),
            'selection_shift':np.isin(family,['tree','rff']),'blind_shift':np.isin(family,['mlp','combinations'])}
    groups.update({str(f):family==f for f in np.unique(family)})
    rows=[];paired=[];ref=next(iter(outputs.values()))
    # Keep repeated-law online panel distinct from independent test cohort.
    for b in np.unique(bank):
        for group,mask in groups.items():
            ids=np.flatnonzero(mask&(bank==b))
            if not len(ids):continue
            for name,stats in outputs.items():
                rows.append(dict(bank=str(b),group=group,method=name,**metric(stats,truth,ids)))
                paired.append(dict(bank=str(b),group=group,method=name,reference=next(iter(outputs)),
                                   **paired_coverage(stats,ref,truth,ids,law)))
    return outputs,rows,paired

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--split',choices=('dev','test'),required=True)
    p.add_argument('--variant',nargs=3,metavar=('NAME','CACHE_MANIFEST','MIXTURE_MODEL'),action='append',required=True)
    p.add_argument('--core',required=True);p.add_argument('--heur',required=True);p.add_argument('--output',required=True)
    p.add_argument('--calibration',required=True,help='Independent v10 calibration cache manifest for the first backbone')
    p.add_argument('--authorization');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads)
    guard=EvaluationGuard(a.protocol,a.authorization);guard.check(a.split)
    if a.split=='test':
        gate=json.loads(Path(a.authorization).read_text());bound=gate.get('evaluation_variants',{})
        existing={name:(manifest,model) for name,manifest,model in a.variant}
        for name,base in (('original_top2','original'),('decoder_top2','decoder')):
            if name in bound and name not in existing:
                a.variant.append([name,existing[base][0],bound[name]['mixture_path']])
        if gate.get('selection_sha256')!=sha(gate['selection_path']):raise ValueError('Frozen DEV selection missing/changed')
        if gate.get('evaluate_source_sha256')!=sha(__file__) or gate.get('mixture_source_sha256')!=sha(V10/'tabcon/mixture.py'):
            raise ValueError('Evaluation or mixture source changed after release')
        for name,manifest,model in a.variant:
            if bound.get(name,{}).get('mixture_sha256')!=sha(model):raise ValueError('Model absent from test release')
        if gate.get('old_consistency_sha256')!=sha(Path(a.heur)/'consistency_1.0.pt'):raise ValueError('Baseline absent from test release')
        if gate.get('old_normalization_sha256')!=sha(Path(a.core)/'normalization.npz'):raise ValueError('Normalization absent from test release')
        if gate.get('calibration_manifest_sha256')!=sha(a.calibration):raise ValueError('Calibration absent from test release')
    # Load and verify frozen parent artifacts before using consistency baseline.
    _,old_ids,old_seeds,_=old.verify_posttrain(a.core);heur.verify_candidate(a.heur,a.core)
    head=heur.load_head(Path(a.heur)/'consistency_1.0.pt',a.device)
    with np.load(Path(a.core)/'normalization.npz',allow_pickle=False) as z:
        norm={k:(z[k+'_center'],z[k+'_scale']) for k in ('x_single','x_multi')}
    variants=[];bindings={};models=[]
    for name,manifest,model in a.variant:
        cache,h=read_manifest(manifest,a.protocol,a.split)
        heads,routers,mean,std,meta=mixture.load(model,a.device)
        if a.split=='test' and gate['evaluation_variants'][name]['checkpoint_sha256']!=h['checkpoint_sha256']:
            raise ValueError('backbone absent from release')
        variants.append((name,cache,heads,routers,mean,std,meta['arguments']['top2']))
        models.append(meta)
        bindings[name]=dict(manifest_sha256=sha(manifest),checkpoint_sha256=h['checkpoint_sha256'],mixture_sha256=sha(model))
    cal,ch=read_manifest(a.calibration,a.protocol,'calibration')
    if ch['checkpoint_sha256']!=bindings[a.variant[0][0]]['checkpoint_sha256']:raise ValueError('calibration backbone mismatch')
    calids=set(cal['mechanism_id'].astype(str));calseeds=set(map(int,cal['seeds']))|set(map(int,cal['fold_seeds']))
    if calids&old_ids or calseeds&old_seeds:raise ValueError('calibration overlaps frozen parent training')
    for _,cache,_,_,_,_,_ in variants:
        if calids&set(cache['mechanism_id'].astype(str)) or calseeds&(set(map(int,cache['seeds']))|set(map(int,cache['fold_seeds']))):
            raise ValueError('calibration overlaps evaluation')
    for meta,variant in zip(models,a.variant):
        training=Path(meta['arguments']['cache'])
        if not training.exists() or sha(training)!=meta['cache_sha256']:raise ValueError('mixture training cache missing/changed')
        with np.load(training,allow_pickle=False) as z:
            mids=set(z['mechanism_id'].astype(str));seeds=set(map(int,z['seeds']))|set(map(int,z['fold_seeds']))
            if str(z['checkpoint_sha256'].item())!=bindings[variant[0]]['checkpoint_sha256']:raise ValueError('mixture train/evaluation backbone mismatch')
        if calids&mids or calseeds&seeds:raise ValueError('calibration overlaps mixture training/dev')
    records=[]
    for i in range(len(cal['truth'])):
        est=float(cal['estimates'][i,1]);se=float(cal['ses'][i,1]);bad=bool(cal['cache_failed'][i]) or not (np.isfinite(est) and np.isfinite(se) and se>0)
        records.append(dict(split='calibration',mechanism_id=str(cal['mechanism_id'][i]),truth=float(cal['truth'][i]),summary=dict(estimate=est,se=se,failed=bad)))
    quantiles=EmpiricalErrorCalibration().fit(records).quantiles(ALPHA)
    outputs,rows,paired=evaluate(variants,head,norm,quantiles,a.device)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);base=variants[0][1]
    with (out/'per_table.npz').open('xb') as f:
        np.savez_compressed(f,**outputs,truth=base['truth'],family=base['family'],bank=base['bank'],
                            mechanism_id=base['mechanism_id'],seeds=base['seeds'],fold_seeds=base['fold_seeds'])
    report=dict(split=a.split,protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),variants=bindings,
                mixture_source_sha256=sha(V10/'tabcon/mixture.py'),
                calibration_manifest_sha256=sha(a.calibration),calibration_quantiles=quantiles,
                old_consistency_sha256=sha(Path(a.heur)/'consistency_1.0.pt'),
                old_normalization_sha256=sha(Path(a.core)/'normalization.npz'),per_table_sha256=sha(out/'per_table.npz'),
                results=rows,paired_vs_first_wald=paired,all_attempts_retained=True,
                note='Repeat-law rows are clustered for paired SE; all means include failures as infinite length/score.')
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(out/'report.json'),tables=len(base['truth']),variants=list(bindings))))

if __name__=='__main__':main()
