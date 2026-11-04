"""CPU shared evaluation science; isolated per-shard output only."""
import hashlib,json,os,types,traceback
from pathlib import Path
import numpy as np
import torch

def digest_array(array):
    array=np.ascontiguousarray(array)
    return {'sha256':hashlib.sha256(array.tobytes()).hexdigest(),'dtype':str(array.dtype),'shape':list(array.shape)}

def _imports(protocol):
    path=Path(protocol['science_imports_source']).resolve(strict=True);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=protocol['source_sha256'].get(str(path)):raise RuntimeError('science import helper changed')
    module=types.ModuleType('cpu_frozen_imports');module.__file__=str(path);exec(compile(raw,str(path),'exec'),module.__dict__);return module

def run(context):
    protocol=context['manifest'];shard=context['shard'];output=Path(context['output_dir']).resolve(strict=True)
    if context['stage']!='evaluation' or protocol.get('scope')!='4m_only':raise RuntimeError('wrong science stage')
    helper=_imports(protocol);helper.verify_calibration_audit(protocol);cases=helper.selected_cases(protocol,'evaluation',shard,'test')
    if os.environ.get('NSLOTS'):torch.set_num_threads(int(os.environ['NSLOTS']))
    covariate_root=str(Path(protocol['covariate_root']).resolve(strict=True))
    if not Path(covariate_root).is_relative_to(Path(protocol['output_root']).resolve(strict=True)):raise RuntimeError('covariates outside CPU study')
    os.environ['NIE_WAVE_XDIR']=covariate_root
    modules=helper.load_registered(protocol,'evaluation');cal=modules['calibration'];mixture=modules['mixture'];quantile=modules['posttrain'].quantile
    crossfit_views=modules['inference'].crossfit_views;load_checkpoint=modules['model'].load_checkpoint;training=modules['robust_retrain_train_seedfix_v2'];loader=modules['cpu_case_loader'];ablation=modules['ablation95'];levels=modules['alpha_inference']
    if tuple(ablation.METHODS)!=tuple(protocol['methods_95']):raise RuntimeError('95 methods differ from protocol')
    back=load_checkpoint(helper.asset(protocol,'backbone'),'cpu').eval()
    heads,routers,mean,std,_=mixture.load(helper.asset(protocol,'mixture'),'cpu')
    cm=torch.load(helper.asset(protocol,'confidence'),map_location='cpu',weights_only=False)
    confidence=cal.Confidence(cm['dim']);confidence.load_state_dict(cm['state']);confidence.eval()
    chosen=torch.load(helper.asset(protocol,'chosen'),map_location='cpu',weights_only=False)
    selection=torch.load(helper.asset(protocol,'selection'),map_location='cpu',weights_only=False)
    if chosen['selected']!='confidence':
        head=training.Head(len(selection['normalization_mean']));head.load_state_dict(selection['states'][chosen['selected']]);head.eval();chosen['head']=head
    levelpath=helper.asset(protocol,'levels_freeze');confidencepath=helper.asset(protocol,'confidence')
    level_context=levels.load_context(levelpath,protocol['artifact_sha256'][levelpath],confidencepath,protocol['artifact_sha256'][confidencepath],cal)
    level_freeze=json.loads(Path(levelpath).read_text())
    for rec in level_freeze['models'].values():
        path=str(Path(rec['head_path']).resolve())
        if protocol['artifact_sha256'].get(path)!=rec['head_sha256']:raise RuntimeError('protocol omits level-specific head')
    rows=[]
    for case in cases:
        row={'case':case}
        try:
            if case.get('testbank_status')=='failed':raise RuntimeError('registered shared test table generation failed')
            if case.get('testbank_status')!='ok':raise RuntimeError('missing registered testbank status')
            observed,fold,truth,pop=loader.make_case(case)
            truth=float(truth)
            if not np.isfinite(truth):raise RuntimeError('nonfinite independent truth')
            with np.load(case['observed_path'],allow_pickle=False) as table:
                if set(table.files)!={'observed'}:raise RuntimeError('registered table schema mismatch')
                saved=np.asarray(table['observed'])
            if saved.dtype!=np.asarray(observed).dtype or saved.shape!=np.asarray(observed).shape or not np.array_equal(saved,observed):raise RuntimeError('observed table differs from registered shared input')
            if digest_array(fold)!=case['fold_identity']:raise RuntimeError('fold differs from preregistered identity')
            row.update({'observation':digest_array(observed),'fold':digest_array(fold),'population_sha256':pop,'truth':truth})
            with torch.no_grad():
                out=crossfit_views(back,observed,fold,seed=int(case['view_seed']),context_size=256,query_batch=128,device='cpu')
            a={'x_multi':np.asarray(out['x_multi'])[None,:], 'estimates':np.asarray(out['estimates'])[None,:],
               'ses':np.asarray(out['ses'])[None,:], 'cache_failed':np.array([False])}
            est=np.asarray(a['estimates'])[:,1];se=np.asarray(a['ses'])[:,1]
            raw_valid=(np.isfinite(a['x_multi']).all() and np.isfinite(est).all()
                       and np.isfinite(se).all() and bool(se[0]>0))
            if not raw_valid:raise RuntimeError('invalid nuisance/point/scale')
            x=q=valid=None;anchor_error=None
            try:
                x,q,est,se,valid=cal.make_features(a,heads,routers['sft'],mean,std)
                if not bool(valid[0]):raise ValueError('invalid original 95% anchor')
                methods=ablation.evaluate(x,q,est,se,valid,cal=cal,training=training,confidence=confidence,
                     confidence_state=cm,selection=selection,chosen=chosen)
                row['original_q95']=q.tolist()
            except Exception as exc:
                anchor_error=repr(exc)
                methods={name:{'ok':False,'error':anchor_error} for name in protocol['methods_95']}
            level_valid=np.array([True])
            more={}
            # Each post-trained nominal level depends on q95 and its own frozen
            # head, not on the SFT target-level inverse CDF.  Evaluate it even if
            # a different method's quantile calculation fails.
            for level in (.90,.99):
                name='posttrain_'+str(int(round(100*level)))
                try:
                    if anchor_error is not None:raise RuntimeError('95% anchor unavailable: '+anchor_error)
                    lo,hi=levels.predict_one_level(x,q,est,se,valid,None,level_context,level,'posttrain')
                    more[name]={'ok':True,'lo':float(lo[0]),'hi':float(hi[0])}
                except Exception as exc:
                    more[name]={'ok':False,'error':repr(exc)}
            try:
                z=torch.as_tensor(np.nan_to_num((a['x_multi']-mean)/std),dtype=torch.float32)
                with torch.no_grad():dist=mixture.mixture(mixture.parts_at(heads,z),routers['sft'](z),True)
            except Exception as exc:
                dist=None;row['cdf95_parity']={'status':'NOT_EVALUABLE','error':repr(exc)}
            if dist is not None:
                try:
                    if anchor_error is not None:raise ValueError('95% anchor unavailable: '+anchor_error)
                    with torch.no_grad():qcheck=np.stack([quantile(dist,t).numpy() for t in (.025,.5,.975)],1)
                    if not np.isfinite(qcheck).all():raise ValueError('nonfinite CDF 95% parity quantiles')
                    delta=float(np.max(np.abs(qcheck-q)))
                    row['cdf95_parity']={'status':'PASS' if delta<=1e-6 else 'FAIL','max_abs':delta}
                except Exception as exc:
                    row['cdf95_parity']={'status':'NOT_EVALUABLE','error':repr(exc)}
            for level,alpha in ((.90,.05),(.99,.005)):
                name='sft_'+str(int(round(100*level)))
                try:
                    if dist is None:raise RuntimeError('original CDF unavailable')
                    with torch.no_grad():
                        target=np.stack((quantile(dist,alpha).numpy(),quantile(dist,1-alpha).numpy()),1)
                    if not np.isfinite(target).all():raise ValueError('nonfinite target-level quantiles')
                    lo,hi=levels.predict_one_level(None,None,est,se,level_valid,target,level_context,level,'sft')
                    more[name]={'ok':True,'lo':float(lo[0]),'hi':float(hi[0])}
                    row['sft_quantiles_'+str(int(round(100*level)))]=target.tolist()
                except Exception as exc:
                    more[name]={'ok':False,'error':repr(exc)}
            if set(more)!=set(protocol['methods_levels']):raise RuntimeError('level method identifiers mismatch')
            if set(methods)&set(more):raise RuntimeError('overlapping registered method identifiers')
            methods.update(more)
            if set(methods)!=set(protocol['allow_methods']):raise RuntimeError('registered methods mismatch')
            for name,item in methods.items():
                if item['ok']:
                    try:
                        lo=float(item['lo']);hi=float(item['hi'])
                        if not np.isfinite([lo,hi]).all() or lo>hi:raise ValueError('nonfinite or reversed interval')
                        item['covered']=bool(lo<=truth<=hi);item['length']=hi-lo
                    except Exception as exc:methods[name]={'ok':False,'error':repr(exc)}
            row.update({'point':float(est[0]),'standard_error':float(se[0]),'methods':methods,'ok':True})
        except Exception as exc:
            row.update({'ok':False,'error':repr(exc),'traceback':traceback.format_exc()[-4000:]})
        rows.append(row)
    target=output/'evaluation.json'
    encoded=json.dumps({'protocol_sha256':context['manifest_sha256'],'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'shard':shard,'rows':rows},allow_nan=False).encode('utf8')
    with target.open('xb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
    return {'records':len(rows),'common_failures':sum(not row['ok'] for row in rows),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'device':'cpu'}

if __name__=='__main__':raise RuntimeError('requires frozen CPU stage and reviewed inputs')
