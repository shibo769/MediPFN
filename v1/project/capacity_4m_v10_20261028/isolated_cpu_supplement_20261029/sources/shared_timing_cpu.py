"""Prospective paired per-method end-to-end timing; never used for selection.

Inputs, methods and warmup/measurement table identities must be preregistered.
Each timed call independently rebuilds its folds and reruns the frozen PFN.
Observed-data generation, on-disk I/O and initial model loading are excluded.
"""
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

B=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9')
D=B/'capacity_4m_v10_20261028';W=D/'isolated_cpu_supplement_20261029/wave'
C=D/'confidence_calibration_4m_20261028';R=D/'robust_partial_composition_retry_20261028'
def cpu_hardware():
    model = ''
    try:
        for line in Path('/proc/cpuinfo').read_text(encoding='utf8').splitlines():
            if line.partition(':')[0].strip().lower() == 'model name':
                model = line.partition(':')[2].strip()
                break
    except OSError:
        pass
    model = model or platform.processor().strip() or 'unavailable'
    slots = int(os.environ['NSLOTS']) if os.environ.get('NSLOTS') else None
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {'device':'cpu','cpu_model':model,'cpu_threads':int(torch.get_num_threads()),
            'allocated_slots':slots,'affinity_cpu_count':affinity}

METHODS=('sft','confidence_temperature','quantile_uncalibrated','selected_calibrated',
         'sft_90','posttrain_90','sft_99','posttrain_99')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()

def main():
    torch.set_num_threads(int(os.environ['NSLOTS'])) if os.environ.get('NSLOTS') else None
    # The protocol controls the path and source SHA of the common validator.
    # Its own digest is independently checked before its code is imported.
    p=W/'protocol.json';expected=os.environ['PROTOCOL_SHA256']
    if sha(p)!=expected:raise RuntimeError('protocol SHA mismatch')
    proto=json.loads(p.read_text(encoding='utf8'))
    worker_path=Path(proto['eval_worker_path']).resolve()
    if proto['source_sha256'].get(str(worker_path))!=sha(worker_path):raise RuntimeError('eval worker SHA mismatch')
    spec=importlib.util.spec_from_file_location('wave_eval_validator',str(worker_path))
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    proto=worker.verify(expected,'pfn_timing')
    expected_xdir=str(Path(proto['covariate_root']).resolve())
    if os.environ.get('NIE_WAVE_XDIR',expected_xdir)!=expected_xdir:
        raise RuntimeError('unexpected covariate root override')
    os.environ['NIE_WAVE_XDIR']=expected_xdir
    if proto['source_sha256'].get(str(Path(__file__).resolve()))!=sha(__file__):raise RuntimeError('timing source mismatch')
    if tuple(proto['timing_methods'])!=METHODS:raise RuntimeError('timing method identity mismatch')
    case_indices=proto['timing_case_indices']; warmup_count=int(proto['timing_warmup_count'])
    if len(case_indices)!=13 or warmup_count!=3:raise RuntimeError('timing design must be fixed before evaluation')
    cases=[proto['tasks'][int(i)] for i in case_indices]
    if any(int(c['index'])!=int(i) for c,i in zip(cases,case_indices)):raise RuntimeError('case index mismatch')
    worker.preflight_imports(proto)
    import calibration as cal
    import mixture
    from posttrain import quantile
    from inference import crossfit_views
    from data import make_folds
    from model import load_checkpoint
    import robust_retrain_train_seedfix_v2 as training
    loader=worker.module_from(proto['case_loader_path'],'timing_loader',proto['source_sha256'])
    ablation=worker.module_from(proto['ablation95_path'],'timing_ablation',proto['source_sha256'])
    levels=worker.module_from(proto['levels_path'],'timing_levels',proto['source_sha256'])
    worker.check_project_imports(proto)
    back=load_checkpoint(str(B/'capacity_full/large_seed0/checkpoint/model.pt'),'cpu').eval()
    heads,routers,mean,std,_=mixture.load(D/'mixture_top2/mixture.pt','cpu')
    cm=torch.load(C/'fitted/confidence.pt',map_location='cpu',weights_only=False)
    conf=cal.Confidence(cm['dim']);conf.load_state_dict(cm['state']);conf.eval()
    chosen=torch.load(R/'trained_v1/model.pt',map_location='cpu',weights_only=False)
    sel=torch.load(R/'trained_v1/selection.pt',map_location='cpu',weights_only=False)
    if chosen['selected']!='confidence':
        head=training.Head(len(sel['normalization_mean']));head.load_state_dict(sel['states'][chosen['selected']]);head.eval();chosen['head']=head
    lc=levels.load_context(proto['levels_freeze_path'],proto['artifact_sha256'][proto['levels_freeze_path']],
                           str(C/'fitted/confidence.pt'),proto['artifact_sha256'][str(C/'fitted/confidence.pt')],cal)
    worker.check_project_imports(proto)
    records=[]
    for i,case in enumerate(cases):
        observed,reference_fold,truth,pop=loader.make_case(case)
        order=list(METHODS[i%len(METHODS):]+METHODS[:i%len(METHODS)])
        for method in order:
            start=time.perf_counter()
            fold=make_folds(int(case['n']),int(case['fold_seed']))
            if not np.array_equal(fold,reference_fold):raise RuntimeError('fold reconstruction mismatch')
            with torch.no_grad():
                v=crossfit_views(back,observed,fold,seed=int(case['view_seed']),context_size=256,query_batch=128,device='cpu')
            a={'x_multi':np.asarray(v['x_multi'])[None,:],'estimates':np.asarray(v['estimates'])[None,:],
               'ses':np.asarray(v['ses'])[None,:],'cache_failed':np.array([False])}
            if method.startswith('sft'):
                est=np.asarray(a['estimates'])[:,1];se=np.asarray(a['ses'])[:,1]
                z=torch.as_tensor(np.nan_to_num((a['x_multi']-mean)/std),dtype=torch.float32)
                with torch.no_grad():
                    dist=mixture.mixture(mixture.parts_at(heads,z),routers['sft'](z),True)
                    al=.025 if method=='sft' else (.05 if method=='sft_90' else .005)
                    qlo=quantile(dist,al).numpy();qhi=quantile(dist,1-al).numpy()
                low=est-se*qhi;high=est-se*qlo
            else:
                x,q,est,se,valid=cal.make_features(a,heads,routers['sft'],mean,std)
                if not bool(valid[0]):raise RuntimeError('invalid timing feature')
                if method in METHODS[:4]:
                    out=ablation.evaluate(x,q,est,se,valid,cal=cal,training=training,confidence=conf,
                        confidence_state=cm,selection=sel,chosen=chosen,only=method)[method]
                    if not out['ok']:raise RuntimeError('timing method failed '+method+' '+repr(out))
                    low=np.array([out['lo']]);high=np.array([out['hi']])
                else:
                    level=.90 if method=='posttrain_90' else .99
                    low,high=levels.predict_one_level(x,q,est,se,valid,None,lc,level,'posttrain')
            if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>high):
                raise RuntimeError('invalid timed interval '+method)
            elapsed=time.perf_counter()-start
            records.append({'case_index':int(case['index']),'kind':case['kind'],'scenario':case['scenario'],
                'n':int(case['n']),'p':int(case['p']),'method':method,'warmup':i<warmup_count,
                'order':order.index(method),'seconds':elapsed,'lo':float(low[0]),'hi':float(high[0]),
                'observation':worker.digest_array(observed),'fold':worker.digest_array(fold),'population_sha256':pop})
    result={'protocol_sha256':expected,'source_sha256':sha(__file__),'hardware':cpu_hardware(),
        'scope':'single-process per-method full crossfit, fold generation, EIF, required CDF/heads and interval; CPU execution; rotating order; excludes model load, data generation, disk I/O',
        'records':records,'median_seconds':{m:float(np.median([r['seconds'] for r in records if r['method']==m and not r['warmup']])) for m in METHODS}}
    dest=W/'eval'/'timing.json'
    with dest.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'TIMING_WRITTEN','sha256':sha(dest)}),flush=True)

if __name__=='__main__': raise RuntimeError('CPU candidate is not authorized for production; obtain a new reviewed protocol')
