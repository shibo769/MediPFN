"""Immutable contracts shared by feature fitting, post-training and deployment."""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib,json,math,os
from pathlib import Path
import numpy as np

ROUTES=('frozen_v3','adapted_v3')
VARIANTS=('head_only','confidence_only','balanced_correction_only','full','full_unbalanced')
PHASE_COUNTS=dict(error_train=10000,error_dev=1024,post_train=1024,post_dev=512,calibration=1024)
TRAIN_SOURCES={'cheng':'cheng','diaz_hejazi':'diaz_hejazi','diaz_intermediate':'diaz_intermediate',
 'he_song_xu':'he_song_xu','jones_correlated':'jones','jones_lnl':'jones','jones_lnn':'jones',
 'miles':'miles','shan':'shan','zuo':'zuo'}
LEVELS=(90,95,99)
TAUS=(.005,.025,.05,.95,.975,.995)
PAIR_INDICES=((2,3),(1,4),(0,5))
CACHE_FORMAT='medipfn-v3-matched-feature-cache-1'
MODEL_FORMAT='medipfn-v3-matched-posttraining-1'
CAL_FORMAT='medipfn-v3-matched-independent-calibration-1'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8388608),b''):h.update(chunk)
    return h.hexdigest()

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def write_json_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def assert_contract(left,right):
    if canonical(left)!=canonical(right):raise ValueError('Frozen point/feature/scale contract differs')

def observed_hash(observed):
    allowed={'x','a','m','y','x_types','mediator_types'}
    if set(observed)!=allowed:raise ValueError('Inference requires exactly observed arrays/type fields')
    h=hashlib.sha256()
    for key in ('x','a','m','y','x_types','mediator_types'):
        arr=np.ascontiguousarray(observed[key])
        header=canonical(dict(key=key,dtype=arr.dtype.str,shape=list(arr.shape))).encode();data=arr.tobytes()
        h.update(len(header).to_bytes(8,'big'));h.update(header);h.update(len(data).to_bytes(8,'big'));h.update(data)
    return h.hexdigest()

def validate_rows(rows,phase):
    if phase not in PHASE_COUNTS:raise ValueError('Unknown fitting/calibration phase')
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Duplicate cache case IDs')
    for row in rows:
        if row.get('phase')!=phase:raise ValueError('Feature row phase differs')
        if TRAIN_SOURCES.get(row['source_id'])!=row['source_group']:raise ValueError('Held-out/unregistered source entered fitting')
        if int(row['n']) not in (200,600,1000):raise ValueError('Unregistered fitting sample size')
        for key in ('case_id','mechanism_id','mechanism_block_id','cluster_id','observed_sha256','observation_seed_sha256','fold_assignment_sha256'):
            if not isinstance(row[key],str) or not row[key]:raise ValueError('Missing immutable row identity: '+key)

def read_cache(path,phase=None):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if meta.get('format')!=CACHE_FORMAT or meta.get('status')!='COMPLETE':raise ValueError('Incomplete/unrecognized feature cache')
    if meta['npz_sha256']!=sha256(path):raise ValueError('Feature cache bytes changed')
    if phase is not None and meta['phase']!=phase:raise ValueError('Wrong feature-cache stage')
    if meta['route'] not in ROUTES:raise ValueError('Unknown point backbone')
    if meta['contract'].get('route')!=meta['route']:raise ValueError('Cache point-model route binding differs')
    with np.load(path,allow_pickle=False) as source:arrays={key:source[key].copy() for key in source.files}
    if set(arrays)!={'features','response','estimate','truth','scale','standard_error'}:raise ValueError('Unknown/missing cache array')
    count=len(meta['rows']);validate_rows(meta['rows'],meta['phase'])
    if count!=meta['cases'] or arrays['features'].shape!=(count,1038):raise ValueError('Feature dimensions/row denominator differ')
    for key,value in arrays.items():
        if len(value)!=count or (key!='features' and value.shape!=(count,)) or not np.isfinite(value).all():
            raise ValueError('Cache array shape/finiteness differs: '+key)
    if np.any(arrays['scale']<=0) or np.any(arrays['standard_error']<0):raise ValueError('Invalid score/interval scale')
    expected=np.maximum(arrays['standard_error'],float(meta['contract']['standard_error_floor']))
    if not np.allclose(expected,arrays['scale'],rtol=1e-12,atol=1e-12):raise ValueError('Scale contract mismatch')
    response=(arrays['estimate']-arrays['truth'])/arrays['scale']
    if not np.allclose(response,arrays['response'],rtol=2e-6,atol=2e-6):raise ValueError('Signed-error label inconsistent with estimate/truth/scale')
    return arrays,meta

def assert_fixture_mode(metadata,requested):
    if bool(metadata.get('fixture',False))!=bool(requested):raise ValueError('Fixture/production cache use differs')

def save_cache(path,arrays,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError('Feature assets are immutable')
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    metadata=dict(metadata,format=CACHE_FORMAT,status='COMPLETE',npz_sha256=sha256(path))
    write_json_new(path.with_suffix('.json'),metadata)
    read_cache(path,metadata['phase'])
    return metadata

def require_disjoint(*metadata):
    seen={key:set() for key in ('case_id','mechanism_block_id','observed_sha256','observation_seed_sha256')}
    for meta in metadata:
        for key in seen:
            values={r[key] for r in meta['rows']}
            if seen[key]&values:raise ValueError('Task/table/seed-block overlap across stages: '+key)
            seen[key]|=values

def balanced_weights(rows):
    presets=defaultdict(set);grouped=defaultdict(set);counts=Counter()
    for row in rows:
        group=row['source_group'];preset=row['source_id'];block=row['mechanism_block_id']
        presets[group].add(preset);grouped[(group,preset)].add(block);counts[(group,preset,block)]+=1
    if not grouped:raise ValueError('Empty post-training task bank')
    return np.asarray([1/(len(presets)*len(presets[r['source_group']])*len(grouped[(r['source_group'],r['source_id'])])*
      counts[(r['source_group'],r['source_id'],r['mechanism_block_id'])]) for r in rows],np.float64)

def torch_save_new(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Checkpoint is immutable: '+str(path))
    # Same-directory temporary file, atomic publication. Protected isolated run only.
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:torch.save(value,stream)
    os.replace(temporary,path)

def check_pairs(pairs):
    p=np.asarray(pairs)
    if p.ndim!=3 or p.shape[1:]!=(3,2) or not np.isfinite(p).all():raise ValueError('Invalid/nonnumeric quantile pairs')
    if np.any(p[:,:,0]>p[:,:,1]) or np.any(np.diff(p[:,:,0],axis=1)>1e-6) or np.any(np.diff(p[:,:,1],axis=1)<-1e-6):
        raise ValueError('Crossed or nonnested uncertainty quantiles')

def interval_endpoints(estimate,scale,pairs,corrections):
    pairs=np.asarray(pairs,float);corrections=np.asarray(corrections,float)
    check_pairs(pairs)
    if corrections.shape!=(3,) or np.any(corrections<0) or np.any(np.diff(corrections)<0):raise ValueError('Invalid calibration expansions')
    estimate=np.asarray(estimate,float);scale=np.asarray(scale,float)
    if scale.shape!=estimate.shape or scale.shape!=(len(pairs),) or np.any(scale<=0):raise ValueError('Interval denominator/scale differs')
    endpoints=np.stack((estimate[:,None]-scale[:,None]*(pairs[:,:,1]+corrections),
                       estimate[:,None]-scale[:,None]*(pairs[:,:,0]-corrections)),axis=-1)
    check_pairs(endpoints)
    return endpoints
