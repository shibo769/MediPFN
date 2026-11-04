"""Verify and restore the trained primary MediPFN v2 research bundle.

Input is a numeric NPZ observed table. Simulator truth is never accessed.
The standard-library --verify-only path does not import NumPy or PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def checked_path(base,relative):
    path=(base/relative).resolve()
    try:path.relative_to(ROOT)
    except ValueError:raise ValueError('Bundle assets must remain inside this v2 distribution') from None
    if not path.is_file():raise FileNotFoundError(path)
    return path


def verify_bundle(bundle_path):
    path=Path(bundle_path).resolve()
    if path.is_dir():path=path/'bundle.json'
    bundle=json.loads(path.read_text(encoding='utf-8'))
    if bundle.get('format')!='medipfn-v2-primary-bundle-1':raise ValueError('Unknown bundle format')
    roles={'factual','nested','error','posttrained_error','calibration'}
    if set(bundle.get('files',{}))!=roles:raise ValueError('Incomplete primary model bundle')
    assets={}
    for role,record in bundle['files'].items():
        target=checked_path(path.parent,record['path'])
        if target.stat().st_size!=record['bytes'] or file_sha(target)!=record['sha256']:
            raise ValueError('Bundle asset SHA/size mismatch: '+role)
        assets[role]=target
    source_path=checked_path(path.parent,bundle['source_manifest'])
    if source_path!=ROOT/'release_manifest.json':raise ValueError('Bundle uses a different source distribution')
    if file_sha(source_path)!=bundle['source_manifest_sha256']:raise ValueError('Source manifest SHA mismatch')
    source=json.loads(source_path.read_text(encoding='utf-8'))
    if source['release_sha256']!=bundle['source_release_sha256']:raise ValueError('Source release identity differs')
    for relative,expected in source['files'].items():
        if file_sha(checked_path(ROOT,relative))!=expected:raise ValueError('Source file SHA mismatch: '+relative)
    return bundle,assets,dict(status='BUNDLE_SHA256_VERIFIED',bundle_sha256=file_sha(path),
                             model_and_calibration_files=len(assets),source_files=len(source['files']),
                             source_release_sha256=source['release_sha256'])


def load_observed_table(path):
    import numpy as np
    # Deliberately access only observed arrays and feature-type metadata. Extra
    # fields, including synthetic truth, are not materialized or used.
    with np.load(path,allow_pickle=False) as data:
        table={key:np.asarray(data[key],dtype=np.float32) for key in ('x','a','m','y')}
        for key in ('x_types','mediator_types'):
            if key in data.files:
                value=np.asarray(data[key])
                if value.ndim!=1 or value.dtype.kind not in 'US':raise ValueError(key+' must be a one-dimensional string array')
                table[key]=value.astype(str).tolist()
    for key in ('a','y'):
        if table[key].ndim==2 and table[key].shape[1]==1:table[key]=table[key][:,0]
        if table[key].ndim!=1:raise ValueError(key+' must contain one value per observed row')
    n=len(table['a'])
    if n<16 or len(table['y'])!=n:raise ValueError('At least 16 complete observed rows are required')
    for key,limit in (('x',32),('m',128)):
        value=table[key]
        if value.ndim==1:value=value[:,None]
        minimum=0 if key=='x' else 1
        if value.ndim!=2 or value.shape[0]!=n or not minimum<=value.shape[1]<=limit:
            raise ValueError(key+' has an invalid row or feature dimension')
        table[key]=value
        type_key='x_types' if key=='x' else 'mediator_types'
        types=table.setdefault(type_key,['continuous']*value.shape[1])
        if len(types)!=value.shape[1] or any(t not in ('continuous','binary','categorical') for t in types):
            raise ValueError(type_key+' differs from the model feature-type interface')
        for j,t in enumerate(types):
            if t=='binary' and not np.isin(value[:,j],[0,1]).all():raise ValueError('A binary-tagged column must contain 0/1')
    if not all(np.isfinite(table[key]).all() for key in ('x','a','m','y')):raise ValueError('Observed inputs must be finite and complete')
    if not np.isin(table['a'],[0,1]).all():raise ValueError('The trained treatment contrast requires treatment values 0 and 1')
    if min(int((table['a']==a).sum()) for a in (0,1))<8:raise ValueError('Both treatment groups need at least eight rows for nested cross-fitting')
    return table


def predict(bundle_path,data_path,device='cpu',seed=123,cpu_threads=1):
    bundle,assets,verification=verify_bundle(bundle_path)
    import torch
    if cpu_threads<1:raise ValueError('cpu_threads must be positive')
    torch.set_num_threads(cpu_threads)
    from calibrate import load_calibrated_error,predict_intervals
    from error_binding import BoundNIEEstimator
    table=load_observed_table(data_path)
    error_model,_,calibration=load_calibrated_error(assets['posttrained_error'],assets['calibration'],device)
    estimator=BoundNIEEstimator(assets['factual'],assets['nested'],device,
                                probability_floor=calibration['probability_floor'],outer_folds=calibration['outer_folds'])
    result=predict_intervals(estimator.estimate(table,seed=seed),table,error_model,calibration)
    result.update(bundle_verification=verification,rows=len(table['a']),seed=seed,
                  target=bundle['target'],model_status=bundle['status'],
                  scientific_id_ood_test_evaluation=bundle['scientific_id_ood_test_evaluation'],
                  calibration_scope=calibration['scope'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',default=str(ROOT/'checkpoints/primary/bundle.json'))
    parser.add_argument('--data',help='NPZ with observed x, a, m, y and optional string feature types')
    parser.add_argument('--output',help='New JSON output path; existing files are not overwritten')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        _,_,report=verify_bundle(args.bundle);print(json.dumps(report,indent=2));return
    if not args.data or not args.output:parser.error('--data and --output are required unless --verify-only')
    result=predict(args.bundle,args.data,args.device,args.seed,args.cpu_threads)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(output=str(output),estimate=result['estimate'],intervals=result['intervals']),indent=2,allow_nan=False))


if __name__=='__main__':main()
