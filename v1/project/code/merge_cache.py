"""Merge compact feature banks without reading/dropping sealed test rows."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from posttrain import Cache,save
from evaluate import EvaluationGuard

REQUIRED=('x_single','x_multi','estimates','ses','truth','mechanism_id','split','bank')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def resolve(base,value):
    p=Path(str(value).replace('\\','/'));return p if p.is_absolute() else Path(base)/p
def entries(path):
    value=read(path);return value,value['groups'] if isinstance(value,dict) else value
def entry_path(entry,manifest):
    value=next((entry[k] for k in ('cache_path','path','npz_path') if k in entry),None)
    if value is None:raise ValueError('Cache manifest entry requires cache_path/path')
    return resolve(Path(manifest).parent,value)
def entry_digest(entry):return next((entry[k] for k in ('cache_sha256','sha256','hash') if k in entry),None)
def load_entry(entry,manifest,protocol_hash=None):
    path=entry_path(entry,manifest);digest=entry_digest(entry)
    if digest is None or sha(path)!=digest:raise ValueError('Cache missing/changed hash '+str(path))
    if 'metadata_path' in entry:
        metadata=resolve(Path(manifest).parent,entry['metadata_path'])
        if sha(metadata)!=entry.get('metadata_sha256'):raise ValueError('Cache metadata changed '+str(metadata))
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if protocol_hash is not None:
        embedded=a.get('protocol_sha256')
        if embedded is not None and str(np.asarray(embedded).item())!=protocol_hash:raise ValueError('Cache protocol hash mismatch')
    return path,a
def row_field(a,entry,key,n):
    if key in a:
        v=np.asarray(a[key]);return np.repeat(v.item(),n) if v.ndim==0 else v
    if key in entry:return np.repeat(entry[key],n)
    raise ValueError('Missing cache field '+key)

def merge(protocol,manifest,output,calibration_only=False):
    EvaluationGuard(protocol).check('calibration' if calibration_only else 'train');protocol_hash=sha(protocol);header,groups=entries(manifest)
    if isinstance(header,dict) and header.get('protocol_sha256',protocol_hash)!=protocol_hash:raise ValueError('Manifest protocol drift')
    selected=[];arrays={k:[] for k in REQUIRED};optional={};dimensions=None;checkpoint_hashes=set()
    for entry in groups:
        split=str(entry.get('split',''));bank=str(entry.get('bank',''))
        allowed=split in ('calibration','calibrate') if calibration_only else split in ('train','dev') and (split=='dev' or bank in ('head','sft','dpo','rlvr'))
        if not allowed:continue
        path,a=load_entry(entry,manifest,protocol_hash);n=len(a['truth'])
        current=(a['x_single'].shape[1],a['x_multi'].shape[1])
        if dimensions is None:dimensions=current
        if dimensions!=current:raise ValueError('Mixed feature dimensions/backbones')
        if a['estimates'].shape!=(n,2) or a['ses'].shape!=(n,2):raise ValueError('Canonical point/SE shape')
        for key in REQUIRED:
            value=row_field(a,entry,key,n)
            if len(value)!=n:raise ValueError('Cache field row mismatch '+key)
            arrays[key].append(value)
        for key,aliases in {'seeds':('seeds','observation_seed','observation_seeds'),'fold_seeds':('fold_seeds','fold_seed','foldseeds')}.items():
            value=next((a[k] for k in aliases if k in a),None)
            if value is None:raise ValueError('Observation/fold identities are required')
            if np.asarray(value).shape!=(n,):raise ValueError('Seed shape')
            optional.setdefault(key,[]).append(np.asarray(value))
        for key in ('failed','error','cost','p','n','family'):
            if key in a:
                value=np.asarray(a[key]);optional.setdefault(key,[]).append(np.repeat(value.item(),n) if value.ndim==0 else value)
        ck=entry.get('checkpoint_sha256',header.get('checkpoint_sha256') if isinstance(header,dict) else None)
        if ck:checkpoint_hashes.add(ck)
        selected.append(dict(path=str(path),sha256=sha(path),split=split,bank=bank,tables=n))
    if not selected:raise ValueError('No eligible cache groups')
    if len(checkpoint_hashes)>1:raise ValueError('Cannot merge different backbone checkpoints')
    result={k:np.concatenate(v) for k,v in arrays.items()}
    for key,v in optional.items():
        if len(v)==len(selected):result[key]=np.concatenate(v)
        elif key in ('seeds','fold_seeds'):raise ValueError('Incomplete identity fields')
    n=len(result['truth']);obs=result['seeds'];fold=result['fold_seeds']
    if len(np.unique(obs))!=n or len(np.unique(fold))!=n or set(map(int,obs))&set(map(int,fold)):raise ValueError('Repeated/overlapping observation and fold seeds')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.suffix!='.npz':raise ValueError('Merged output must end in .npz')
    if output.exists():raise FileExistsError('Merged bank is immutable '+str(output))
    temporary=output.with_name(output.stem+'.partial.npz')
    if temporary.exists():raise FileExistsError('Interrupted partial merge requires explicit recovery')
    try:
        np.savez_compressed(temporary,**result,protocol_sha256=protocol_hash,checkpoint_sha256=next(iter(checkpoint_hashes),'unknown'))
        if not calibration_only:Cache(temporary)
        temporary.replace(output)
    finally:
        if temporary.exists():temporary.unlink()
    save(output.with_suffix('.manifest.json'),dict(passed=True,protocol_sha256=protocol_hash,cache_manifest_sha256=sha(manifest),output_sha256=sha(output),checkpoint_sha256=next(iter(checkpoint_hashes),'unknown'),tables=n,feature_dims=dimensions,calibration_only=calibration_only,failed_rows_retained=int((~np.isfinite(result['x_single']).all(1)|~np.isfinite(result['x_multi']).all(1)).sum()),groups=selected,source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('posttrain.py'))}))
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--cache-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--calibration-only',action='store_true');a=p.parse_args();out=merge(a.protocol,a.cache_manifest,a.output,a.calibration_only);print('MERGED',len(out['truth']),flush=True)
if __name__=='__main__':main()
