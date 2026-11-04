"""Verify and restore the retained v1 pipeline from this private archive.

No model fitting, benchmark access, calibration, or model selection occurs here.
The archived project files are preserved byte-for-byte; their import names and
absolute artifact references are resolved using the verified archive manifest.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import sys

V1=Path(__file__).resolve().parent
REPO=V1.parent

def verify():
    manifest=json.loads((V1/'manifest.json').read_text())
    for rec in manifest['files']:
        p=(REPO/rec['path']).resolve(strict=True)
        if not p.is_relative_to(REPO):raise ValueError('manifest path escapes archive')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
        if p.stat().st_size!=rec['bytes'] or h.hexdigest()!=rec['sha256']:
            raise ValueError('archive integrity failure: '+rec['path'])
    return manifest

class BoundModules(importlib.abc.MetaPathFinder):
    def __init__(self,mapping):self.mapping=mapping
    def find_spec(self,fullname,path=None,target=None):
        if fullname in self.mapping:
            return importlib.util.spec_from_file_location(fullname,self.mapping[fullname])

def _imports(manifest):
    inventory=json.loads((V1/'production_bindings.json').read_text())
    old=manifest['original_project_root'].rstrip('/')+'/'
    def relocated(path):
        if not path.startswith(old):raise ValueError('unexpected original project path')
        return V1/'project'/path[len(old):]
    mapping={Path(r['path']).stem:REPO/r['path'] for r in manifest['files'] if r['path'].endswith('.py')}
    mapping.update({name:relocated(path) for name,path in inventory['project_import_paths'].items()})
    # All project imports must come from this archive, even on the original SCC.
    finder=BoundModules(mapping);sys.meta_path.insert(0,finder)
    import model,inference,mixture,calibration,robust_retrain_train_seedfix_v2,ablation95,alpha_inference,alpha_heads
    for name in ('model','inference','mixture','calibration','robust_retrain_train_seedfix_v2','ablation95','alpha_inference','alpha_heads'):
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=mapping[name].resolve():raise RuntimeError('non-archive project import '+name)
    return locals()

def restore():
    manifest=verify();modules=_imports(manifest)
    import torch
    asset=lambda name:REPO/manifest['model_assets'][name]
    back=modules['model'].load_checkpoint(asset('backbone'),'cpu').eval()
    heads,routers,mean,std,_=modules['mixture'].load(asset('mixture'),'cpu')
    cm=torch.load(asset('confidence'),map_location='cpu',weights_only=False)
    confidence=modules['calibration'].Confidence(cm['dim'])
    confidence.load_state_dict(cm['state']);confidence.eval()
    chosen=torch.load(asset('chosen'),map_location='cpu',weights_only=False)
    selection=torch.load(asset('selection'),map_location='cpu',weights_only=False)
    if chosen['selected']!='confidence':
        head=modules['robust_retrain_train_seedfix_v2'].Head(len(selection['normalization_mean']))
        head.load_state_dict(selection['states'][chosen['selected']]);head.eval();chosen['head']=head
    freeze=json.loads(asset('levels_freeze').read_text())
    if freeze.get('status')!='PASS':raise ValueError('invalid archived calibration freeze')
    levels={'calibration':modules['calibration'],'confidence':confidence,'temperature':cm['temperature'],'heads':{}}
    for key,rec in freeze['models'].items():
        path=modules['relocated'](rec['head_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['head_sha256']:raise ValueError('level head mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        level=float(key)
        head=modules['alpha_heads'].QuantileHead(saved['dimension'])
        head.load_state_dict(saved['state']);head.eval()
        levels['heads'][level]={'head':head,'mean':saved['normalization_mean'],'sd':saved['normalization_sd'],'expansion':rec['relative_expansion']}
    return locals()

def predict(rows,fold_ids,*,level=.95,seed=2026102601,threads=4):
    """Return an NIE point estimate and the frozen interval for one v1 table.

    rows use the original v1 format [X..., A, M, Y]; fold_ids contain 0 and 1.
    The archived format and model support are validated by the original code.
    This is the v1 archive, not the broader v2 model under development.
    """
    import numpy as np
    import torch
    torch.set_num_threads(int(threads))
    ctx=restore();m=ctx['modules']
    with torch.no_grad():
        out=m['inference'].crossfit_views(ctx['back'],rows,fold_ids,seed=int(seed),context_size=256,query_batch=128,device='cpu')
    a={'x_multi':np.asarray(out['x_multi'])[None,:],'estimates':np.asarray(out['estimates'])[None,:],'ses':np.asarray(out['ses'])[None,:],'cache_failed':np.array([False])}
    x,q,est,se,valid=m['calibration'].make_features(a,ctx['heads'],ctx['routers']['sft'],ctx['mean'],ctx['std'])
    if level==.95:
        result=m['ablation95'].evaluate(x,q,est,se,valid,cal=m['calibration'],training=m['robust_retrain_train_seedfix_v2'],confidence=ctx['confidence'],confidence_state=ctx['cm'],selection=ctx['selection'],chosen=ctx['chosen'],only='selected_calibrated')['selected_calibrated']
        if not result['ok']:raise RuntimeError(result)
        lo,hi=result['lo'],result['hi']
    elif level in (.90,.99):
        lower,upper=m['alpha_inference'].predict_one_level(x,q,est,se,valid,None,ctx['levels'],level,'posttrain')
        lo,hi=float(lower[0]),float(upper[0])
    else:raise ValueError('retained levels are .90, .95, .99')
    return {'estimate':float(est[0]),'standard_error':float(se[0]),'lower':lo,'upper':hi,'level':level,'method':'MediPFN v1 frozen selected calibrated'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='run a synthetic restore test; not a benchmark result')
    args=parser.parse_args()
    manifest=verify()
    if args.verify_only:
        print(json.dumps({'status':'ARCHIVE_SHA256_PASS','files':len(manifest['files']),'checkpoint_files':sum(r['path'].endswith('.pt') for r in manifest['files'])}));return
    if args.smoke:
        import numpy as np
        rng=np.random.default_rng(202611030771)
        x=rng.normal(size=(256,3));a=rng.binomial(1,.5,size=256)
        m=rng.binomial(1,1/(1+np.exp(-(.2+.4*a+.1*x[:,0]))))
        y=.2*a+.5*m+.1*x[:,0]+rng.normal(size=256)
        rows=np.column_stack((x,a,m,y));fold=np.arange(256)%2
        results=[predict(rows,fold,level=level) for level in (.9,.95,.99)]
        if not all(np.isfinite([r['estimate'],r['standard_error'],r['lower'],r['upper']]).all() and r['lower']<=r['upper'] for r in results):raise RuntimeError('smoke inference failed')
        print(json.dumps({'status':'SYNTHETIC_RESTORE_SMOKE_PASS','results':results}));return
    parser.error('choose --verify-only or --smoke; Python callers may import predict')

if __name__=='__main__':main()
