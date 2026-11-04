"""Isolated persistent CPU worker for exact frozen old v1 and native FSP.

Fresh interpreter prevents old model.py/nuisance_model.py from colliding with
new v9 modules. Reads observed rows/folds only, never effect truth or labels.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(args):
    root=Path(args.legacy_dir).resolve();code=root/'code';sys.path.insert(0,str(code))
    import numpy as np
    import torch
    from scipy.special import expit
    from nuisance_model import FrozenNuisancePFN
    if args.threads:torch.set_num_threads(args.threads)
    elif os.environ.get('NSLOTS'):torch.set_num_threads(int(os.environ['NSLOTS']))
    model=FrozenNuisancePFN(root/'base.pt');checkpoint=torch.load(root/'v1_head.pt',map_location='cpu',weights_only=False);model.decoder.load_state_dict(checkpoint['state_dict']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    print(json.dumps(dict(ready=True,device='cpu',base_sha256=sha(root/'base.pt'),head_sha256=sha(root/'v1_head.pt'))),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('command')=='stop':break
        with np.load(request['raw_path'],allow_pickle=False) as z:raw={k:z[k] for k in ('x','a','m','y','fold_ids')}
        x=raw['x'];b,n,p=x.shape
        if p!=3:raise ValueError('Old model accepts exactly three X; never truncate/pad to emulate generality')
        nu=np.full((b,n,5),np.nan);point=np.full(b,np.nan);logV=np.full(b,np.nan);old_errors=[];native_errors=[]
        for i in range(b):
            rows=np.column_stack((x[i],raw['a'][i],raw['m'][i],raw['y'][i])).astype(np.float32);folds=raw['fold_ids'][i];error=''
            try:
                with torch.inference_mode():
                    for fold in (0,1):
                        support=torch.from_numpy(rows[folds!=fold][None]);context=model.encode_support(support);ids=np.flatnonzero(folds==fold)
                        for start in range(0,len(ids),128):
                            take=ids[start:start+128];prediction=model.predict_nuisance(context,torch.from_numpy(rows[take,:3][None])).cpu().numpy()[0].astype(float)
                            nu[i,take]=np.column_stack((expit(prediction[:,:3]),prediction[:,5:7]))
                if not np.isfinite(nu[i]).all():raise FloatingPointError('Nonfinite old-v1 nuisances')
            except Exception as exc:nu[i]=np.nan;error=repr(exc)
            old_errors.append(error);error=''
            try:
                with torch.inference_mode():theta,variance=model.base(torch.from_numpy(rows[None]))
                point[i]=float(theta[0]);logV[i]=float(variance[0])
                with np.errstate(over='ignore',invalid='ignore'):se=float(np.exp(logV[i]/2)/np.sqrt(n))
                if not np.isfinite([point[i],se]).all() or not se>0:raise FloatingPointError('Invalid native point/SE, no variance clipping or inflation')
            except Exception as exc:error=repr(exc)
            native_errors.append(error)
        output=Path(request['output_path'])
        with output.open('xb') as f:np.savez_compressed(f,nuisance=nu,native_estimate=point,native_logV=logV,old_errors=np.asarray(old_errors,str),native_errors=np.asarray(native_errors,str))
        print(json.dumps(dict(complete=True,output_path=str(output),sha256=sha(output),tables=b)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-dir',required=True);parser.add_argument('--threads',type=int);worker(parser.parse_args())
