"""Create an isolated pinned TabPFN-v3 runtime and fetch public academic weights.

No model fitting or scheduler submission occurs here. Existing environments are
read only. The large existing CUDA torch runtime is shared read-only through a
documented .pth file; all new/upgraded Python packages are installed in the new venv.
No credentials or license-form responses are fabricated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

SOURCE_COMMIT='953a4b916a19856fbd8dcda089a206b5bfe83696'
HF_REPO='Prior-Labs/tabpfn_3'
HF_REVISION='24a16a89d245878b846555110985634aa2e656d7'
WEIGHTS=('tabpfn-v3-classifier-v3_default.ckpt','tabpfn-v3-regressor-v3_default.ckpt')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def download(url,dest,expected=None):
    dest=Path(dest)
    if dest.exists() and (expected is None or digest(dest)==expected):return digest(dest)
    tmp=dest.with_suffix(dest.suffix+'.part')
    with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:
        while True:
            b=r.read(8388608)
            if not b:break
            f.write(b)
    actual=digest(tmp)
    if expected and actual!=expected:raise RuntimeError('Downloaded SHA mismatch: '+dest.name)
    tmp.replace(dest);return actual


def main(root,base_python):
    root=Path(root).resolve()
    project=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9').resolve()
    if project not in root.parents or root.name!='medipfn_v2_20261103':raise ValueError('Unexpected isolated project root')
    artifacts=root/'tabular_backbones'/'v3_runtime';artifacts.mkdir(parents=True,exist_ok=True)
    manifest_path=artifacts/'bootstrap_manifest.json'
    m=dict(status='STARTED',source_commit=SOURCE_COMMIT,hf_repository=HF_REPO,hf_revision=HF_REVISION,
           base_python=str(base_python),environment=str(root/'env_tabpfn_v3'),started_unix=time.time())
    def save():manifest_path.write_text(json.dumps(m,indent=2)+'\n')
    def run(args):
        p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,
                         env=dict(os.environ,PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_INPUT='1',PYTHONPATH=''),timeout=1200)
        with (artifacts/'installation.log').open('a') as f:f.write('COMMAND '+repr(args)+'\n'+p.stdout+'\n')
        if p.returncode:raise RuntimeError('Command failed; see installation.log: '+args[0])
        return p.stdout
    save()
    try:
        env=root/'env_tabpfn_v3';py=env/'bin/python'
        if not py.exists():run([base_python,'-m','venv',str(env)])
        base_site=json.loads(run([base_python,'-c','import json,sysconfig;print(json.dumps(sysconfig.get_paths()["purelib"]))']).strip())
        new_site=json.loads(run([str(py),'-c','import json,sysconfig;print(json.dumps(sysconfig.get_paths()["purelib"]))']).strip())
        (Path(new_site)/'read_only_existing_cuda_runtime.pth').write_text(base_site+'\n')
        m['shared_read_only_site']=base_site
        source=artifacts/('TabPFN-'+SOURCE_COMMIT+'.zip')
        m['source_archive_sha256']=download('https://codeload.github.com/PriorLabs/TabPFN/zip/'+SOURCE_COMMIT,source)
        m['status']='INSTALLING';save()
        run([str(py),'-m','pip','install',str(source)])
        (artifacts/'requirements.lock.txt').write_text(run([str(py),'-m','pip','freeze','--all']))
        sanity=run([str(py),'-c','import json,inspect,importlib.metadata,torch;from tabpfn import TabPFNRegressor,TabPFNClassifier;from tabpfn.constants import ModelVersion;print(json.dumps(dict(tabpfn=importlib.metadata.version("tabpfn"),torch=torch.__version__,cuda=torch.version.cuda,has_v3=hasattr(ModelVersion,"V3"),regressor_differentiable=hasattr(TabPFNRegressor,"fit_with_differentiable_input"),classifier_differentiable=hasattr(TabPFNClassifier,"fit_with_differentiable_input"))))'])
        m['runtime']=json.loads(sanity.strip().splitlines()[-1]);m['status']='DOWNLOADING_WEIGHTS';save()
        if not m['runtime']['has_v3']:raise RuntimeError('Installed runtime does not support V3')
        with urllib.request.urlopen('https://huggingface.co/api/models/'+HF_REPO+'/revision/'+HF_REVISION+'?blobs=true',timeout=60) as r:
            meta=json.load(r)
        (artifacts/'hf_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        if meta.get('sha')!=HF_REVISION:raise RuntimeError('HF revision mismatch')
        m['hf_gated']=meta.get('gated')
        siblings={s['rfilename']:s for s in meta['siblings']}
        weights=artifacts/'weights';weights.mkdir(exist_ok=True)
        m['license_sha256']=download('https://huggingface.co/'+HF_REPO+'/resolve/'+HF_REVISION+'/LICENSE',artifacts/'TABPFN-3-LICENSE')
        m['weights']={}
        for name in WEIGHTS:
            lfs=siblings[name].get('lfs',{})
            expected=lfs.get('sha256')
            if not expected:raise RuntimeError('No independent HF LFS SHA for '+name)
            actual=download('https://huggingface.co/'+HF_REPO+'/resolve/'+HF_REVISION+'/'+name+'?download=true',weights/name,expected)
            m['weights'][name]=dict(path=str(weights/name),bytes=(weights/name).stat().st_size,sha256=actual)
            save()
        c=dict(model_version='v3',package_version=m['runtime']['tabpfn'],
               classifier_path=str(weights/WEIGHTS[0]),classifier_sha256=m['weights'][WEIGHTS[0]]['sha256'],
               regressor_path=str(weights/WEIGHTS[1]),regressor_sha256=m['weights'][WEIGHTS[1]]['sha256'],
               device='cuda',n_estimators=1,n_preprocessing_jobs=1,probability_floor=.01)
        (artifacts/'v3_local_weights_config.json').write_text(json.dumps(c,indent=2)+'\n')
        m['status']='READY_FOR_COMPUTE_FIXTURE';m['finished_unix']=time.time();save()
    except urllib.error.HTTPError as e:
        m.update(status='AUTH_REQUIRED' if e.code in (401,403) else 'FAILED',error='HTTP '+str(e.code),url=e.url)
        save();raise
    except Exception as e:
        m.update(status='FAILED',error=repr(e));save();raise
    print(json.dumps(m),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base-python',required=True)
    a=p.parse_args();main(a.root,a.base_python)
