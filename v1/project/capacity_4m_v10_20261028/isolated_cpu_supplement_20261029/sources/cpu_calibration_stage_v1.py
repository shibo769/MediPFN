"""CPU calibration science for the registered four-group plan.

Only the frozen 4M backbone and mixture are read. Synthetic truth supervises
calibration; it is never supplied to inference. The stage runner must first
verify the frozen manifest, case RNG schedules and actual predecessor qacct.
"""
import hashlib,json,os,sys,types
from pathlib import Path
import numpy as np
import torch

def _imports(protocol):
 path=Path(protocol['science_imports_source']).resolve(strict=True);raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=protocol['source_sha256'].get(str(path)):raise RuntimeError('science import helper changed')
 module=types.ModuleType('cpu_frozen_imports');module.__file__=str(path);exec(compile(raw,str(path),'exec'),module.__dict__);return module

def run(context):
 p=context['manifest'];shard=context['shard'];output=Path(context['output_dir']).resolve(strict=True)
 if context['stage']!='calibration' or p.get('scope')!='4m_only':raise RuntimeError('wrong science stage')
 helper=_imports(p);cases=helper.selected_cases(p,'calibration',shard)
 if len(cases)!=10:raise RuntimeError('calibration shard must have ten preregistered cases')
 if os.environ.get('NSLOTS'):torch.set_num_threads(int(os.environ['NSLOTS']))
 modules=helper.load_registered(p,'calibration');model=modules['model'];mixture=modules['mixture'];data=modules['data'];inference=modules['inference'];cal=modules['calibration']
 back=model.load_checkpoint(helper.asset(p,'backbone'),'cpu').eval();heads,routers,mean,std,_=mixture.load(helper.asset(p,'mixture'),'cpu')
 rows=[]
 for case in cases:
  scm=data.sample_mechanism(case['mechanism_seed'],case['p'],case['family'])
  drop={'tree_rff':'mlp_output','tree_mlp':'rff_weights','rff_mlp':'tree_values'}.get(case['group'])
  if drop:scm.parameters[drop][...]=0.
  truth=float(scm.theta_NIE);table=scm.draw(case['n'],case['observation_seed']);fold=data.make_folds(case['n'],case['fold_seed'])
  observed=np.column_stack([table['x'],table['a'],table['m'],table['y']])
  with torch.no_grad():result=inference.crossfit_views(back,observed,fold,seed=case['view_seed'],context_size=256,query_batch=128,device='cpu')
  arrays={'x_multi':np.asarray(result['x_multi'])[None,:],'estimates':np.asarray(result['estimates'])[None,:], 'ses':np.asarray(result['ses'])[None,:],'cache_failed':np.array([False])}
  features,q95,est,se,valid=cal.make_features(arrays,heads,routers['sft'],mean,std)
  if valid.tolist()!=[True] or not np.isfinite(truth):raise RuntimeError('invalid calibration result')
  rows.append((features[0],q95[0],float(est[0]),float(se[0]),truth,case['group'],[case[k] for k in ('mechanism_seed','observation_seed','fold_seed','view_seed')],case['n'],case['p']))
 target=output/'calibration.npz'
 with target.open('xb') as stream:
  np.savez_compressed(stream,features=np.stack([x[0] for x in rows]),q=np.stack([x[1] for x in rows]),est=np.asarray([x[2] for x in rows]),se=np.asarray([x[3] for x in rows]),truth=np.asarray([x[4] for x in rows]),group=np.asarray([x[5] for x in rows]),seeds=np.asarray([x[6] for x in rows],dtype=np.uint64),n=np.asarray([x[7] for x in rows]),p=np.asarray([x[8] for x in rows]))
 return {'count':len(rows),'case_ids':[c['case_id'] for c in cases],'npz_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'device':'cpu','truth_scope':'synthetic supervision only'}

if __name__=='__main__':raise RuntimeError('requires frozen CPU stage and reviewed inputs')
