"""Finalize 400 CPU calibration cases, independently checking synthetic truth.

Actual calibration array qacct and all predecessor output hashes must be
checked by the stage runner before entry. No test table is opened.
"""
import hashlib,json,os,types
from pathlib import Path
import numpy as np
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _imports(protocol):
 path=Path(protocol['science_imports_source']).resolve(strict=True);raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=protocol['source_sha256'].get(str(path)):raise RuntimeError('science import helper changed')
 module=types.ModuleType('cpu_frozen_imports');module.__file__=str(path);exec(compile(raw,str(path),'exec'),module.__dict__);return module

def run(context):
 p=context['manifest'];root=Path(p['output_root']).resolve(strict=True);output=Path(context['output_dir']).resolve(strict=True)
 if context['stage']!='calibration_finalize' or p.get('scope')!='4m_only':raise RuntimeError('wrong science stage')
 helper=_imports(p);mods=helper.load_registered(p,'calibration_finalize');ah=mods['alpha_heads'];truth_audit=mods['alpha_truth_v2'];cal=mods['calibration'];data=mods['data']
 if os.environ.get('NSLOTS'):torch.set_num_threads(int(os.environ['NSLOTS']))
 case_plan=json.loads(Path(p['case_plans']['calibration']['path']).read_text());cases=case_plan['cases']
 spec=p['stages']['calibration'];shards=spec['shards']
 if len(cases)!=400 or set(shards)!=set(range(1,41)):raise RuntimeError('calibration plan size')
 prior=p['stages']['calibration_finalize']['predecessors'];cal_job=spec['job_id'];cal_name=spec['jobname']
 if len([x for x in prior if x.get('job')==cal_job and x.get('jobname')==cal_name and set(x.get('tasks',[]))==set(range(1,41))])!=1:raise RuntimeError('calibration array accounting scope missing')
 arrays=[];shard_evidence=[]
 for task in sorted(shards):
  taskcases=helper.selected_cases(p,'calibration',task)
  path=root/'outputs'/'calibration'/('shard_%05d'%task)/'calibration.npz';record=path.parent/'stage_record.json'
  for item in (path,record):
   if p['input_sha256'].get(str(item))!=sha(item):raise RuntimeError('calibration output input drift')
  info=json.loads(record.read_text())
  if (info.get('status')!='STAGE_COMPLETED_NOT_INDEPENDENTLY_AUDITED' or info.get('stage')!='calibration' or info.get('shard')!=task or info.get('job_id')!=cal_job or info.get('scheduler_task_id')!=task or
      info.get('manifest_sha256')!=p['calibration_stage_manifest_sha256'] or info.get('result',{}).get('npz_sha256')!=sha(path) or
      info.get('result',{}).get('case_ids')!=[case['case_id'] for case in taskcases]):raise RuntimeError('calibration shard provenance')
  with np.load(path,allow_pickle=False) as f:values={key:f[key] for key in ('features','q','est','se','truth','group','seeds','n','p')}
  if len(taskcases)!=10 or any(len(v)!=10 for v in values.values()) or values['q'].shape!=(10,3) or values['seeds'].shape!=(10,4):raise RuntimeError('calibration shard shape')
  for j,case in enumerate(taskcases):
   seeds=[case[k] for k in ('mechanism_seed','observation_seed','fold_seed','view_seed')]
   if list(map(int,values['seeds'][j]))!=seeds or values['group'][j]!=case['group'] or int(values['n'][j])!=case['n'] or int(values['p'][j])!=case['p']:raise RuntimeError('calibration case mismatch')
   scm=data.sample_mechanism(case['mechanism_seed'],case['p'],case['family'])
   drop={'tree_rff':'mlp_output','tree_mlp':'rff_weights','rff_mlp':'tree_values'}.get(case['group'])
   if drop:scm.parameters[drop][...]=0.
   truth=truth_audit.independent_truth(scm);reference=float(scm.theta_NIE)
   if not np.isclose(reference,truth,rtol=1e-12,atol=1e-12) or not np.isclose(values['truth'][j],truth,rtol=1e-12,atol=1e-12):raise RuntimeError('calibration truth mismatch')
  arrays.append(values);shard_evidence.append({'path':str(path),'sha256':sha(path),'record_sha256':sha(record)})
 a={key:np.concatenate([x[key] for x in arrays]) for key in arrays[0]}
 if len(a['truth'])!=400 or len(set(map(int,a['seeds'][:,0])))!=400 or not all(np.sum(a['group']==name)==100 for name in ('id','tree_rff','tree_mlp','rff_mlp')):raise RuntimeError('calibration mechanism count')
 for key in ('features','q','est','se','truth'):
  if not np.isfinite(a[key]).all():raise RuntimeError('nonfinite calibration value '+key)
 if (a['se']<=0).any():raise RuntimeError('invalid calibration se')
 confidence_path=helper.asset(p,'confidence');freeze_path=helper.asset(p,'trained_levels_freeze');freeze=json.loads(Path(freeze_path).read_text())
 cm=torch.load(confidence_path,map_location='cpu',weights_only=False);confidence=cal.Confidence(cm['dim']);confidence.load_state_dict(cm['state']);confidence.eval()
 scale95=cal.scale_choice(confidence,a['features'],np.ones(400,dtype=bool),cm['temperature']);z=(a['est']-a['truth'])/a['se'];models={}
 for level in ah.LEVELS:
  desc=freeze['models'][str(level)];hp=Path(desc['model']).resolve(strict=True)
  if p['artifact_sha256'].get(str(hp))!=desc['sha256'] or sha(hp)!=desc['sha256']:raise RuntimeError('level head source changed')
  saved=torch.load(hp,map_location='cpu',weights_only=False)
  if saved['level']!=level or saved['anchor']!='fixed pre-existing temperature-scaled 95% confidence':raise RuntimeError('head level/anchor mismatch')
  head=ah.QuantileHead(saved['dimension']);head.load_state_dict(saved['state']);head.eval()
  x=np.clip((a['features']-saved['normalization_mean'])/saved['normalization_sd'],-8,8).astype(np.float32)
  lo,hi=ah.residual_endpoints(head,x,a['q'],scale95);correction,k=ah.relative_order_statistic(z,lo,hi,level)
  if k!={0.90:361,0.99:397}[level]:raise RuntimeError('calibration rank mismatch')
  models[str(level)]={'head_path':str(hp),'head_sha256':desc['sha256'],'relative_expansion':correction,'n':400,'k':k}
 result={'status':'PASS','protocol_sha256':context['manifest_sha256'],'calibration_stage_manifest_sha256':p['calibration_stage_manifest_sha256'],'groups':'4 independent groups of 100 mechanisms','fixed_95_confidence_anchor_not_arbitrary_level_confidence':True,'models':models,'shards':shard_evidence,'source_sha256':sha(__file__),'guarantee_scope':'fixed stratified calibration; no arbitrary-OOD or conditional coverage guarantee'}
 raw=json.dumps(result,indent=2,allow_nan=False).encode();target=output/'alpha_calibration_freeze.json'
 with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'freeze_path':str(target),'freeze_sha256':sha(target),'count':400,'status':'CALIBRATED_NOT_INDEPENDENTLY_AUDITED'}

if __name__=='__main__':raise RuntimeError('requires frozen CPU stage, predecessor qacct and reviewed inputs')
