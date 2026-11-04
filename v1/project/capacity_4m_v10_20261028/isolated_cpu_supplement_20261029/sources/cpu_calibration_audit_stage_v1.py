"""Independent CPU calibration recomputation with bounded guarantee scope.

This re-executes truth and rank checks using the same registered truth helper;
it is independent of the SCM truth implementation, not a second truth helper.
The caller must first verify actual array/finalizer qacct and artifact hashes.
"""
import hashlib,json,os,types
from pathlib import Path
import numpy as np
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _imports(p):
 path=Path(p['science_imports_source']).resolve(strict=True);raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=p['source_sha256'].get(str(path)):raise RuntimeError('science import helper changed')
 m=types.ModuleType('cpu_frozen_imports');m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def run(context):
 p=context['manifest'];root=Path(p['output_root']).resolve(strict=True);output=Path(context['output_dir']).resolve(strict=True)
 if context['stage']!='calibration_audit' or p.get('scope')!='4m_only':raise RuntimeError('wrong calibration audit scope')
 helper=_imports(p);mods=helper.load_registered(p,'calibration_audit');ah=mods['alpha_heads'];truthmod=mods['alpha_truth_v2'];cal=mods['calibration'];data=mods['data']
 if os.environ.get('NSLOTS'):torch.set_num_threads(int(os.environ['NSLOTS']))
 target=Path(p['calibration_freeze_path']).resolve(strict=True)
 expected=root/'outputs'/'calibration_finalize'/'shard_00001'/'alpha_calibration_freeze.json'
 if target!=expected.resolve(strict=True) or p['input_sha256'].get(str(target))!=sha(target):raise RuntimeError('calibration freeze identity')
 freeze=json.loads(target.read_text());finalrec=target.parent/'stage_record.json'
 if p['input_sha256'].get(str(finalrec))!=sha(finalrec):raise RuntimeError('finalizer record drift')
 info=json.loads(finalrec.read_text());fspec=p['stages']['calibration_finalize'];cspec=p['stages']['calibration']
 if (info.get('status')!='STAGE_COMPLETED_NOT_INDEPENDENTLY_AUDITED' or info.get('job_id')!=fspec['job_id'] or info.get('shard')!=1 or
     info.get('result',{}).get('freeze_sha256')!=sha(target) or freeze.get('protocol_sha256')!=info.get('manifest_sha256') or
     freeze.get('calibration_stage_manifest_sha256')!=p['calibration_stage_manifest_sha256']):raise RuntimeError('finalizer identity mismatch')
 plan=json.loads(Path(p['case_plans']['calibration']['path']).read_text());cases=plan['cases']
 if len(cases)!=400 or set(cspec['shards'])!=set(range(1,41)):raise RuntimeError('calibration design mismatch')
 chunks=[];mismatches=0
 for task in sorted(cspec['shards']):
  selected=helper.selected_cases(p,'calibration',task);path=root/'outputs'/'calibration'/('shard_%05d'%task)/'calibration.npz';record=path.parent/'stage_record.json'
  if p['input_sha256'].get(str(path))!=sha(path) or p['input_sha256'].get(str(record))!=sha(record):raise RuntimeError('calibration shard drift')
  rec=json.loads(record.read_text())
  if (rec.get('status')!='STAGE_COMPLETED_NOT_INDEPENDENTLY_AUDITED' or rec.get('job_id')!=cspec['job_id'] or rec.get('scheduler_task_id')!=task or rec.get('manifest_sha256')!=p['calibration_stage_manifest_sha256'] or rec.get('result',{}).get('npz_sha256')!=sha(path) or rec.get('result',{}).get('case_ids')!=[c['case_id'] for c in selected]):raise RuntimeError('calibration shard identity')
  with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in ('features','q','est','se','truth','group','seeds','n','p')}
  if len(selected)!=10 or any(len(v)!=10 for v in a.values()) or a['q'].shape!=(10,3) or a['seeds'].shape!=(10,4):raise RuntimeError('shard shape')
  for j,case in enumerate(selected):
   seeds=[case[k] for k in ('mechanism_seed','observation_seed','fold_seed','view_seed')]
   if list(map(int,a['seeds'][j]))!=seeds or a['group'][j]!=case['group'] or int(a['n'][j])!=case['n'] or int(a['p'][j])!=case['p']:raise RuntimeError('case metadata')
   scm=data.sample_mechanism(case['mechanism_seed'],case['p'],case['family']);drop={'tree_rff':'mlp_output','tree_mlp':'rff_weights','rff_mlp':'tree_values'}.get(case['group'])
   if drop:scm.parameters[drop][...]=0.
   truth=float(truthmod.independent_truth(scm));source=float(scm.theta_NIE)
   if not np.isclose(truth,source,rtol=1e-12,atol=1e-12) or not np.isclose(truth,a['truth'][j],rtol=1e-12,atol=1e-12):mismatches+=1
  chunks.append(a)
 if mismatches:raise RuntimeError('independent truth reconstruction mismatch')
 allrows={k:np.concatenate([a[k] for a in chunks]) for k in chunks[0]}
 if len(set(map(int,allrows['seeds'][:,0])))!=400 or not all(np.sum(allrows['group']==g)==100 for g in ('id','tree_rff','tree_mlp','rff_mlp')):raise RuntimeError('group/mechanism count')
 if any(not np.isfinite(allrows[k]).all() for k in ('features','q','est','se','truth')) or (allrows['se']<=0).any():raise RuntimeError('invalid calibration values')
 confidence_path=helper.asset(p,'confidence');heads_freeze=helper.asset(p,'trained_levels_freeze');train=json.loads(Path(heads_freeze).read_text());cm=torch.load(confidence_path,map_location='cpu',weights_only=False)
 confidence=cal.Confidence(cm['dim']);confidence.load_state_dict(cm['state']);confidence.eval();scale=cal.scale_choice(confidence,allrows['features'],np.ones(400,dtype=bool),cm['temperature']);z=(allrows['est']-allrows['truth'])/allrows['se'];recomputed={}
 for level in ah.LEVELS:
  desc=train['models'][str(level)];hp=Path(desc['model']).resolve(strict=True)
  if p['artifact_sha256'].get(str(hp))!=desc['sha256'] or sha(hp)!=desc['sha256']:raise RuntimeError('head drift')
  saved=torch.load(hp,map_location='cpu',weights_only=False)
  if saved['level']!=level or saved['anchor']!='fixed pre-existing temperature-scaled 95% confidence':raise RuntimeError('head identity')
  head=ah.QuantileHead(saved['dimension']);head.load_state_dict(saved['state']);head.eval();x=np.clip((allrows['features']-saved['normalization_mean'])/saved['normalization_sd'],-8,8).astype(np.float32)
  lo,hi=ah.residual_endpoints(head,x,allrows['q'],scale);correction,k=ah.relative_order_statistic(z,lo,hi,level);reported=freeze['models'][str(level)]
  if k!={.90:361,.99:397}[level] or reported['k']!=k or reported['n']!=400 or reported['head_path']!=str(hp) or reported['head_sha256']!=sha(hp) or not np.isclose(correction,reported['relative_expansion'],rtol=1e-12,atol=1e-12):raise RuntimeError('calibration correction mismatch')
  recomputed[str(level)]={'k':k,'relative_expansion':float(correction)}
 report={'status':'SCOPED_CPU_CALIBRATION_AUDIT_PASS','protocol_sha256':context['manifest_sha256'],'freeze_sha256':sha(target),'calibration_job':cspec['job_id'],'finalizer_job':fspec['job_id'],'mechanisms':400,'truth_mismatches':0,'models':recomputed,'limitations':['Truth recomputation uses the same independently-of-SCM helper as the finalizer, not a second truth implementation.','No OOD coverage guarantee or independent test result is established.']}
 raw=json.dumps(report,indent=2,allow_nan=False).encode();dest=output/'calibration_audit.json'
 with dest.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'audit_path':str(dest),'audit_sha256':sha(dest),'freeze_sha256':sha(target),'mechanisms':400,'status':'CALIBRATION_AUDIT_WRITTEN'}

if __name__=='__main__':raise RuntimeError('requires frozen CPU stage, predecessor qacct and reviewed inputs')
