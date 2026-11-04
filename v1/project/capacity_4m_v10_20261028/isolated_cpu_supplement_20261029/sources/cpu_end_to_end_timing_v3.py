"""Cold CPU inference timing from an existing, fixed observed-table file.

The caller must independently validate a frozen protocol, sources and table
identity. No simulator is invoked. This module does not submit or release jobs.
Python/module startup and process creation are excluded; each measurement
includes reading/hash checking the table, loading and checking method-required
model assets, fold construction, feature processing, interval inference, and
encoding the interval as JSON. It is not an amortized warm-model latency.
"""
import hashlib,io,json,time,os,platform,sys,types,inspect
from pathlib import Path
import numpy as np

OUTPUT_ROOT=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9/capacity_4m_v10_20261028/isolated_cpu_supplement_20261029')
METHODS=('sft','confidence_temperature','quantile_uncalibrated','selected_calibrated','sft_90','posttrain_90','sft_99','posttrain_99')
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return h.hexdigest()

def measure_one(case,method,table_path,table_sha256,assets,deps):
 """Measure one table/method. Dependencies are already imported and source-pinned.

 assets maps backbone, mixture, confidence, chosen, selection, levels_freeze
 to {'path': absolute path, 'sha256': digest}; deps provides the exact frozen
 model, mixture, confidence, ablation, levels, training and inference modules.
 Throws on a failure; a scheduler wrapper must record that failure explicitly.
 """
 if method not in METHODS:raise ValueError('unregistered method')
 started=time.perf_counter()
 raw=Path(table_path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=table_sha256:raise ValueError('observed table SHA mismatch')
 with np.load(io.BytesIO(raw),allow_pickle=False) as data:
  if set(data.files)!={'observed'}:raise ValueError('table schema')
  observed=np.array(data['observed'],copy=True)
 if observed.shape!=(int(case['n']),int(case['p'])+3) or not np.isfinite(observed).all():raise ValueError('observed table shape or finiteness')
 required=['backbone','mixture']
 if not method.startswith('sft'):
  required.append('confidence')
  if method in ('quantile_uncalibrated','selected_calibrated'):required.extend(('chosen','selection'))
  elif method in ('posttrain_90','posttrain_99'):required.append('levels_freeze')
 for key in required:
  item=assets[key]
  if sha_file(item['path'])!=item['sha256']:raise ValueError('model asset SHA mismatch: '+key)
 torch=deps['torch'];cal=deps['cal'];mixture=deps['mixture'];training=deps['training'];levels=deps['levels'];ablation=deps['ablation']
 back=deps['load_checkpoint'](assets['backbone']['path'],'cpu').eval()
 heads,routers,mean,std,_=mixture.load(assets['mixture']['path'],'cpu')
 if method in METHODS[1:4]:
  cm=torch.load(assets['confidence']['path'],map_location='cpu',weights_only=False)
  conf=cal.Confidence(cm['dim']);conf.load_state_dict(cm['state']);conf.eval()
  chosen=sel=None
  if method in ('quantile_uncalibrated','selected_calibrated'):
   chosen=torch.load(assets['chosen']['path'],map_location='cpu',weights_only=False)
   sel=torch.load(assets['selection']['path'],map_location='cpu',weights_only=False)
   if chosen['selected']!='confidence':
    head=training.Head(len(sel['normalization_mean']));head.load_state_dict(sel['states'][chosen['selected']]);head.eval();chosen['head']=head
 elif method in ('posttrain_90','posttrain_99'):
  level=.90 if method=='posttrain_90' else .99
  lc=levels.load_context_level(assets['levels_freeze']['path'],assets['levels_freeze']['sha256'],assets['confidence']['path'],assets['confidence']['sha256'],cal,level)
 fold=deps['make_folds'](int(case['n']),int(case['fold_seed']))
 with torch.no_grad():v=deps['crossfit_views'](back,observed,fold,seed=int(case['view_seed']),context_size=256,query_batch=128,device='cpu')
 a={'x_multi':np.asarray(v['x_multi'])[None,:],'estimates':np.asarray(v['estimates'])[None,:],'ses':np.asarray(v['ses'])[None,:],'cache_failed':np.array([False])}
 if method.startswith('sft'):
  est=np.asarray(a['estimates'])[:,1];se=np.asarray(a['ses'])[:,1]
  z=torch.as_tensor(np.nan_to_num((a['x_multi']-mean)/std),dtype=torch.float32)
  with torch.no_grad():
   dist=mixture.mixture(mixture.parts_at(heads,z),routers['sft'](z),True)
   al=.025 if method=='sft' else (.05 if method=='sft_90' else .005)
   qlo=deps['quantile'](dist,al).numpy();qhi=deps['quantile'](dist,1-al).numpy()
  low=est-se*qhi;high=est-se*qlo
 else:
  x,q,est,se,valid=cal.make_features(a,heads,routers['sft'],mean,std)
  if not bool(valid[0]):raise RuntimeError('invalid CPU feature')
  if method in METHODS[:4]:
   item=ablation.evaluate(x,q,est,se,valid,cal=cal,training=training,confidence=conf,confidence_state=cm,selection=sel,chosen=chosen,only=method)[method]
   if not item['ok']:raise RuntimeError('interval method failed: '+repr(item))
   low=np.array([item['lo']]);high=np.array([item['hi']])
  else:
   level=.90 if method=='posttrain_90' else .99
   low,high=levels.predict_one_level(x,q,est,se,valid,None,lc,level,'posttrain')
 if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>high):raise ValueError('invalid interval')
 encoded=json.dumps({'lo':float(low[0]),'hi':float(high[0])},allow_nan=False,separators=(',',':')).encode()
 elapsed=time.perf_counter()-started
 return {'ok':True,'method':method,'seconds':elapsed,'interval_json':encoded.decode(),'table_sha256':table_sha256,'observed':{'sha256':hashlib.sha256(np.ascontiguousarray(observed).tobytes()).hexdigest(),'dtype':str(observed.dtype),'shape':list(observed.shape)},'fold':{'sha256':hashlib.sha256(np.ascontiguousarray(fold).tobytes()).hexdigest(),'dtype':str(fold.dtype),'shape':list(fold.shape)},'device':'cpu','scope':'table read and SHA, frozen model asset SHA/load, folds/features, CPU inference and interval JSON; excludes process/module startup, simulation and output file writes; OS cache not controlled'}

def cpu_hardware(torch):
 model=''
 try:
  for line in Path('/proc/cpuinfo').read_text(encoding='utf8').splitlines():
   if line.partition(':')[0].strip().lower()=='model name':model=line.partition(':')[2].strip();break
 except OSError:pass
 model=model or platform.processor().strip() or 'unavailable'
 slots=int(os.environ['NSLOTS']) if os.environ.get('NSLOTS') else None
 try:affinity=len(os.sched_getaffinity(0))
 except (AttributeError,OSError):affinity=None
 return {'device':'cpu','cpu_model':model,'cpu_threads':int(torch.get_num_threads()),'allocated_slots':slots,'affinity_cpu_count':affinity}

def measure_record(case,method,table_path,table_sha256,assets,deps):
 started=time.perf_counter()
 try:
  record=measure_one(case,method,table_path,table_sha256,assets,deps)
 except Exception as exc:
  record={'ok':False,'method':method,'seconds':time.perf_counter()-started,'error_type':type(exc).__name__,'error':str(exc),'device':'cpu'}
 try:record['hardware']=cpu_hardware(deps['torch'])
 except Exception as exc:record['hardware']={'device':'cpu','cpu_model':'unavailable','cpu_threads':None,'allocated_slots':None,'affinity_cpu_count':None,'error_type':type(exc).__name__,'error':str(exc)}
 record['case_index']=case.get('index');record['scenario']=case.get('scenario');record['kind']=case.get('kind');record['input_table_sha256']=table_sha256
 return record

def measure_registered(manifest_path,manifest_sha,case_id,method):
 """Check immutable declared source/inputs before using the timing helper.

 Scheduler accounting and independent review are external and not implied by
 this function. Direct execution remains disabled until an approved adapter is
 wired to a real protocol.
 """
 manifest_path=Path(manifest_path).resolve(strict=True)
 if sha_file(manifest_path)!=manifest_sha:raise ValueError('CPU protocol SHA mismatch')
 p=json.loads(manifest_path.read_text(encoding='utf8'))
 if p.get('schema')!='cpu_mediation_stage_v1' or p.get('status')!='FROZEN_CPU_STAGE' or p.get('scope')!='4m_only':raise ValueError('CPU protocol not frozen')
 root=OUTPUT_ROOT.resolve(strict=True)
 if Path(p['output_root']).resolve(strict=True)!=root or not manifest_path.is_relative_to(root):raise ValueError('protocol outside isolated root')
 sources=p['source_sha256'];own=str(Path(__file__).resolve())
 if sources.get(own)!=sha_file(own):raise ValueError('timing source is not bound')
 for source,digest in sources.items():
  if sha_file(source)!=digest:raise ValueError('source changed')
 workflow=Path(p['cpu_workflow_source']).resolve(strict=True);raw=workflow.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=sources.get(str(workflow)):raise ValueError('workflow source not bound')
 wf=types.ModuleType('registered_cpu_workflow');wf.__file__=str(workflow);exec(compile(raw,str(workflow),'exec'),wf.__dict__)
 wf.OUTPUT_ROOT=root;wf.validate_case_plans(p,root,'timing');wf.require_stage_predecessors(p,'timing');wf.check_predecessors(p['stages']['timing'])
 helper_path=Path(p['science_imports_source']).resolve(strict=True);raw=helper_path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=sources.get(str(helper_path)):raise ValueError('science import helper not bound')
 helper=types.ModuleType('cpu_timing_import_helper');helper.__file__=str(helper_path);exec(compile(raw,str(helper_path),'exec'),helper.__dict__)
 helper.verify_calibration_audit(p);helper.verify_testbank_audit(p)
 modules=helper.load_registered(p,'timing')
 required=('cal','mixture','training','levels','ablation','load_checkpoint','make_folds','crossfit_views','quantile')
 assignments=p['timing_dependency_roles']
 if set(assignments)!=set(required):raise ValueError('timing dependency roles incomplete')
 deps={}
 for name in required:
  rule=assignments[name]
  if set(rule)!={'module','attribute'} or rule['module'] not in modules:raise ValueError('timing dependency mapping')
  module=modules[rule['module']];attribute=rule['attribute']
  deps[name]=module if attribute is None else getattr(module,attribute)
 import torch
 if getattr(torch,'__version__',None)!=p['torch_version']:raise ValueError('torch runtime changed')
 deps['torch']=torch
 cases=[x for x in p['timing_cases'] if x.get('index')==case_id]
 if len(cases)!=1 or method not in p['timing_methods']:raise ValueError('unregistered timing case/method')
 case=cases[0]
 testplan=json.loads(Path(p['case_plans']['test']['path']).read_text(encoding='utf8'));matches=[x for x in testplan['cases'] if x.get('case_id')==case.get('case_id')]
 fields=('mechanism_seed','observation_seed','fold_seed','view_seed','n','p','observed_path','observed_file_sha256','testbank_status','testbank_report_path')
 if len(matches)!=1 or any(matches[0].get(k)!=case.get(k) for k in fields):raise ValueError('timing case differs from frozen test plan')
 if case['testbank_status']=='failed':
  try:hardware=cpu_hardware(deps['torch'])
  except Exception as exc:hardware={'device':'cpu','cpu_model':'unavailable','cpu_threads':None,'allocated_slots':None,'affinity_cpu_count':None,'error_type':type(exc).__name__,'error':str(exc)}
  return {'ok':False,'method':method,'seconds':0.0,'error_type':'TestbankFailure','error':'registered shared test table generation failed; timing not started','device':'cpu','hardware':hardware,'case_index':case.get('index'),'scenario':case.get('scenario'),'kind':case.get('kind'),'input_table_sha256':None}
 if case['testbank_status']!='ok':raise ValueError('missing testbank status')
 table=Path(case['observed_path']).resolve(strict=True);table_hash=case['observed_file_sha256']
 if not table.is_relative_to(root/'tables') or p['input_sha256'].get(str(table))!=table_hash:raise ValueError('table identity not registered')
 assets=p['timing_assets']
 for item in assets.values():
  path=str(Path(item['path']).resolve(strict=True))
  if p['artifact_sha256'].get(path)!=item['sha256']:raise ValueError('asset identity not registered')
 return measure_record(case,method,table,table_hash,assets,deps)

def production_entry(*args,**kwargs):raise RuntimeError('requires separately reviewed frozen CPU protocol and manually checked predecessors')
if __name__=='__main__':production_entry()
