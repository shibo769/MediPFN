"""RF-DML/B100 on the frozen shared CPU test bank; source candidate only.

Uses actual saved observed bytes after exact reconstruction and fold checks.
All methods share the same table/fold and failure case denominator as PFN.
The RF mapping must be formally frozen before any test table is generated.
"""
import hashlib,json,os,types,traceback
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(a):
 a=np.ascontiguousarray(a);return {'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'dtype':str(a.dtype),'shape':list(a.shape)}
def _imports(p):
 path=Path(p['science_imports_source']).resolve(strict=True);raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=p['source_sha256'].get(str(path)):raise RuntimeError('science import helper changed')
 m=types.ModuleType('cpu_rf_imports');m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__);return m

def run(context):
 p=context['manifest'];root=Path(p['output_root']).resolve(strict=True);out=Path(context['output_dir']).resolve(strict=True)
 if context['stage']!='rf' or p.get('scope')!='4m_only' or not out.is_relative_to(root/'outputs'/'rf'):raise RuntimeError('RF stage scope')
 helper=_imports(p);helper.verify_calibration_audit(p);helper.verify_testbank_audit(p);selected=helper.selected_cases(p,'rf',context['shard'],'test')
 seed_cases_path=Path(p['rf_seed_cases_path']).resolve(strict=True);seed_mapping_path=Path(p['rf_seed_mapping_path']).resolve(strict=True)
 pretest=root/'protocols'/'testbank.json'
 if sha(pretest)!=p['testbank_stage_manifest_sha256']:raise RuntimeError('pre-test protocol hash mismatch')
 pre=json.loads(pretest.read_text())
 if (pre.get('status')!='FROZEN_CPU_STAGE' or pre.get('rf_seed_cases_path')!=str(seed_cases_path) or pre.get('rf_seed_mapping_path')!=str(seed_mapping_path) or pre['input_sha256'].get(str(seed_cases_path))!=sha(seed_cases_path) or pre['input_sha256'].get(str(seed_mapping_path))!=sha(seed_mapping_path)):raise RuntimeError('RF seed mapping was not committed before shared test generation')
 if p['input_sha256'].get(str(seed_cases_path))!=sha(seed_cases_path) or p['input_sha256'].get(str(seed_mapping_path))!=sha(seed_mapping_path):raise RuntimeError('RF seed plan not frozen')
 seed_cases=json.loads(seed_cases_path.read_text())['cases'];test_cases=json.loads(Path(p['case_plans']['test']['path']).read_text())['cases']
 if len(seed_cases)!=len(test_cases) or len(seed_cases)!=2952 or [x['index'] for x in seed_cases]!=list(range(2952)):raise RuntimeError('RF/test case design')
 for a,b in zip(seed_cases,test_cases):
  if any(b.get(k)!=v for k,v in a.items()):raise RuntimeError('RF seed schedule differs from test plan')
 mods=helper.load_registered(p,'rf');core=mods['rf_supplement_core'];binding=mods['rf_seed_binding_v1'];execution=mods['rf_method_execution_v2'];baseline=mods['baselines'];evaluate=mods['evaluate'];loader=mods['cpu_case_loader']
 if baseline.DEFAULTS['rf']!=p['rf_settings']:raise RuntimeError('RF settings changed')
 subset=core.select_subset(seed_cases,54,45)
 if subset!=p['rf_bootstrap_indices'] or len(subset)!=261 or p['rf_bootstrap_repeats']!=100:raise RuntimeError('RF bootstrap subset mismatch')
 reserved=binding.validate(json.loads(seed_mapping_path.read_text()),seed_cases,subset)
 covroot=Path(p['covariate_root']).resolve(strict=True)
 if not covroot.is_relative_to(root):raise RuntimeError('real X outside study')
 os.environ['NIE_WAVE_XDIR']=str(covroot)
 slots=int(os.environ.get('NSLOTS','1'))
 if slots<1:raise RuntimeError('invalid CPU allocation')
 rows=[]
 for case in selected:
  idx=int(case['index']);planned=idx in subset;row={'index':idx,'case':case,'bootstrap_planned':planned,'common_ok':False,'methods':{}}
  try:
   if case.get('testbank_status')=='failed':raise RuntimeError('registered shared test table generation failed')
   if case.get('testbank_status')!='ok':raise RuntimeError('missing testbank status')
   observed,fold,truth,pop=loader.make_case(case);truth=float(truth)
   if not np.isfinite(truth):raise RuntimeError('nonfinite independent truth')
   with np.load(case['observed_path'],allow_pickle=False) as z:
    if set(z.files)!={'observed'}:raise RuntimeError('test bank table schema')
    saved=np.asarray(z['observed'])
   if saved.shape!=observed.shape or saved.dtype!=observed.dtype or not np.array_equal(saved,observed):raise RuntimeError('shared observed table differs')
   if identity(fold)!=case['fold_identity']:raise RuntimeError('shared fold differs')
   cols=int(case['p']);table={'x':saved[:,:cols],'a':saved[:,cols],'m':saved[:,cols+1],'y':saved[:,cols+2]}
   row.update(common_ok=True,truth=truth,population_sha256=pop,observed=identity(saved),fold=identity(fold))
   initial,boot=reserved[idx]
   row.update(execution.run(table,fold,truth,initial,boot if planned else None,slots,core,baseline.crossfit_nuisance,evaluate.canonical_score,evaluate.bootstrap_refit_interval,planned))
  except Exception as exc:
   row.update(common_ok=False,error=repr(exc),traceback=traceback.format_exc()[-4000:]);row['methods']={'rf_dml_'+str(level):{'ok':False,'error':repr(exc)} for level in core.LEVELS}
   if planned:row['methods'].update({'rf_bootstrap_'+str(level):{'ok':False,'error':repr(exc)} for level in core.LEVELS})
  rows.append(row)
 result={'protocol_sha256':context['manifest_sha256'],'shard':context['shard'],'job_id':context['job_id'],'device':'cpu','slots':slots,'rows':rows}
 raw=json.dumps(result,sort_keys=True,allow_nan=False).encode();target=out/'rf_evaluation.json'
 with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'records':len(rows),'common_failures':sum(not x['common_ok'] for x in rows),'sha256':sha(target),'device':'cpu'}

def production_entry():raise RuntimeError('requires frozen RF seed mapping, shared test bank audit and actual predecessor qacct')
if __name__=='__main__':production_entry()
