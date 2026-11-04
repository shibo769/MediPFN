"""Independently reconstruct the frozen shared tables; fictional tests only so far."""
import hashlib,json,os,types
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(a):
 a=np.ascontiguousarray(a);return {'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'dtype':str(a.dtype),'shape':list(a.shape)}
def run(context):
 p=context['manifest'];root=Path(p['output_root']).resolve(strict=True);out=Path(context['output_dir']).resolve(strict=True)
 if context['stage']!='testbank_audit' or not out.is_relative_to(root/'outputs'/'testbank_audit'):raise RuntimeError('invalid testbank audit scope')
 helper_path=Path(p['science_imports_source']).resolve(strict=True);raw=helper_path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=p['source_sha256'].get(str(helper_path)):raise RuntimeError('import helper changed')
 helper=types.ModuleType('cpu_testbank_audit_imports');helper.__file__=str(helper_path);exec(compile(raw,str(helper_path),'exec'),helper.__dict__)
 cases=helper.selected_cases(p,'testbank_audit',context['shard'],'test');byid={x['case_id']:x for x in cases}
 xroot=Path(p['covariate_root']).resolve(strict=True)
 if not xroot.is_relative_to(root):raise RuntimeError('covariates outside isolated root')
 os.environ['NIE_WAVE_XDIR']=str(xroot)
 paths=lambda stage:[Path(x['path']).resolve(strict=True) for x in p['stages'][stage]['science_imports'] if x['name']=='cpu_case_loader']
 if len(paths('testbank_audit'))!=1 or paths('testbank_audit')!=paths('testbank') or paths('testbank')!=paths('evaluation'):raise RuntimeError('loader mismatch')
 loader=helper.load_registered(p,'testbank_audit')['cpu_case_loader'];tb=p['stages']['testbank'];seen=set();failed=[];verified=0
 for task in tb['shards']:
  report=root/'outputs'/'testbank'/('shard_%05d'%task)/'testbank.json'
  if p['input_sha256'].get(str(report))!=sha(report):raise RuntimeError('testbank report hash mismatch')
  doc=json.loads(report.read_text())
  if doc.get('status')!='TESTBANK_SHARD_WRITTEN_NOT_AUDITED' or doc.get('job_id')!=tb['job_id'] or doc.get('task')!=task or doc.get('expected')!=len(doc['cases']):raise RuntimeError('testbank report identity')
  for row in doc['cases']:
   cid=row['case_id']
   if cid in seen or cid not in byid:raise RuntimeError('duplicate/unplanned case')
   seen.add(cid)
   if row['status']=='failed':failed.append(cid);continue
   if row['status']!='ok':raise RuntimeError('unknown table status')
   path=Path(row['observed_path']).resolve(strict=True);expected=root/'tables'/('shard_%05d'%task)/(cid+'.npz')
   if path!=expected or sha(path)!=row['observed_file_sha256']:raise RuntimeError('testbank table hash mismatch')
   with np.load(path,allow_pickle=False) as z:
    if set(z.files)!={'observed'}:raise RuntimeError('testbank table schema')
    saved=np.asarray(z['observed'])
   observed,fold,truth,pop=loader.make_case(byid[cid])
   if not np.isfinite(float(truth)) or not np.array_equal(observed,saved) or identity(observed)!=row['observation_identity'] or identity(fold)!=row['fold_identity'] or pop!=row['population_sha256']:raise RuntimeError('testbank reconstruction mismatch')
   verified+=1
  if doc.get('failed')!=sum(x['status']=='failed' for x in doc['cases']):raise RuntimeError('testbank failure count mismatch')
 if seen!=set(byid):raise RuntimeError('testbank denominator incomplete')
 plan_id=hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 result={'status':'SCOPED_TESTBANK_RECONSTRUCTION_PASS','testbank_job':tb['job_id'],'case_plan_identity_sha256':plan_id,'expected':len(byid),'verified':verified,'failed_case_ids':sorted(failed),'limitations':['Same reviewed generator is rerun; this is reproducibility and pairing validation, not a second independent truth implementation.']}
 target=out/'testbank_audit.json'
 with target.open('x',encoding='utf8') as f:json.dump(result,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 return {'audit_sha256':sha(target),'expected':len(byid),'verified':verified,'failures':len(failed)}

def production_entry():raise RuntimeError('requires reviewed CPU protocol and actual predecessor accounting')
if __name__=='__main__':production_entry()
