"""Generate preregistered shared CPU test tables in a new isolated root.

Runs only under a frozen source-bound study manifest. No model or result selection.
The caller must account for all failed cases and freeze the generated inputs before
any method reads them. This draft is not an authorization to run.
"""
import hashlib,json,os,types
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(array):
 array=np.ascontiguousarray(array)
 return {'sha256':hashlib.sha256(array.tobytes()).hexdigest(),'dtype':str(array.dtype),'shape':list(array.shape)}

def run(context):
 p=context['manifest'];root=Path(p['output_root']).resolve(strict=True);stage=context['stage'];out=Path(context['output_dir']).resolve(strict=True)
 if stage!='testbank' or p.get('scope')!='4m_only' or not out.is_relative_to(root/'outputs'/'testbank'):raise RuntimeError('wrong testbank scope')
 helper_path=Path(p['science_imports_source']).resolve(strict=True);raw=helper_path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=p['source_sha256'].get(str(helper_path)):raise RuntimeError('import helper changed')
 helper=types.ModuleType('cpu_testbank_imports');helper.__file__=str(helper_path);exec(compile(raw,str(helper_path),'exec'),helper.__dict__)
 helper.verify_calibration_audit(p)
 cases=helper.selected_cases(p,'testbank',context['shard'],'test')
 xroot=Path(p['covariate_root']).resolve(strict=True)
 if not xroot.is_relative_to(root):raise RuntimeError('covariates outside isolated root')
 os.environ['NIE_WAVE_XDIR']=str(xroot)
 tb=[x for x in p['stages']['testbank']['science_imports'] if x['name']=='cpu_case_loader']
 ev=[x for x in p['stages']['evaluation']['science_imports'] if x['name']=='cpu_case_loader']
 if len(tb)!=1 or len(ev)!=1 or Path(tb[0]['path']).resolve(strict=True)!=Path(ev[0]['path']).resolve(strict=True):raise RuntimeError('testbank and evaluation loader differ')
 modules=helper.load_registered(p,'testbank');loader=modules['cpu_case_loader']
 bank=root/'tables';bank.mkdir(exist_ok=True);shard=bank/('shard_%05d'%context['shard']);shard.mkdir()
 rows=[]
 for c in cases:
  cid=c['case_id']
  if not isinstance(cid,str) or not cid or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for ch in cid):raise RuntimeError('invalid case ID')
  row={'case_id':cid,'status':'failed'}
  try:
   observed,fold,truth,pop=loader.make_case(c)
   observed=np.asarray(observed);fold=np.asarray(fold)
   if not np.isfinite(float(truth)) or not np.isfinite(observed).all():raise ValueError('nonfinite true effect or table')
   path=shard/(cid+'.npz')
   with path.open('xb') as f:np.savez_compressed(f,observed=observed);f.flush();os.fsync(f.fileno())
   row.update({'status':'ok','observed_path':str(path),'observed_file_sha256':sha(path),'observation_identity':identity(observed),'fold_identity':identity(fold),'population_sha256':pop})
  except Exception as exc:row['error_type']=type(exc).__name__;row['error']=str(exc)
  rows.append(row)
 report={'status':'TESTBANK_SHARD_WRITTEN_NOT_AUDITED','job_id':context['job_id'],'task':context['shard'],'cases':rows,'expected':len(cases),'failed':sum(x['status']!='ok' for x in rows)}
 result=out/'testbank.json'
 with result.open('x',encoding='utf8') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 return {'testbank_report':str(result),'testbank_report_sha256':sha(result),'expected':len(cases),'failed':report['failed']}

def production_entry():raise RuntimeError('requires independently reviewed frozen CPU protocol and source-bound scheduler execution')
if __name__=='__main__':production_entry()
