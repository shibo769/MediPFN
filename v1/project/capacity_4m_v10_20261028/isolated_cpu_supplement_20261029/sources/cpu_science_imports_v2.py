"""Load reviewed project modules from pinned bytes in a fresh CPU process."""
import hashlib,sys,types
from pathlib import Path

def load_registered(protocol,stage):
 specs=protocol['stages'][stage]['science_imports'];sources=protocol['source_sha256']
 if not isinstance(specs,list) or not specs:raise RuntimeError('missing science imports')
 guard_path=Path(protocol['import_guard_source']).resolve(strict=True);guard_raw=guard_path.read_bytes()
 if hashlib.sha256(guard_raw).hexdigest()!=sources.get(str(guard_path)):raise RuntimeError('import guard source changed')
 guard=types.ModuleType('cpu_project_import_guard');guard.__file__=str(guard_path);exec(compile(guard_raw,str(guard_path),'exec'),guard.__dict__)
 for directory in reversed(protocol['project_import_dirs']):
  path=Path(directory).resolve(strict=True)
  if not path.is_relative_to(Path(protocol['project_root']).resolve(strict=True)):raise RuntimeError('project import directory outside project')
  sys.path.insert(0,str(path))
 guard.install(sources,protocol['project_import_paths'],protocol['project_root'],protocol['environment_root'])
 result={}
 for entry in specs:
  name=entry['name'];path=Path(entry['path']).resolve(strict=True)
  if not isinstance(name,str) or not name or name in result:raise RuntimeError('duplicate project module: '+str(name))
  raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=sources.get(str(path)):raise RuntimeError('unregistered science source: '+str(path))
  if name in sys.modules:
   cached=sys.modules[name];loader=getattr(getattr(cached,'__spec__',None),'loader',None)
   if (not isinstance(loader,guard.BoundLoader) or Path(getattr(cached,'__file__','')).resolve(strict=True)!=path or
       Path(loader.path).resolve(strict=True)!=path or hashlib.sha256(loader.raw).hexdigest()!=sources.get(str(path))):raise RuntimeError('unverified transitive project import: '+name)
   result[name]=cached;continue
  mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
  try:exec(compile(raw,str(path),'exec'),mod.__dict__)
  except BaseException:sys.modules.pop(name,None);raise
  result[name]=mod
 return result

def selected_cases(protocol,stage,shard,split=None):
 import json
 path=Path(protocol['case_plans'][split or stage]['path']);plan=json.loads(path.read_text())
 by_id={case['case_id']:case for case in plan['cases']};spec=protocol['stages'][stage];mapping=spec['case_ids_by_shard']
 if set(mapping)!={str(v) for v in spec['shards']}:raise RuntimeError('shard assignment incomplete')
 flat=[cid for group in mapping.values() for cid in group]
 if len(by_id)!=len(plan['cases']) or len(flat)!=len(set(flat)) or set(flat)!=set(by_id):raise RuntimeError('case assignment duplicate or incomplete')
 selected=mapping[str(shard)]
 if not selected or any(k not in by_id for k in selected):raise RuntimeError('invalid registered shard')
 return [by_id[k] for k in selected]

def asset(protocol,name):
 path=Path(protocol['model_assets'][name]).resolve(strict=True)
 if str(path) not in protocol['artifact_sha256']:raise RuntimeError('unregistered model asset '+name)
 return str(path)


def verify_calibration_audit(protocol):
 import json
 root=Path(protocol['output_root']).resolve(strict=True);directory=root/'outputs'/'calibration_audit'/'shard_00001';report=directory/'calibration_audit.json';record=directory/'stage_record.json'
 for path in (report,record):
  path=path.resolve(strict=True)
  if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['input_sha256'].get(str(path)):raise RuntimeError('calibration audit input changed')
 doc=json.loads(report.read_text());rec=json.loads(record.read_text());audit=protocol['stages']['calibration_audit'];freeze=Path(protocol['model_assets']['levels_freeze']).resolve(strict=True)
 if (doc.get('status')!='SCOPED_CPU_CALIBRATION_AUDIT_PASS' or doc.get('freeze_sha256')!=protocol['artifact_sha256'].get(str(freeze)) or
     doc.get('calibration_job')!=protocol['stages']['calibration']['job_id'] or doc.get('finalizer_job')!=protocol['stages']['calibration_finalize']['job_id'] or
     rec.get('status')!='STAGE_COMPLETED_NOT_INDEPENDENTLY_AUDITED' or rec.get('job_id')!=audit['job_id'] or
     rec.get('result',{}).get('audit_sha256')!=hashlib.sha256(report.read_bytes()).hexdigest()):raise RuntimeError('calibration audit not bound to selected freeze')
 return True


def verify_testbank_audit(protocol):
 import json
 root=Path(protocol['output_root']).resolve(strict=True);directory=root/'outputs'/'testbank_audit'/'shard_00001';report=(directory/'testbank_audit.json').resolve(strict=True);record=(directory/'stage_record.json').resolve(strict=True)
 for path in (report,record):
  if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['input_sha256'].get(str(path)):raise RuntimeError('testbank audit input changed')
 doc=json.loads(report.read_text());rec=json.loads(record.read_text());audit=protocol['stages']['testbank_audit'];testbank=protocol['stages']['testbank']
 plan=json.loads(Path(protocol['case_plans']['test']['path']).read_text());failed=sorted(c['case_id'] for c in plan['cases'] if c.get('testbank_status')=='failed')
 excluded={'observed_path','observed_file_sha256','fold_identity','testbank_status','testbank_report_path'}
 original=[{k:v for k,v in c.items() if k not in excluded} for c in plan['cases']]
 plan_id=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 if doc.get('case_plan_identity_sha256')!=plan_id:raise RuntimeError('test plan changed after table generation')
 if (doc.get('status')!='SCOPED_TESTBANK_RECONSTRUCTION_PASS' or doc.get('testbank_job')!=testbank['job_id'] or doc.get('expected')!=len(plan['cases']) or doc.get('failed_case_ids')!=failed or doc.get('verified')+len(failed)!=len(plan['cases']) or rec.get('status')!='STAGE_COMPLETED_NOT_INDEPENDENTLY_AUDITED' or rec.get('job_id')!=audit['job_id'] or rec.get('result',{}).get('audit_sha256')!=hashlib.sha256(report.read_bytes()).hexdigest()):raise RuntimeError('testbank audit does not bind case denominator')
 return True
