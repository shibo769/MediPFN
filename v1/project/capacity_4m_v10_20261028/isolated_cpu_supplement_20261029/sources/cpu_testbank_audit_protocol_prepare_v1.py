"""Prepare an independent CPU shared-table audit only after actual completion.

This reads only the fixed protocol, scheduler accounting, and generated
testbank artifacts after every one of 296 tasks has succeeded. It cannot create
test data, select methods, submit or release jobs. The caller must independently
review this source and all actual predecessor/output identities before use.
"""
import hashlib,json,os,subprocess,types
from pathlib import Path
R=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9/capacity_4m_v10_20261028/isolated_cpu_supplement_20261029');BASE=R/'protocols/testbank.json';BASE_SHA='b60494991f5c1da2542cfc7ec3a3de79ef7e687475b44031aac3b1a82d39b5ae';JOB=8342075
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def accounting():
 q=subprocess.run(['qacct','-j',str(JOB)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=180)
 if q.returncode:raise RuntimeError('testbank qacct unavailable')
 seen=set()
 for block in q.stdout.split('=============================================================='):
  fields={}
  for line in block.splitlines():
   x=line.split()
   if len(x)>=2 and x[0] in ('jobnumber','jobname','owner','taskid','failed','exit_status'):
    if x[0] in fields:raise RuntimeError('ambiguous qacct field')
    fields[x[0]]=x[1]
  if not fields:continue
  try:task=int(fields['taskid'])
  except (KeyError,ValueError):raise RuntimeError('missing task accounting')
  if task in seen or task not in range(1,297) or fields!={'jobnumber':str(JOB),'jobname':'nie4m_cpu_testbank_v1','owner':'shibo','taskid':str(task),'failed':'0','exit_status':'0'}:raise RuntimeError('testbank task failed or identity mismatch')
  seen.add(task)
 if seen!=set(range(1,297)):raise RuntimeError('testbank qacct incomplete')
def prepare():
 if sha(BASE)!=BASE_SHA:raise RuntimeError('testbank protocol changed')
 p=json.loads(BASE.read_text());rec=json.loads((R/'receipts/testbank.json').read_text())
 if rec.get('job_id')!=JOB or rec.get('protocol_sha256')!=BASE_SHA:raise RuntimeError('testbank receipt mismatch')
 accounting();plan=json.loads(Path(p['case_plans']['test']['path']).read_text())['cases'];byid={x['case_id']:x for x in plan}
 if len(byid)!=2952:raise RuntimeError('test plan size or identity')
 outputs={};tables={};seen=set();failures=0
 for task in range(1,297):
  d=R/'outputs/testbank'/('shard_%05d'%task);record=d/'stage_record.json';report=d/'testbank.json';r=json.loads(record.read_text());x=json.loads(report.read_text());ids=p['stages']['testbank']['case_ids_by_shard'][str(task)]
  if (r.get('status')!='STAGE_COMPLETED_NOT_INDEPENDENTLY_AUDITED' or r.get('stage')!='testbank' or r.get('job_id')!=JOB or r.get('scheduler_task_id')!=task or r.get('manifest_sha256')!=BASE_SHA or r.get('result',{}).get('testbank_report_sha256')!=sha(report) or x.get('status')!='TESTBANK_SHARD_WRITTEN_NOT_AUDITED' or x.get('job_id')!=JOB or x.get('task')!=task or x.get('expected')!=len(ids) or [z.get('case_id') for z in x.get('cases',[])]!=ids):raise RuntimeError('testbank shard identity '+str(task))
  if x.get('failed')!=sum(z.get('status')=='failed' for z in x['cases']):raise RuntimeError('failure count mismatch')
  for row in x['cases']:
   cid=row['case_id']
   if cid in seen or cid not in byid:raise RuntimeError('duplicate or unplanned case')
   seen.add(cid)
   if row.get('status')=='failed':failures+=1;continue
   if row.get('status')!='ok':raise RuntimeError('unrecognized case status')
   path=R/'tables'/('shard_%05d'%task)/(cid+'.npz')
   if row.get('observed_path')!=str(path) or row.get('observed_file_sha256')!=sha(path):raise RuntimeError('testbank table drift')
   tables[str(path)]=row['observed_file_sha256']
  outputs[str(record)]=sha(record);outputs[str(report)]=sha(report)
 if seen!=set(byid) or len(tables)+failures!=2952:raise RuntimeError('testbank attempts incomplete')
 for group in ('source_sha256','artifact_sha256','input_sha256'):
  for path,h in p[group].items():
   if sha(path)!=h:raise RuntimeError('frozen input drift '+path)
 source=Path(__file__).resolve(strict=True);worker=R/'sources/cpu_testbank_audit_stage_v2.py';workflow=R/'sources/cpu_simple_workflow_v5.py';prequeue=R/'sources/cpu_testbank_audit_prequeue_v1.py'
 if sha(worker)!='bd56b376b00a90e12a08abca3191b9bbb39b9dd48628b5aece5ad39e7e25ae8d':raise RuntimeError('audit worker source mismatch')
 for path in (source,worker,prequeue):p['source_sha256'][str(path)]=sha(path)
 p['input_sha256'].update(outputs);p['input_sha256'].update(tables);p['stages']['testbank']['job_id']=JOB;p['testbank_stage_manifest_sha256']=BASE_SHA;p['stage_receipts']['testbank_audit']=str(R/'receipts/testbank_audit.json')
 loader=p['stages']['testbank']['science_imports'][0]
 if loader.get('name')!='cpu_case_loader':raise RuntimeError('loader order mismatch')
 p['stages']['testbank_audit']={'worker_source':str(worker),'jobname':'nie4m_cpu_tbaudit_v1','kind':'singleton','shards':[1],'pe_slots':4,'qsub_resources':'no_gpu=TRUE,mem_per_core=4G,h_rt=24:00:00','expected_qstat_resources':{'no_gpu':'TRUE','mem_per_core':'4G','h_rt':'86400'},'science_imports':[loader],'input_paths':[p['case_plans']['test']['path']]+sorted(outputs),'predecessors':[{'job':JOB,'jobname':p['stages']['testbank']['jobname'],'tasks':list(range(1,297)),'outputs':outputs}],'case_ids_by_shard':{'1':[x['case_id'] for x in plan]}}
 raw=workflow.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=p['source_sha256'].get(str(workflow)):raise RuntimeError('workflow source mismatch')
 wf=types.ModuleType('cpu_testbank_audit_workflow');wf.__file__=str(workflow);exec(compile(raw,str(workflow),'exec'),wf.__dict__);wf.validate_case_plans(p,R,'testbank_audit');wf.require_stage_predecessors(p,'testbank_audit');wf.check_predecessors(p['stages']['testbank_audit'])
 target=R/'protocols/testbank_audit.json';digest=R/'protocols/testbank_audit.sha256';outdir=R/'outputs/testbank_audit'
 if target.exists() or digest.exists() or outdir.exists() or (R/'receipts/testbank_audit_intent.json').exists():raise RuntimeError('existing audit attempt; reconcile')
 outdir.mkdir()
 with target.open('x') as f:json.dump(p,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 h=sha(target)
 with digest.open('x') as f:f.write(h+'\n');f.flush();os.fsync(f.fileno())
 return {'status':'TESTBANK_AUDIT_PROTOCOL_PREPARED_NOT_SUBMITTED','sha256':h,'testbank_job':JOB,'tasks':296,'expected_cases':2952,'failures':failures,'table_count':len(tables)}
if __name__=='__main__':raise RuntimeError('requires actual 296 successful tasks and independently reviewed invocation')
