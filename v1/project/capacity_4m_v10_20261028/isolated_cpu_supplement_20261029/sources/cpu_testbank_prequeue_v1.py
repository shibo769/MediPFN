"""One frozen testbank array, submitted USER-HELD. Never releases a job."""
import hashlib,json,os,re,subprocess,types,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9/capacity_4m_v10_20261028/isolated_cpu_supplement_20261029')
class PrequeueError(RuntimeError):pass
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(args):return subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120).stdout
def save(path,doc):
 with Path(path).open('x',encoding='utf8') as f:json.dump(doc,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
def held_tasks(text,jid):
 seen=set();rows=[x.split() for x in text.splitlines() if x.split() and x.split()[0]==str(jid)]
 for row in rows:
  if len(row)!=9 or row[3:5]!=['shibo','hqw']:raise PrequeueError('not user-held array')
  match=re.fullmatch(r'([1-9][0-9]*)(?:-([1-9][0-9]*)(?::([1-9][0-9]*))?)?',row[8])
  if not match:raise PrequeueError('bad task range')
  start=int(match.group(1));end=int(match.group(2) or start);step=int(match.group(3) or 1);part=set(range(start,end+1,step))
  if end<start or seen&part:raise PrequeueError('overlapping tasks')
  seen.update(part)
 return seen

def submit(protocol_path,expected_sha):
 root=ROOT.resolve(strict=True);pp=Path(protocol_path).resolve(strict=True)
 if pp!=root/'protocols'/'testbank.json' or sha(pp)!=expected_sha:raise PrequeueError('protocol not frozen')
 p=json.loads(pp.read_text());spec=p['stages']['testbank'];sources=p['source_sha256']
 if p.get('status')!='FROZEN_CPU_STAGE' or p.get('scope')!='4m_only' or spec.get('kind')!='array' or spec.get('shards')!=list(range(1,297)):raise PrequeueError('invalid testbank specification')
 if p['stage_receipts']['testbank']!=str(root/'receipts'/'testbank.json'):raise PrequeueError('receipt path mismatch')
 receipt=Path(p['stage_receipts']['testbank']);intent=root/'receipts'/'testbank_intent.json';raw_receipt=root/'receipts'/'testbank_qsub_raw.json'
 if any(x.exists() for x in (receipt,intent,raw_receipt)):raise PrequeueError('previous attempt; reconcile first')
 workflow=(root/'sources'/'cpu_simple_workflow_v5.py').resolve(strict=True);launcher=(root/'sources'/'cpu_stage_launch_v5.py').resolve(strict=True);wrapper=(root/'sources'/'cpu_stage_launch_v5.sh').resolve(strict=True)
 for path in (workflow,launcher,wrapper,Path(__file__).resolve(strict=True)):
  if sources.get(str(path))!=sha(path):raise PrequeueError('source drift')
 for path,digest in sources.items():
  if sha(path)!=digest:raise PrequeueError('frozen source changed')
 for path,digest in p['artifact_sha256'].items():
  if sha(path)!=digest:raise PrequeueError('frozen artifact drift')
 raw=workflow.read_bytes();wf=types.ModuleType('testbank_prequeue_workflow');wf.__file__=str(workflow);exec(compile(raw,str(workflow),'exec'),wf.__dict__)
 for item in spec['input_paths']:
  if p['input_sha256'].get(item)!=sha(item):raise PrequeueError('testbank input drift')
 wf.validate_case_plans(p,root,'testbank');wf.require_stage_predecessors(p,'testbank');wf.check_predecessors(spec)
 def load(name):
  path=(root/'sources'/name).resolve(strict=True);raw=path.read_bytes()
  if sources.get(str(path))!=hashlib.sha256(raw).hexdigest():raise PrequeueError('RF validator source drift')
  mod=types.ModuleType(name);mod.__file__=str(path);exec(compile(raw,str(path),'exec'),mod.__dict__);return mod
 binding=load('rf_seed_binding_v1.py');core=load('rf_supplement_core.py')
 cases=json.loads(Path(p['rf_seed_cases_path']).read_text())['cases'];mapping=json.loads(Path(p['rf_seed_mapping_path']).read_text())
 chosen=core.select_subset(cases,54,45)
 if chosen!=p['rf_bootstrap_indices'] or len(chosen)!=261:raise PrequeueError('RF subset mismatch')
 binding.validate(mapping,cases,chosen)
 existing=ET.fromstring(run(['qstat','-xml','-u','shibo']))
 if any((node.text or '')==spec['jobname'] for node in existing.iter('JB_name')):raise PrequeueError('job name exists; reconcile')
 args=['testbank',sha(wrapper),sha(launcher),sha(workflow),expected_sha]
 slots=spec['pe_slots'];resources=spec['qsub_resources'];logdir=root/'logs'
 if not logdir.is_dir() or type(slots) is not int or slots<1:raise PrequeueError('stage resources')
 argv=['qsub','-h','-terse','-P','tmd-ortho','-wd',str(root),'-pe','omp',str(slots),'-l',resources,'-t','1-296:1','-j','y','-o',str(logdir),'-N',spec['jobname'],str(wrapper)]+args
 save(intent,{'protocol_sha256':expected_sha,'argv':argv,'authorization':False})
 out=run(argv).strip();m=re.fullmatch(r'([1-9][0-9]*)(?:\.[0-9:-]+)?',out)
 if not m:raise PrequeueError('ambiguous qsub; retain intent')
 jid=int(m.group(1));save(raw_receipt,{'job_id':jid,'raw':out,'protocol_sha256':expected_sha})
 fields={}
 for line in run(['qstat','-j',str(jid)]).splitlines():
  key,sep,value=line.partition(':');key=key.strip()
  if sep and key in ('job_number','owner','job_name','script_file','cwd','job_args','hard resource_list','parallel environment'):
   if key in fields:raise PrequeueError('duplicate field')
   fields[key]=value.strip()
 parallel=fields.pop('parallel environment',None);resource=fields.pop('hard resource_list','');actual=dict(item.strip().split('=',1) for item in resource.split(',') if '=' in item)
 expected={'job_number':str(jid),'owner':'shibo','job_name':spec['jobname'],'script_file':str(wrapper),'cwd':str(root),'job_args':','.join(args)}
 if fields!=expected or parallel!='omp range: '+str(slots) or any(actual.get(k)!=str(v) for k,v in spec['expected_qstat_resources'].items()):raise PrequeueError('job identity/resources mismatch')
 if held_tasks(run(['qstat','-s','hu','-u','shibo']),jid)!=set(range(1,297)):raise PrequeueError('array not fully user-held')
 doc={'status':'ACTUAL_QSUB_RECEIPT','stage':'testbank','job_id':jid,'jobname':spec['jobname'],'kind':'array','shards':spec['shards'],'protocol_sha256':expected_sha,'wrapper_sha256':sha(wrapper),'launcher_sha256':sha(launcher),'workflow_sha256':sha(workflow),'user_hold_verified':True,'authorization':False}
 save(receipt,doc);return doc

def production_entry():raise PrequeueError('requires independently reviewed frozen protocol; never auto-releases')
if __name__=='__main__':production_entry()
