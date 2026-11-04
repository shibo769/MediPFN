"""Submit a reviewed singleton CPU calibration follow-up USER-HELD only."""
import hashlib,json,os,re,subprocess,sys,types,xml.etree.ElementTree as ET
from pathlib import Path
R=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9/capacity_4m_v10_20261028/isolated_cpu_supplement_20261029')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(args):return subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120).stdout
def save(path,doc):
 with Path(path).open('x') as f:json.dump(doc,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
def submit(stage,expected):
 if stage not in ('calibration_finalize','calibration_audit'):raise RuntimeError('unsupported follow-up')
 protocol=R/'protocols'/(stage+'.json')
 if len(expected)!=64 or sha(protocol)!=expected or (R/'protocols'/(stage+'.sha256')).read_text().strip()!=expected:raise RuntimeError('protocol mismatch')
 p=json.loads(protocol.read_text());spec=p['stages'][stage]
 if p.get('status')!='FROZEN_CPU_STAGE' or p.get('scope')!='4m_only' or spec.get('kind')!='singleton' or spec.get('shards')!=[1]:raise RuntimeError('follow-up scope')
 for mapping in (p['source_sha256'],p['artifact_sha256'],p['input_sha256']):
  for path,h in mapping.items():
   if sha(path)!=h:raise RuntimeError('frozen source/input drift '+path)
 source=Path(__file__).resolve(strict=True)
 if p['source_sha256'].get(str(source))!=sha(source):raise RuntimeError('prequeue source mismatch')
 workflow=R/'sources/cpu_simple_workflow_v3.py';raw=workflow.read_bytes();wf=types.ModuleType('followup_workflow');wf.__file__=str(workflow);exec(compile(raw,str(workflow),'exec'),wf.__dict__)
 wf.validate_case_plans(p,R,stage);wf.require_stage_predecessors(p,stage);wf.check_predecessors(spec)
 receipt=Path(p['stage_receipts'][stage]);intent=R/'receipts'/(stage+'_intent.json');raw_receipt=R/'receipts'/(stage+'_qsub_raw.json')
 if receipt!=R/'receipts'/(stage+'.json') or any(x.exists() for x in (receipt,intent,raw_receipt)):raise RuntimeError('follow-up already attempted')
 listing=ET.fromstring(run(['qstat','-xml','-u','shibo']))
 if any((node.text or '')==spec['jobname'] for node in listing.iter('JB_name')):raise RuntimeError('existing job name')
 wrapper=R/'sources/cpu_stage_launch_v3.sh';launcher=R/'sources/cpu_stage_launch_v3.py';sources=p['source_sha256'];args=[stage,sources[str(wrapper)],sources[str(launcher)],sources[str(workflow)],expected]
 argv=['qsub','-h','-terse','-P','tmd-ortho','-wd',str(R),'-pe','omp',str(spec['pe_slots']),'-l',spec['qsub_resources'],'-j','y','-o',str(R/'logs'),'-N',spec['jobname'],str(wrapper)]+args
 save(intent,{'protocol_sha256':expected,'argv':argv,'authorization':False});out=run(argv).strip()
 if not re.fullmatch(r'[1-9][0-9]*',out):raise RuntimeError('ambiguous qsub; reconcile')
 jid=int(out);save(raw_receipt,{'job_id':jid,'raw':out,'protocol_sha256':expected});fields={}
 for line in run(['qstat','-j',out]).splitlines():
  k,sep,v=line.partition(':')
  if sep and k in ('job_number','owner','job_name','script_file','cwd','job_args','hard resource_list','parallel environment'):
   if k in fields:raise RuntimeError('duplicate job field')
   fields[k]=v.strip()
 parallel=fields.pop('parallel environment',None);resource=fields.pop('hard resource_list','');actual=dict(x.strip().split('=',1) for x in resource.split(',') if '=' in x)
 want={'job_number':out,'owner':'shibo','job_name':spec['jobname'],'script_file':str(wrapper),'cwd':str(R),'job_args':','.join(args)}
 if fields!=want or parallel!='omp range: '+str(spec['pe_slots']) or any(actual.get(k)!=str(v) for k,v in spec['expected_qstat_resources'].items()):raise RuntimeError('job identity/resources')
 rows=[x.split() for x in run(['qstat','-s','hu','-u','shibo']).splitlines() if x.split() and x.split()[0]==out]
 if len(rows)!=1 or len(rows[0])!=8 or rows[0][3:5]!=['shibo','hqw']:raise RuntimeError('singleton user hold')
 doc={'status':'ACTUAL_QSUB_RECEIPT','stage':stage,'job_id':jid,'jobname':spec['jobname'],'kind':'singleton','shards':[1],'protocol_sha256':expected,'wrapper_sha256':sha(wrapper),'launcher_sha256':sha(launcher),'workflow_sha256':sha(workflow),'user_hold_verified':True,'authorization':False};save(receipt,doc);return doc
if __name__=='__main__':
 if len(sys.argv)!=3:raise RuntimeError('stage and actual protocol SHA required')
 print(json.dumps(submit(sys.argv[1],sys.argv[2]),sort_keys=True))
