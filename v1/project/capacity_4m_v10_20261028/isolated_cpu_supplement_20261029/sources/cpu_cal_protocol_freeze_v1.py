"""Freeze the user-requested 4M CPU calibration protocol from verified facts.
Only metadata is written; this does not submit or release any job.
"""
import hashlib,json,os,subprocess,types,xml.etree.ElementTree as ET
from pathlib import Path
B=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9');R=B/'capacity_4m_v10_20261028/isolated_cpu_supplement_20261029';DRAFT=R/'drafts/cpu_calibration_candidate_v1.json';EV=R/'scoped_evidence_v1';SNAP=B/'audit/cpu_recovery_writer_census_20261029_v3/snapshot.json';FIX=B/'audit/cpu_cal_fictional_stage_20261029_v1';PAR=B/'audit/cpu_gpu_fictional_posttrain_parity_20261029_v6/result.json'
EXPECTED={'draft':'75f5dd029045ea2cd1c864be977a402f8873bcd52777ca8a47d1010210b78583','evidence':'518b0e513266ae34cb5b7f6efabd3ff8ecf822b830a88f19845037979b6ff244','review':'2637ae90c5b4c813fb8b81e14a8cc19692433134af48e53400545960758e74f4','snapshot':'8c27d7082b03c2dbaa176c7f119352f028888f85cfa80b34bb62620cf7d28639','fixture':'3333f008282ab02e9e34d320c7900e0c8e0506fe78d96768640dd012837a4c11','parity':'ba0b2c2b8cf7e6214c16223ad3124e11c5901ee83c6e9fe26a72a8b6c67ad54e'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=90)
 if p.returncode:raise RuntimeError((args,p.returncode,p.stderr))
 return p.stdout
def account(job,name,exitcode):
 fields={}
 for line in command(['qacct','-j',str(job)]).splitlines():
  x=line.split()
  if len(x)>=2 and x[0] in ('jobnumber','jobname','owner','taskid','failed','exit_status'):
   if x[0] in fields:raise RuntimeError('ambiguous accounting')
   fields[x[0]]=x[1]
 if fields!={'jobnumber':str(job),'jobname':name,'owner':'shibo','taskid':'undefined','failed':'0','exit_status':str(exitcode)}:raise RuntimeError('accounting mismatch '+str(job))
def listing():
 xml=command(['qstat','-xml','-u','shibo']);root=ET.fromstring(xml);jobs={}
 for item in root.iter('job_list'):
  jid=int(item.findtext('JB_job_number'));name=item.findtext('JB_name') or '';state=item.findtext('state') or ''
  if name.lower().startswith(('nie4m','nie10bootspeed')) or jid==8299223:
   jobs.setdefault(jid,{'name':name,'states':set()})['states'].add(state)
 return {k:{'name':v['name'],'states':sorted(v['states'])} for k,v in jobs.items()}
def freeze():
 files={'draft':DRAFT,'evidence':EV/'cal400_scoped_scientific_evidence_draft_v2.json','review':EV/'cal400_scoped_scientific_review_v1.json','snapshot':SNAP,'fixture':FIX/'result.json','parity':PAR}
 for k,path in files.items():
  if sha(path)!=EXPECTED[k]:raise RuntimeError('input changed '+k)
 p=json.loads(DRAFT.read_text());e=json.loads(files['evidence'].read_text());review=json.loads(files['review'].read_text());snapshot=json.loads(SNAP.read_text());fixture=json.loads(files['fixture'].read_text());parity=json.loads(PAR.read_text())
 if p.get('status')!='INCOMPLETE_CPU_DRAFT_NOT_AUTHORIZATION' or e.get('status')!='DRAFT_REQUIRES_INDEPENDENT_REVIEW_NOT_AUTHORIZATION' or review.get('status')!='SCOPED_LOCAL_REVIEW_NOT_AUTHORIZATION' or not snapshot.get('listing_stable'):raise RuntimeError('evidence status')
 if fixture.get('status')!='CPU_FICTIONAL_CALIBRATION_STAGE_PASS' or fixture.get('job_id')!=8340425 or fixture.get('count')!=10 or sha(FIX/'output/calibration.npz')!=fixture.get('output_sha256'):raise RuntimeError('CPU fixture mismatch')
 if parity.get('status')!='FAIL' or parity.get('job_id')!=8340422:raise RuntimeError('device result mismatch')
 account(8340425,'nie4m_cpu_calfixture_v1',0);account(8340422,'nie4m_gpu_cpu_head_v6',1)
 for mapping in (p['source_sha256'],p['artifact_sha256'],p['input_sha256']):
  for path,h in mapping.items():
   if sha(path)!=h:raise RuntimeError('frozen source/input mismatch '+path)
 wfpath=R/'sources/cpu_simple_workflow_v3.py';raw=wfpath.read_bytes();wf=types.ModuleType('freeze_workflow');wf.__file__=str(wfpath);exec(compile(raw,str(wfpath),'exec'),wf.__dict__);wf.validate_case_plans(p,R,'calibration');wf.check_predecessors(p['stages']['calibration'])
 original={int(x['job']):x for x in snapshot['jobs']};before=listing()
 for jid,job in before.items():
  if jid in (8340422,8340425):continue
  if jid not in original:raise RuntimeError('new selected job requires review '+str(jid))
  allowed=['qw','r'] if jid==8299223 else (['s'] if jid==8338294 else ['hqw'])
  if job['name']!=original[jid]['name'] or any(x not in allowed for x in job['states']):raise RuntimeError('selected writer state changed '+str(jid))
 if 8299223 in before:
  fields={}
  for line in command(['qstat','-j','8299223']).splitlines():
   k,sep,v=line.partition(':')
   if sep and k in ('owner','job_name','script_file','cwd'):fields[k]=v.strip()
  if fields.get('owner')!='shibo' or fields.get('job_name')!='nie10bootspeed':raise RuntimeError('retained job identity')
  script=Path(fields['script_file']);script=script if script.is_absolute() else Path(fields['cwd'])/script
  if sha(script)!='f1ba0e1ac506404320d7f2461b1f745ae03b5d029953f02b06d545cdc0643087':raise RuntimeError('retained job source drift')
 after=listing()
 if before!=after:raise RuntimeError('selected writer listing changed during check')
 p['status']='FROZEN_CPU_STAGE';p.pop('authorization',None);p.pop('required_missing',None)
 p['scientific_scope']={'device':'cpu','model':'frozen 4M','target':'400 new independent calibration mechanisms','seed_review_sha256':EXPECTED['review'],'evidence_sha256':EXPECTED['evidence'],'selected_writer_snapshot_sha256':EXPECTED['snapshot'],'selected_writer_check':before,'device_parity_result_sha256':EXPECTED['parity'],'device_parity_status':'FAIL_IN_SINGLE_FICTIONAL_TABLE','limitations':['Named retained sources and specified plans only; e65 old execution bytes and old target spool not recovered.','268 other same-owner jobs were unclassified in the snapshot; current checks cover selected task names and retained 8299223, not all possible indirect writers.','CPU and GPU are not claimed numerically equivalent.']}
 p['source_sha256'][str(Path(__file__).resolve())]=sha(Path(__file__))
 for key in ('evidence','review','snapshot','fixture','parity'):p['input_sha256'][str(files[key])]=EXPECTED[key]
 spec=p['stages']['calibration'];spec['input_paths']+= [str(files[k]) for k in ('evidence','review','snapshot','fixture','parity')];spec['predecessors'].append({'job':8340425,'jobname':'nie4m_cpu_calfixture_v1','tasks':[0],'outputs':{str(files['fixture']):EXPECTED['fixture']}})
 for directory in (R/'protocols',R/'receipts',R/'logs',R/'outputs',R/'outputs/calibration'):directory.mkdir(exist_ok=True)
 target=R/'protocols/calibration.json';digest=R/'protocols/calibration.sha256'
 if target.exists() or digest.exists() or (R/'receipts/calibration_intent.json').exists():raise RuntimeError('protocol or submission already exists; reconcile')
 with target.open('x') as f:json.dump(p,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 h=sha(target)
 with digest.open('x') as f:f.write(h+'\n');f.flush();os.fsync(f.fileno())
 return {'status':'CPU_CALIBRATION_PROTOCOL_FROZEN_NOT_SUBMITTED','sha256':h,'protocol':str(target),'selected_jobs':len(before),'limitations':p['scientific_scope']['limitations']}
if __name__=='__main__':print(json.dumps(freeze(),sort_keys=True))
