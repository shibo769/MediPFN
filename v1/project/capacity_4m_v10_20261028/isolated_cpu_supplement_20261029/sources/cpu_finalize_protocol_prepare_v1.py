"""Freeze finalizer protocol only after all 40 real array tasks succeeded."""
import hashlib,json,os,subprocess
from pathlib import Path
B=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9');R=B/'capacity_4m_v10_20261028/isolated_cpu_supplement_20261029';BASE=R/'protocols/calibration.json';BASE_SHA='abaa8927bdecc725add36fc6b9b703590285fcaa85af021eec7071ff718cfa71';JOB=8340506
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def account():
 result=subprocess.run(['qacct','-j',str(JOB)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120)
 if result.returncode:raise RuntimeError('array accounting unavailable')
 rows={}
 for block in result.stdout.split('=============================================================='):
  fields={}
  for line in block.splitlines():
   x=line.split()
   if len(x)>=2 and x[0] in ('jobnumber','jobname','owner','taskid','failed','exit_status'):
    if x[0] in fields:raise RuntimeError('duplicate accounting field')
    fields[x[0]]=x[1]
  if not fields:continue
  try:task=int(fields['taskid'])
  except (KeyError,ValueError):raise RuntimeError('array accounting task')
  if task in rows or fields!={'jobnumber':str(JOB),'jobname':'nie4m_cpu_cal_v1','owner':'shibo','taskid':str(task),'failed':'0','exit_status':'0'}:raise RuntimeError('failed or mismatched shard')
  rows[task]=fields
 if set(rows)!=set(range(1,41)):raise RuntimeError('not all 40 tasks accounted successfully')
 return rows

def prepare():
 if sha(BASE)!=BASE_SHA:raise RuntimeError('calibration protocol drift')
 p=json.loads(BASE.read_text());receipt=R/'receipts/calibration.json';rec=json.loads(receipt.read_text())
 if rec.get('job_id')!=JOB or rec.get('protocol_sha256')!=BASE_SHA:raise RuntimeError('calibration submission receipt')
 account();plan=json.loads(Path(p['case_plans']['calibration']['path']).read_text());cases={x['case_id']:x for x in plan['cases']}
 if len(cases)!=400:raise RuntimeError('calibration plan size')
 outputs={}
 for task in range(1,41):
  out=R/'outputs/calibration'/('shard_%05d'%task);record=out/'stage_record.json';npz=out/'calibration.npz';info=json.loads(record.read_text());ids=p['stages']['calibration']['case_ids_by_shard'][str(task)]
  if (info.get('status')!='STAGE_COMPLETED_NOT_INDEPENDENTLY_AUDITED' or info.get('job_id')!=JOB or info.get('scheduler_task_id')!=task or info.get('manifest_sha256')!=BASE_SHA or info.get('result',{}).get('case_ids')!=ids or len(ids)!=10 or any(x not in cases for x in ids) or info.get('result',{}).get('npz_sha256')!=sha(npz)):raise RuntimeError('calibration shard provenance '+str(task))
  for path in (record,npz):outputs[str(path)]=sha(path)
 for path,h in p['source_sha256'].items():
  if sha(path)!=h:raise RuntimeError('source drift')
 for path,h in p['artifact_sha256'].items():
  if sha(path)!=h:raise RuntimeError('artifact drift')
 source=Path(__file__).resolve(strict=True);prequeue=R/'sources/cpu_followup_prequeue_v1.py';p['source_sha256'][str(source)]=sha(source);p['source_sha256'][str(prequeue)]=sha(prequeue)
 p['input_sha256'].update(outputs);p['calibration_stage_manifest_sha256']=BASE_SHA;p['stages']['calibration']['job_id']=JOB;p['stage_receipts']['calibration_finalize']=str(R/'receipts/calibration_finalize.json')
 d=B/'capacity_4m_v10_20261028';p['model_assets']['confidence']=str(d/'confidence_calibration_4m_20261028/fitted/confidence.pt');p['model_assets']['trained_levels_freeze']=str(d/'multilevel_4m_20261028/trained_heads/freeze.json')
 for path in (p['model_assets']['confidence'],p['model_assets']['trained_levels_freeze']):
  if p['artifact_sha256'].get(path)!=sha(path):raise RuntimeError('finalizer model asset mismatch')
 modules=[{'name':n,'path':p['project_import_paths'][n]} for n in ('data','calibration','alpha_heads','alpha_truth_v2')]
 p['stages']['calibration_finalize']={'worker_source':str(R/'sources/cpu_calibration_finalize_stage_v1.py'),'jobname':'nie4m_cpu_calfinal_v1','kind':'singleton','shards':[1],'pe_slots':4,'qsub_resources':'no_gpu=TRUE,mem_per_core=4G,h_rt=02:00:00','expected_qstat_resources':{'no_gpu':'TRUE','mem_per_core':'4G','h_rt':'7200'},'science_imports':modules,'input_paths':[p['case_plans']['calibration']['path']]+sorted(outputs),'predecessors':[{'job':JOB,'jobname':'nie4m_cpu_cal_v1','tasks':list(range(1,41)),'outputs':outputs}]}
 target=R/'protocols/calibration_finalize.json';digest=R/'protocols/calibration_finalize.sha256'
 if target.exists() or digest.exists():raise RuntimeError('finalizer protocol exists')
 (R/'outputs/calibration_finalize').mkdir(exist_ok=True)
 with target.open('x') as f:json.dump(p,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 h=sha(target)
 with digest.open('x') as f:f.write(h+'\n');f.flush();os.fsync(f.fileno())
 return {'status':'FINALIZER_PROTOCOL_FROZEN_NOT_SUBMITTED','protocol':str(target),'sha256':h,'array_job':JOB,'accounted_tasks':40,'output_files':len(outputs)}
if __name__=='__main__':print(json.dumps(prepare(),sort_keys=True))
