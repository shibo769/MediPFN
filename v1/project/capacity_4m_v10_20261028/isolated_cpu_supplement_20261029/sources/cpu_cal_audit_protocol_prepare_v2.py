"""Freeze independent calibration-audit protocol after actual finalizer success."""
import hashlib,json,os,types
from pathlib import Path
B=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9');R=B/'capacity_4m_v10_20261028/isolated_cpu_supplement_20261029';PREV=R/'protocols/calibration_finalize.json';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
 prevsha=(R/'protocols/calibration_finalize.sha256').read_text().strip()
 if len(prevsha)!=64 or sha(PREV)!=prevsha:raise RuntimeError('finalizer protocol changed')
 p=json.loads(PREV.read_text());receipt=json.loads((R/'receipts/calibration_finalize.json').read_text());job=receipt.get('job_id')
 if type(job) is not int or job<=0 or receipt.get('status')!='ACTUAL_QSUB_RECEIPT' or receipt.get('stage')!='calibration_finalize' or receipt.get('protocol_sha256')!=prevsha:raise RuntimeError('finalizer receipt mismatch')
 finaldir=R/'outputs/calibration_finalize/shard_00001';record=finaldir/'stage_record.json';freeze=finaldir/'alpha_calibration_freeze.json';rec=json.loads(record.read_text());doc=json.loads(freeze.read_text())
 if (rec.get('status')!='STAGE_COMPLETED_NOT_INDEPENDENTLY_AUDITED' or rec.get('stage')!='calibration_finalize' or rec.get('job_id')!=job or rec.get('shard')!=1 or rec.get('manifest_sha256')!=prevsha or rec.get('result',{}).get('freeze_sha256')!=sha(freeze) or doc.get('status')!='PASS' or doc.get('protocol_sha256')!=prevsha):raise RuntimeError('finalizer output mismatch')
 outputs={str(record):sha(record),str(freeze):sha(freeze)};p['input_sha256'].update(outputs);p['artifact_sha256'][str(freeze)]=sha(freeze);p['model_assets']['levels_freeze']=str(freeze);p['calibration_freeze_path']=str(freeze);p['stages']['calibration_finalize']['job_id']=job;p['stage_receipts']['calibration_audit']=str(R/'receipts/calibration_audit.json')
 source=Path(__file__).resolve(strict=True);p['source_sha256'][str(source)]=sha(source)
 array=p['stages']['calibration_finalize']['predecessors'][0];spec={'worker_source':str(R/'sources/cpu_calibration_audit_stage_v1.py'),'jobname':'nie4m_cpu_calaudit_v1','kind':'singleton','shards':[1],'pe_slots':4,'qsub_resources':'no_gpu=TRUE,mem_per_core=4G,h_rt=02:00:00','expected_qstat_resources':{'no_gpu':'TRUE','mem_per_core':'4G','h_rt':'7200'},'science_imports':p['stages']['calibration_finalize']['science_imports'],'input_paths':[p['case_plans']['calibration']['path']]+sorted(array['outputs'])+sorted(outputs),'predecessors':[array,{'job':job,'jobname':p['stages']['calibration_finalize']['jobname'],'tasks':[0],'outputs':outputs}]};p['stages']['calibration_audit']=spec
 for mapping in (p['source_sha256'],p['artifact_sha256'],p['input_sha256']):
  for path,h in mapping.items():
   if sha(path)!=h:raise RuntimeError('frozen source/input drift '+path)
 workflow=R/'sources/cpu_simple_workflow_v3.py';raw=workflow.read_bytes();wf=types.ModuleType('audit_protocol_workflow');wf.__file__=str(workflow);exec(compile(raw,str(workflow),'exec'),wf.__dict__);wf.validate_case_plans(p,R,'calibration_audit');wf.require_stage_predecessors(p,'calibration_audit');wf.check_predecessors(spec)
 target=R/'protocols/calibration_audit.json';digest=R/'protocols/calibration_audit.sha256'
 if target.exists() or digest.exists():raise RuntimeError('audit protocol already exists')
 (R/'outputs/calibration_audit').mkdir(exist_ok=True)
 with target.open('x') as f:json.dump(p,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 h=sha(target)
 with digest.open('x') as f:f.write(h+'\n');f.flush();os.fsync(f.fileno())
 return {'status':'AUDIT_PROTOCOL_FROZEN_NOT_SUBMITTED','protocol':str(target),'sha256':h,'finalizer_job':job}
if __name__=='__main__':print(json.dumps(prepare(),sort_keys=True))
