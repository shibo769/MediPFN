"""Run one stage from its finalized isolated protocol and SGE identity."""
import hashlib,json,os,sys,types
from pathlib import Path
ROOT=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9/capacity_4m_v10_20261028/isolated_cpu_supplement_20261029')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 if len(sys.argv)!=5:raise RuntimeError('stage, workflow SHA, launcher SHA and protocol SHA required')
 stage,expected_workflow,expected_self,expected_protocol=sys.argv[1:]
 if stage not in ('calibration','calibration_finalize','calibration_audit','testbank','testbank_audit','evaluation','rf','timing'):raise RuntimeError('unknown stage')
 own=Path(__file__).resolve(strict=True);root=ROOT.resolve(strict=True)
 if own!=root/'sources'/'cpu_stage_launch_v5.py' or sha(own.read_bytes())!=expected_self:raise RuntimeError('launcher source mismatch')
 wf=root/'sources'/'cpu_simple_workflow_v5.py';raw=wf.read_bytes()
 if sha(raw)!=expected_workflow:raise RuntimeError('workflow source mismatch')
 protocol=root/'protocols'/(stage+'.json');digest_file=root/'protocols'/(stage+'.sha256')
 if not protocol.is_file() or not digest_file.is_file():raise RuntimeError('frozen stage protocol missing')
 expected=digest_file.read_text(encoding='utf8').strip()
 if expected!=expected_protocol or len(expected)!=64 or any(x not in '0123456789abcdef' for x in expected) or sha(protocol.read_bytes())!=expected:raise RuntimeError('stage protocol SHA mismatch')
 m=types.ModuleType('cpu_science_workflow');m.__file__=str(wf);exec(compile(raw,str(wf),'exec'),m.__dict__)
 task=os.environ.get('SGE_TASK_ID');shard=1 if task in (None,'','undefined') else int(task)
 print('CPU_STAGE_SOURCE_AND_PROTOCOL',os.environ.get('JOB_ID',''),stage,expected_workflow,expected_self,expected,flush=True)
 result=m.run_stage(str(protocol),expected,stage,shard)
 print('CPU_STAGE_FINISHED',os.environ['JOB_ID'],stage,shard,result['status'],flush=True)
if __name__=='__main__':main()
