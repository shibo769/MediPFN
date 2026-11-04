"""Minimal CPU stage runner, source draft; no scheduler submission.

This draft checks predecessor qacct and frozen predecessor output hashes before
starting a stage. Independent review of the declared plans is still required. Scientific workers must implement run(context) and use only the
source-bound, manifest-declared inputs. No worker adapter is registered yet.
"""
import hashlib,json,os,types,subprocess
from pathlib import Path
OUTPUT_ROOT=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9/capacity_4m_v10_20261028/isolated_cpu_supplement_20261029')
class ProtocolError(RuntimeError):pass

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()

def validate_case_plans(p,root,stage,shard=None):
    splits=('train','dev','calibration','test');seen_ids=set();streams={};testbank_reports={}
    plans=p['case_plans']
    seed_source=Path(p['seed_replay_source']).resolve(strict=True);raw=seed_source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=p['source_sha256'].get(str(seed_source)):raise ProtocolError('seed replay source not bound')
    replay=types.ModuleType('registered_cpu_seed_replay');replay.__file__=str(seed_source);exec(compile(raw,str(seed_source),'exec'),replay.__dict__)
    if not callable(getattr(replay,'labeled_streams',None)):raise ProtocolError('labeled seed replay implementation missing')
    if not callable(getattr(replay,'verify_sources',None)):raise ProtocolError('source-bound RNG replay required')
    replay.verify_sources(p)
    if set(plans)!=set(splits):raise ProtocolError('four data splits required')
    for split in splits:
        item=plans[split];path=Path(item['path']).resolve(strict=True)
        if p['input_sha256'].get(str(path))!=item['sha256'] or sha(path)!=item['sha256']:raise ProtocolError('case plan source changed')
        data=json.loads(path.read_text(encoding='utf8'))
        if data.get('split')!=split or not isinstance(data.get('cases'),list) or not data['cases']:raise ProtocolError('invalid case plan')
        scoped_ids=None
        if split=='test' and stage in ('evaluation','rf') and shard is not None:
            spec=p['stages'][stage];mapping=spec.get('case_ids_by_shard');shards=spec.get('shards')
            if not isinstance(mapping,dict) or not isinstance(shards,list) or any(type(x) is not int or x<1 for x in shards) or len(shards)!=len(set(shards)) or set(mapping)!={str(x) for x in shards} or str(shard) not in mapping:raise ProtocolError('case assignment shards')
            all_ids=[x for group in mapping.values() for x in group];case_ids=[c.get('case_id') for c in data['cases']]
            if len(all_ids)!=len(set(all_ids)) or set(all_ids)!=set(case_ids) or len(case_ids)!=len(set(case_ids)) or not mapping[str(shard)]:raise ProtocolError('case assignment incomplete or duplicated')
            scoped_ids=set(mapping[str(shard)])
        seeds=set();owners={};aliases=p.get('rng_aliases',{}).get(split,{})
        if not isinstance(aliases,dict):raise ProtocolError('invalid split alias registry')
        for c in data['cases']:
            cid=c['case_id']
            if not isinstance(cid,str) or not cid or cid in seen_ids:raise ProtocolError('duplicate case ID')
            seen_ids.add(cid)
            required=('mechanism_seed','observation_seed','fold_seed','view_seed')
            values=[c[k] for k in required]
            actual=c.get('reserved_rng_seeds')
            if any(type(x) is not int or x<0 for x in values) or not isinstance(actual,list) or not actual or len(actual)!=len(set(actual)) or any(type(x) is not int or x<0 for x in actual) or not set(values).issubset(actual):raise ProtocolError('incomplete conservative case seed reservation')
            labeled=replay.labeled_streams(c)
            if not isinstance(labeled,dict) or not labeled or any(not isinstance(k,str) or not k or type(v) is not int or v<0 for k,v in labeled.items()) or len(labeled)!=len(set(labeled.values())) or set(labeled.values())!=set(actual):raise ProtocolError('case RNG replay mismatch')
            derived=list(labeled.values());seeds.update(derived)
            claims=c.get('rng_aliases',{})
            if not isinstance(claims,dict):raise ProtocolError('invalid case RNG aliases')
            for role,seed in labeled.items():
                alias=claims.get(str(seed))
                if alias is not None and (not isinstance(alias,str) or not alias):raise ProtocolError('invalid RNG alias')
                owners.setdefault(seed,[]).append((cid,role,alias))
            if set(claims)-{str(v) for v in derived}:raise ProtocolError('alias references absent RNG stream')
            path_value=c.get('observed_path')
            if split=='test' and stage in ('evaluation','rf','timing','summary','audit') and (scoped_ids is None or cid in scoped_ids):
                report_path=str(Path(c['testbank_report_path']).resolve(strict=True))
                if report_path not in testbank_reports:
                    if p['input_sha256'].get(report_path)!=sha(report_path):raise ProtocolError('testbank report not bound')
                    report=json.loads(Path(report_path).read_text(encoding='utf8'))
                    tb=p['stages']['testbank'];task=report.get('task')
                    if (report.get('status')!='TESTBANK_SHARD_WRITTEN_NOT_AUDITED' or report.get('job_id')!=tb['job_id'] or task not in tb['shards'] or
                        Path(report_path)!=root/'outputs'/'testbank'/('shard_%05d'%task)/'testbank.json'):raise ProtocolError('bad testbank report identity')
                    testbank_reports[report_path]=report
                report=testbank_reports[report_path];matches=[x for x in report['cases'] if x.get('case_id')==cid]
                if len(matches)!=1 or matches[0].get('status')!=c.get('testbank_status'):raise ProtocolError('testbank case status mismatch')
                if c['testbank_status']=='failed':
                    if path_value or c.get('observed_file_sha256') or c.get('fold_identity'):raise ProtocolError('failed testbank case has table')
                elif c['testbank_status']=='ok':
                    if not path_value or any(matches[0].get(k)!=c.get(k) for k in ('observed_path','observed_file_sha256','fold_identity')):raise ProtocolError('testbank case identity mismatch')
                else:raise ProtocolError('unknown testbank case status')
            if path_value:
                table=Path(path_value).resolve(strict=False)
                if split=='test' and not table.is_relative_to(root/'tables'):raise ProtocolError('test table outside new isolated test bank')
                if split!='test' or (stage in ('evaluation','rf','timing','summary','audit') and (scoped_ids is None or cid in scoped_ids)):
                    table=table.resolve(strict=True)
                    if p['input_sha256'].get(str(table))!=c.get('observed_file_sha256') or sha(table)!=c['observed_file_sha256']:raise ProtocolError('observed table not source-bound')
        used=set()
        for seed,entries in owners.items():
            if len(entries)==1:
                if entries[0][2] is not None:raise ProtocolError('orphan RNG alias')
                continue
            names={entry[2] for entry in entries}
            if len(names)!=1 or None in names:raise ProtocolError('undeclared within-split RNG reuse')
            alias=next(iter(names));rule=aliases.get(alias)
            expected_owners=sorted((item['case_id'],item['role']) for item in rule.get('owners',[])) if isinstance(rule,dict) else []
            if not isinstance(rule,dict) or rule.get('seed')!=seed or expected_owners!=sorted((cid,role) for cid,role,_ in entries):raise ProtocolError('RNG alias differs from preregistered owners')
            used.add(alias)
        if set(aliases)!=used:raise ProtocolError('unused or incomplete RNG alias registry')
        streams[split]=seeds
    for i,left in enumerate(splits):
        for right in splits[i+1:]:
            if streams[left]&streams[right]:raise ProtocolError('cross-split RNG collision')
    return True

def validate_shared_io_paths(p,root,stage):
    """Reject per-shard protocols that rehash the whole test bank as shared input.

    All assigned tables/reports are still checked by validate_case_plans; a
    no-shard check is required once at prequeue and again in the final audit.
    """
    if stage not in ('evaluation','rf'):return True
    spec=p['stages'][stage];paths=list(spec.get('input_paths',[]));pred=spec.get('predecessors',[])
    if not isinstance(pred,list):raise ProtocolError('invalid predecessor list')
    for item in pred:paths.extend(item.get('outputs',{}))
    bank=root/'tables';testbank=root/'outputs'/'testbank';testbank_paths=[]
    for item in paths:
        path=Path(item).resolve(strict=False)
        if path.is_relative_to(bank):raise ProtocolError('table belongs in shard-specific validation')
        if path.is_relative_to(root/'outputs'/'calibration'):raise ProtocolError('calibration shards belong in independent audit, not per-test worker')
        if path.is_relative_to(testbank):testbank_paths.append(path)
    if len(set(testbank_paths))>2:raise ProtocolError('too many shared testbank files per worker')
    return True

def check_predecessors(spec):
    required=spec.get('predecessors')
    if not isinstance(required,list) or not required:raise ProtocolError('actual predecessor evidence required')
    for item in required:
        job=item['job'];name=item['jobname'];tasks=item['tasks']
        if type(job) is not int or job<=0 or not isinstance(name,str) or not name or not isinstance(tasks,list) or not tasks:raise ProtocolError('invalid predecessor identity')
        out=subprocess.run(['qacct','-j',str(job)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120).stdout
        rows={}
        for block in out.split('=============================================================='):
            fields={}
            for line in block.splitlines():
                parts=line.split()
                if len(parts)>=2 and parts[0] in ('jobnumber','jobname','owner','taskid','failed','exit_status'):
                    if parts[0] in fields:raise ProtocolError('ambiguous qacct field')
                    fields[parts[0]]=parts[1]
            if not fields:continue
            raw=fields.get('taskid');task=0 if raw=='undefined' else int(raw)
            if task not in tasks:continue
            if task in rows or fields!={'jobnumber':str(job),'jobname':name,'owner':'shibo','taskid':raw,'failed':'0','exit_status':'0'}:raise ProtocolError('predecessor failed or identity mismatch')
            rows[task]=fields
        if set(rows)!=set(tasks):raise ProtocolError('missing predecessor accounting')
        outputs=item.get('outputs')
        if not isinstance(outputs,dict) or not outputs or any(sha(path)!=digest for path,digest in outputs.items()):raise ProtocolError('predecessor output SHA mismatch')
    return True

def require_stage_predecessors(p,stage):
    requirements={'calibration_finalize':('calibration',),'calibration_audit':('calibration','calibration_finalize'),'testbank':('calibration_audit',),'testbank_audit':('testbank',),'evaluation':('calibration_audit','testbank','testbank_audit'),'rf':('calibration_audit','testbank','testbank_audit'),'timing':('calibration_audit','testbank','testbank_audit')}
    for name in requirements.get(stage,()):
        target=p['stages'][name];tasks=target['shards'] if target['kind']=='array' else [0]
        matches=[x for x in p['stages'][stage]['predecessors'] if x.get('job')==target['job_id'] and x.get('jobname')==target['jobname'] and set(x.get('tasks',[]))==set(tasks)]
        if len(matches)!=1:raise ProtocolError('required stage predecessor missing: '+name)
    return True

def run_stage(manifest_path,manifest_sha,stage,shard):
 manifest_path=Path(manifest_path).resolve(strict=True)
 if sha(manifest_path)!=manifest_sha:raise ProtocolError('manifest drift')
 p=json.loads(manifest_path.read_text(encoding='utf8'))
 if p.get('schema')!='cpu_mediation_stage_v1' or p.get('status')!='FROZEN_CPU_STAGE' or p.get('scope')!='4m_only':raise ProtocolError('CPU protocol not frozen')
 root=OUTPUT_ROOT.resolve(strict=True)
 if Path(p['output_root']).resolve(strict=True)!=root or not manifest_path.is_relative_to(root):raise ProtocolError('output/manifest outside isolated root')
 for kind in ('source_sha256','artifact_sha256'):
  mapping=p.get(kind)
  if not isinstance(mapping,dict) or not mapping:raise ProtocolError('missing '+kind)
  for path,digest in mapping.items():
   if sha(path)!=digest:raise ProtocolError('changed '+kind+' '+path)
 validate_case_plans(p,root,stage,shard)
 spec=p['stages'][stage]
 validate_shared_io_paths(p,root,stage)
 paths=spec.get('input_paths')
 if not isinstance(paths,list) or not paths:raise ProtocolError('stage input list missing')
 for path in paths:
  if path not in p['input_sha256'] or sha(path)!=p['input_sha256'][path]:raise ProtocolError('stage input changed')
 if type(shard) is not int or shard not in spec['shards']:raise ProtocolError('unregistered shard')
 # This stage is submitted after the protocol is frozen.  Its actual job ID
 # is captured in the exclusive post-submission receipt, avoiding a cycle.
 receipt_path=Path(p['stage_receipts'][stage]).resolve(strict=True)
 if not receipt_path.is_relative_to(root/'receipts'):raise ProtocolError('receipt outside isolated root')
 receipt=json.loads(receipt_path.read_text(encoding='utf8'))
 if (receipt.get('status')!='ACTUAL_QSUB_RECEIPT' or receipt.get('stage')!=stage or receipt.get('protocol_sha256')!=manifest_sha or
     receipt.get('jobname')!=spec.get('jobname') or receipt.get('kind')!=spec.get('kind') or receipt.get('shards')!=spec.get('shards')):raise ProtocolError('stage receipt differs from frozen specification')
 job=receipt.get('job_id');kind=spec.get('kind');actual_job=os.environ.get('JOB_ID');actual_task=os.environ.get('SGE_TASK_ID')
 if type(job) is not int or job<=0 or actual_job!=str(job):raise ProtocolError('scheduler job identity')
 if kind=='array':
  if actual_task!=str(shard):raise ProtocolError('scheduler task identity')
 elif kind=='singleton':
  if actual_task not in (None,'','undefined') or shard!=1:raise ProtocolError('singleton task identity')
 else:raise ProtocolError('invalid stage kind')
 require_stage_predecessors(p,stage)
 check_predecessors(spec)
 source=Path(spec['worker_source']).resolve(strict=True);raw=source.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=p['source_sha256'].get(str(source)):raise ProtocolError('worker source drift')
 if stage not in ('calibration','calibration_finalize','calibration_audit','testbank','testbank_audit','evaluation','rf','timing','summary','audit'):raise ProtocolError('unknown stage')
 output=root/'outputs'/stage/('shard_%05d'%shard)
 if not output.parent.resolve(strict=True).is_relative_to(root):raise ProtocolError('output outside isolated root')
 output.mkdir() # exclusive; preserves prior outputs on retry
 context={'manifest':p,'manifest_sha256':manifest_sha,'stage':stage,'shard':shard,'job_id':job,'output_dir':str(output),'actual_predecessor_accounting_checked':True}
 try:
  obj=types.ModuleType('isolated_cpu_science_worker');obj.__file__=str(source);exec(compile(raw,str(source),'exec'),obj.__dict__)
  if not callable(getattr(obj,'run',None)):raise ProtocolError('CPU scientific stage adapter unavailable')
  result=obj.run(context)
  if not isinstance(result,dict):raise ProtocolError('worker result must be a dict')
  record={'status':'STAGE_COMPLETED_NOT_INDEPENDENTLY_AUDITED','manifest_sha256':manifest_sha,'stage':stage,'shard':shard,'job_id':job,'scheduler_task_id':shard,'worker_source_sha256':sha(source),'result':result,'actual_predecessor_accounting_checked':True}
 except BaseException as exc:
  record={'status':'STAGE_FAILED','manifest_sha256':manifest_sha,'stage':stage,'shard':shard,'job_id':job,'scheduler_task_id':shard,'error_type':type(exc).__name__,'error':str(exc)}
  with (output/'stage_record.json').open('x',encoding='utf8') as f:json.dump(record,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
  raise
 with (output/'stage_record.json').open('x',encoding='utf8') as f:json.dump(record,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 return record

def production_entry(*args,**kwargs):raise ProtocolError('CPU stage adapters, frozen input protocol, independent review and manual predecessor checks are still required')
if __name__=='__main__':production_entry()
