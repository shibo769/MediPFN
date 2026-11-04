"""Validate named, bounded calibration seed evidence; no general history claim."""
from datetime import datetime
from pathlib import Path
import hashlib,json,re
PROPOSAL='1c9fb427a0f536a287e56e41b3e8e7becc9a14897a2ed65fdf2cafa12d6cd402'
STREAM='84485ca0cc2439b98e4af4c1599b34bfded18f4f6c6a8c4dae7867225cb7d53c'
SCOPE='enumerated_selected_ancestors_and_named_parallel_plans'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def validate(path,expected_sha):
 if sha(path)!=expected_sha:raise RuntimeError('calibration audit changed')
 a=json.loads(Path(path).read_text(encoding='utf8'))
 if a.get('schema')!='mediation_4m_calibration_seed_scoped_audit_v2' or a.get('status')!='PASS' or a.get('fixture_only') is not False or a.get('authorization') is not False or a.get('scope')!=SCOPE:raise RuntimeError('scoped audit identity')
 if datetime.fromisoformat(a['generated_utc'].replace('Z','+00:00')).tzinfo is None:raise RuntimeError('timestamp')
 q=a['calibration']
 if (q.get('proposal_source_sha256')!=PROPOSAL or q.get('stream_digest')!=STREAM or q.get('mechanisms')!=400 or q.get('streams')!=4000 or q.get('group_counts')!={'id':100,'tree_rff':100,'tree_mlp':100,'rff_mlp':100}):raise RuntimeError('calibration plan')
 if a.get('checks')!={'within_calibration':'PASS','named_history_blocks':'PASS','named_bootstrap_plans':'PASS','retained_829_plan':'PASS','active_direct_census':'PASS'}:raise RuntimeError('named checks')
 if a.get('named_overlap_counts')!={'calibration_internal':0,'history_20_blocks_and_prior':0,'named_bootstrap_6_blocks':0,'retained_829_plan':0}:raise RuntimeError('named collisions')
 history=a['history'];active=a['active'];parallel=a['registered_parallel']
 if (history.get('scope')!=SCOPE or history.get('unretained_history_not_covered') is not True or history.get('named_blocks')!=20 or history.get('direct_4m_plans')!=5 or not history.get('limitations')):raise RuntimeError('history scope')
 if active.get('scope')!='point_in_time_direct_project_jobs' or active.get('owner')!='shibo' or active.get('unclassified_direct_jobs')!=0:raise RuntimeError('active scope')
 if parallel.get('scope')!='named_bootstrap_parallel_and_retained_job_plans' or parallel.get('bootstrap_blocks')!=6 or parallel.get('retained_job')!=8299223:raise RuntimeError('parallel scope')
 for item in (history['inventory'],active['scheduler_snapshot'],active['scoped_snapshot'],active['scoped_policy'],parallel['inventory']):
  if not re.fullmatch(r'[0-9a-f]{64}',item.get('sha256','')) or sha(item['path'])!=item['sha256']:raise RuntimeError('named inventory changed')
 limitations=history['limitations']
 if not isinstance(limitations,list) or not all(isinstance(x,str) and x.strip() for x in limitations) or not any('e65' in x for x in limitations) or not any('spool' in x.lower() for x in limitations) or not any('other' in x.lower() and ('histor' in x.lower() or 'plan' in x.lower()) for x in limitations):raise RuntimeError('missing residual limitations')
 sources=a.get('source_sha256')
 if not isinstance(sources,dict) or PROPOSAL not in sources.values() or not re.fullmatch(r'[0-9a-f]{64}',a.get('producer_source_sha256','')):raise RuntimeError('source map')
 for path,digest in sources.items():
  if not re.fullmatch(r'[0-9a-f]{64}',digest) or sha(path)!=digest:raise RuntimeError('source drift')
 return a
