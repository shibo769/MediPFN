"""Prepare a fixed 4M CPU testbank protocol for user-held prequeue only.

The resulting protocol is NOT authorization to release or generate data.
Independent review and the required history/parallel and guard checks remain
separate prerequisites.  No test observations or scheduler jobs are created.
"""
import hashlib,json,os,subprocess,types,xml.etree.ElementTree as ET
from pathlib import Path

B=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9')
R=B/'capacity_4m_v10_20261028/isolated_cpu_supplement_20261029'
E=R/'test_rf_scoped_evidence_v1'; C=R/'test_rf_candidate_metadata_v1'; X=R/'candidate_covariates_v1'; S=R/'sources'
BASE=R/'protocols/calibration_audit.json'
SOURCE={
 'workflow':('cpu_simple_workflow_v5.py','761a10265c675d6c3397f46af66d6eb6700dd8b1fddbd1b45acd421cbe06731a'),
 'launcher':('cpu_stage_launch_v5.py','43ca77ecfe33c6977eca1935e75bf407430e5d0eda35903e3609ebb68019dedc'),
 'wrapper':('cpu_stage_launch_v5.sh','d6db2e15e4aae2ee093e7f27b6011a283072723990553ae4231eefd3335543bb'),
 'imports':('cpu_science_imports_v3.py','7c7c1fee5912b4b569676bf167c660d3025441295f1cc0e64f26b1d879843c96'),
 'loader':('case_loader_id_v3.py','36d8c97759e3d688cdd19eca5dc088910189afd8fbbfaa29a36bfd08e61980e4'),
 'rf_core':('rf_supplement_core.py','3647cf3436cd87ca7834b7afdb34c9dec4a6890457a8a8a77a50a88a4b479f2d'),
 'rf_binding':('rf_seed_binding_v1.py','c3f25964f926d83e80a83c405c4470f26efa4d165120b83758cc5f2d4dfdd188'),
 'testbank_worker':('cpu_testbank_stage_v2.py','9bd951a93085e13184201d41db5b062c25cc2c92489100aeacfe0aaf50d0ba3c'),
 'validator':('cpu_test_rf_candidate_validation_v2.py','fa877aac1dff5fc9f78390c2036d1a9069d84a7dcfd656465132b0682ddd2754'),
 'assembler':('cpu_testbank_protocol_assembly_v1.py','dac61367c4d371b07373190773f0fc5a6ccb13d81ed74b610b4d28130783db64'),
 'evaluation':('cpu_evaluation_stage_v2.py','dadeccb1266470285377802f4d4ad14c5da55e9bafbc9286af9f2cb824f7c6ce'),
 'rf_execution':('rf_method_execution_v2.py','54c41170cc53e79a4049627a495b27d52f88be49ddf12ca46b99b2c7356c221f'),
 'alpha_inference':('alpha_inference_cpu_v2.py','4c8ac70446ff3558c941f2ad2fa48ae4c6974395c490dd383ed794d26a17234c'),
 'rf_stage':('cpu_rf_stage_v1.py','1117d2dfa585ed4f1315f8feac3320de26249cff590d46e4acb367bd366163e0'),
 'ablation':('ablation95_v2.py','f40dc38082e8aca82ad2667fabcf9d7d0b3e83952e3078885fdb78ce336cc823'),
 'summary':('rf_supplement_summary.py','f5bd77db42b8124a15f81216773d180b41d9c71d702ded81b807700812664354'),
 'metrics':('wave_metrics_independent_v1.py','5037f26b882df3f6c337bba3fe7480379a0e024846596add24501f87712d3a8c'),
 'rf_cold_timing':('cpu_rf_end_to_end_timing_v2.py','f8e9e234264d6288e1a2b06d64c4b31bd477f2317af476b08ce2c1640f3f2ee4'),
 'cold_timing':('cpu_end_to_end_timing_v3.py','cc66b18ebf385042032106a88140bc45e2d45a65d5aef9873bf085724262809f'),
 'warm_timing_candidate':('shared_timing_cpu.py','fa977d6da85080311ad22298a8bbfc81f8df71d04dff6737690f36617b92d2db'),
 'prequeue':('cpu_testbank_prequeue_v1.py','6dbe3afae77eb05cda2581a1ecc5e21d0beda44ac94f9cd81f15f466fa00aecc'),
}
EVIDENCE={
 'cpu_test_rf_scoped_seed_source_evidence_v2.json':'0e480c14beb963bb2650eebaa4a934cdfe326307fd5e3f4ef4f0eb683c5f6a9c',
 'cpu_test_rf_scoped_seed_source_local_review_v1.json':'72cb70e350b269e96452d9cf1a1fd7f80dbc1ee89a9e6e16908c6f8588707321',
 'test_rf_829_bounded_recheck_v1.json':'e016e6e6052c64a0ad844cb476909ae964b6f17939f3c4b978759aadc0d5b977',
 'openml13_x_source_terms_review_20261029.md':'7e4335f995aebe990fc8ed1e70d48584ebaea9b30e48139e1e2028d9743658d2',
 'openml_cc18_source_citation_inventory_20261028.json':'60aab671bedd4034751e01c45a83b3c343edd0341f71fdfc0ed1b561db7d6b40',
}
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,expected):
 raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=expected:raise RuntimeError('source mismatch '+str(path))
 mod=types.ModuleType(path.stem);mod.__file__=str(path);exec(compile(raw,str(path),'exec'),mod.__dict__);return mod
def run(argv):
 p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120)
 if p.returncode:raise RuntimeError('command failed '+str(argv)+': '+p.stderr)
 return p.stdout
def named_history_precheck():
    """Recheck only the already reviewed named qacct and retained results."""
    specs=((8325518,'nie4m_candrng2',(0,)),(8325516,'nie4m_oldroots',(1,2,3)),
           (8326227,'nie4m_cfdag',(0,)),(8326806,'nie4m_tdv7',(1,2,3,4,5,6,7)),
           (8326927,'nie4m_xor',(0,)),(8328540,'nie4m_calhist',(0,)),
           (8331935,'nie4m_oldactive',(0,)),(8332059,'nie4m_oldreview',(0,)))
    for job,name,tasks in specs:
        out=run(['qacct','-j',str(job)]);seen=set()
        for block in out.split('=============================================================='):
            fields={}
            for line in block.splitlines():
                a=line.split()
                if len(a)>=2 and a[0] in ('jobnumber','jobname','owner','taskid','failed','exit_status'):
                    if a[0] in fields:raise RuntimeError('duplicate historical qacct field')
                    fields[a[0]]=a[1]
            if not fields:continue
            raw=fields.get('taskid');task=0 if raw=='undefined' else int(raw)
            if task in tasks:
                if task in seen or fields!={'jobnumber':str(job),'jobname':name,'owner':'shibo','taskid':raw,'failed':'0','exit_status':'0'}:raise RuntimeError('history qacct mismatch '+str(job))
                seen.add(task)
        if seen!=set(tasks):raise RuntimeError('history qacct incomplete '+str(job))
    A=B/'audit';files={
      'independent_seed_replay_20261029_0314/work/realx_stress_4m_wave_20261028/candidate_combined_internal_crosscheck_20261029.json':'9d4024d8b19612943b9e2b90a4feb915f9c4f0eb8da83b722ac3a1e4d7b95844',
      'independent_traindev_cfdag_20261029_0623/train_dev_cf_dag_result.json':'71b2259eb8d8636c4fb5e3aa627adc6a4c8007270cbe9109832dd93e2c2b294f',
      'independent_traindev_hintxor_20261029_1124/result.json':'a9c1580e62f4e2fd52480803cc502a4cd8e4762f44fabdf79e0f608c017cc844',
      'independent_cal400_history_accounting_20261029/accounting_precheck_result.json':'282656e1a640cb61466e3f8b277fdca8d9d175b665ab2b23a953ff3458b044fd',
      'independent_cal400_active8299223_review_v2_20261029/retained8299223_review.json':'2c03972c9c1198f3bb560aebc48acaec9849eda74169528deca4ce59adfa4354',
      'independent_cal400_active8299223_v2_20261029/retained8299223_seed_integers.json':'99c208f9454aed102d0e3f5938649cb1a420ae5070002972bc6a933e016af50e'}
    roots=('14f653e2052c7055d546011342511ec1c3e22f55e81ebb87e159d15d35375fbb','894b83ca2409ac2396ea597d12a9ab43e6e1e13fc12abe67e75a0d39ced3b7a0','5e3d4ce2b58e83afdb8dffe7beeeebe1f24a3505be678206cc9868831dd56508')
    for i,h in enumerate(roots,1):files['independent_old_bootstrap_roots_20261029_0322/root_result_%02d.json'%i]=h
    seven=('f201cc0380e406ac7dd43fac6cc634844725cbaccf39df3311de5a1ecff37585','ad1244dfb151f5a59e3e7e3a366239e00cdd8644768f929eee75cd17d8a21539','a9660ced552afdcc71928409a74a8804ff1a687522ae141e27a3336271fc261b','4664f76abbbd1487288ae7f63789ad2887ad2d84b44b78fa95e78d32be4254c3','93cda67934f567c51c623400d78cce68fc2ef1173cb0fcc39fad963d88e87d51','a591416db568ddaa6a9c2ec0f930132f7b6d6780e99ac597479dfdcd7199b9d1','c20fb4a798df7a46808787fa38612239f17ba05b251789eb99c1a2b8b1cb966e')
    for i,h in enumerate(seven,1):files['independent_traindev_seven_transforms_20261029_1026/result_%02d.json'%i]=h
    if any(sha(A/path)!=h for path,h in files.items()):raise RuntimeError('retained historical result drift')
    return {str(A/path):h for path,h in files.items()}

def live_writer_check():
 root=ET.fromstring(run(['qstat','-xml','-u','shibo']));jobs={}
 for row in root.iter('job_list'):
  jid=row.findtext('JB_job_number');state=row.findtext('state') or '';name=row.findtext('JB_name') or ''
  if not state.startswith('h'):jobs.setdefault(jid,{'name':name,'states':set()})['states'].add(state)
 if set(jobs)!={'8299223'} or jobs['8299223']['name']!='nie10bootspeed' or not jobs['8299223']['states'].issubset({'qw','r'}):raise RuntimeError('active-writer review required: '+str(jobs))
 fields={}
 for line in run(['qstat','-j','8299223']).splitlines():
  k,sep,v=line.partition(':')
  if sep and k in ('job_number','owner','job_name','script_file','cwd'):
   if k in fields:raise RuntimeError('duplicate scheduler identity')
   fields[k]=v.strip()
 script=B/'posttrain_v10_20261026/bootstrap_speed_20261027/run.sh'
 if (fields.get('job_number')!='8299223' or fields.get('owner')!='shibo' or fields.get('job_name')!='nie10bootspeed' or fields.get('cwd')!='/usr3/graduate/shibo' or fields.get('script_file')!=str(script)):raise RuntimeError('retained actor identity changed')
 pinned={script:'f1ba0e1ac506404320d7f2461b1f745ae03b5d029953f02b06d545cdc0643087',B/'posttrain_v10_20261026/bootstrap_speed_20261027/run.py':'955fdb4c236a429a31a4b166e0e517e80c0209059c3706aeeb6409752ff5941e',B/'posttrain_v10_20261026/protocol/time_full.py':'17c9c905766311f501372ffe7ae53545f3c72f9801aed128edfa9ac5788ad560',B/'code/baselines.py':'e70469e83a0659946bcff6ecc4adea4472a78d647b1c7dee2c7bbc8c1321a895'}
 if any(sha(path)!=h for path,h in pinned.items()):raise RuntimeError('retained actor source changed')
 return {'job':8299223,'name':fields['job_name'],'states':sorted(jobs['8299223']['states']),'current_source_sha256':{str(k):v for k,v in pinned.items()},'limitations':['point-in-time scheduler observation','old actor spool not recovered; not a continuing lock']}

def freeze_for_held_prequeue():
 target=R/'protocols/testbank.json';digest=R/'protocols/testbank.sha256';out=R/'test_rf_prequeue_metadata_v1';receipt=R/'receipts/testbank.json';intent=R/'receipts/testbank_intent.json'
 if any(p.exists() for p in (target,digest,out,receipt,intent)):raise RuntimeError('earlier attempt exists; reconcile')
 for name,h in EVIDENCE.items():
  if sha(E/name)!=h:raise RuntimeError('scoped evidence drift: '+name)
 method_policy=S/'cpu_test_scientific_commitment_candidate_v6.json'
 if sha(method_policy)!='d2da8adf2c5a93d3528a657db4e3d7918399e8a4d4f37a2a6c86b69a4e41fb85':raise RuntimeError('method policy drift')
 policy=json.loads(method_policy.read_text())
 if policy.get('status')!='DRAFT_NOT_AUTHORIZATION' or policy.get('primary_95')!='selected_calibrated' or policy['frozen_95']['selected']!='quantile_group':raise RuntimeError('method policy identity')
 old=B/'capacity_4m_v10_20261028/robust_partial_composition_retry_20261028/trained_v1';chosen_path=old/'model.pt';selection=old/'selection.pt'
 if sha(chosen_path)!='6cbd813b84513f43d003f970a50cf202ca5c4d46b408be7d886def72189b69ed' or sha(selection)!='d7e35ca89fe6829a5630307e713d6b443f8c976c8799c5627cf357574fa57612':raise RuntimeError('95% chosen/selection drift')
 evidence=json.loads((E/'cpu_test_rf_scoped_seed_source_evidence_v2.json').read_text());review=json.loads((E/'cpu_test_rf_scoped_seed_source_local_review_v1.json').read_text());retained=json.loads((E/'test_rf_829_bounded_recheck_v1.json').read_text())
 if (evidence.get('status')!='DRAFT_BOUNDED_EVIDENCE_NOT_AUTHORIZATION' or review.get('status')!='SCOPED_LOCAL_REVIEW_NOT_AUTHORIZATION' or review['evidence']['sha256']!=EVIDENCE['cpu_test_rf_scoped_seed_source_evidence_v2.json'] or retained['candidate_hashes']['rf_mapping']!='aa59dead72e7765f2a033a036a0276068e2473638e5dadfe9fac6a75129f86f4' or any(retained['intersection_counts'].values())):raise RuntimeError('scoped review identity')
 modules={k:load(S/name,h) for k,(name,h) in SOURCE.items() if k in ('validator','assembler','rf_core','rf_binding','workflow')}
 for name,h in SOURCE.values():
  if sha(S/name)!=h:raise RuntimeError('candidate source drift '+name)
 base_raw=BASE.read_bytes();base=json.loads(base_raw)
 validator=modules['validator'];audit=validator.validate_accounted_calibration(R)
 test_path=Path(base['case_plans']['test']['path']);test_raw=test_path.read_bytes();test=json.loads(test_raw)
 seeds_path=C/'rf_seed_cases_candidate_v1.json';mapping_path=C/'rf_mapping_candidate_v1.json'
 if sha(seeds_path)!='773d4155c63d07353184051dd6c801c346e9e711919db361c80733e493ec22db' or sha(mapping_path)!='aa59dead72e7765f2a033a036a0276068e2473638e5dadfe9fac6a75129f86f4':raise RuntimeError('candidate inputs changed')
 seeds=json.loads(seeds_path.read_text());mapping=json.loads(mapping_path.read_text());chosen=modules['rf_core'].select_subset(seeds['cases'],54,45)
 inventory=validator.validate_candidate(test,seeds,mapping,chosen,modules['rf_binding'])
 for path,h in ((X/'provenance.json','8a1bcab4191771b620b7127ac7b96aeb69e3a43dab456f02978364b94df80631'),(X/'x_populations.npz','191d0cd36a707c32dd3f09dfb6ce0d3d8a97a65d8fa7ad832341a22e3eef5b81')):
  if sha(path)!=h:raise RuntimeError('X source changed')
 for group in ('source_sha256','artifact_sha256'):
  for path,h in base[group].items():
   if sha(path)!=h:raise RuntimeError('baseline source/artifact changed '+path)
 history=named_history_precheck();writer=live_writer_check()
 frozen=dict(mapping);frozen['status']='FROZEN_BEFORE_TEST';frozen['tasks_sha256']=inventory['case_digest'];frozen['scope_review_sha256']=EVIDENCE['cpu_test_rf_scoped_seed_source_local_review_v1.json']
 frozen_bytes=(json.dumps(frozen,sort_keys=True,indent=2)+'\n').encode();frozen_sha=hashlib.sha256(frozen_bytes).hexdigest();frozen_path=out/'rf_mapping.json'
 paths={k:str(S/v[0]) for k,v in SOURCE.items() if k in ('workflow','launcher','wrapper','imports','loader','rf_core','rf_binding','testbank_worker')};hashes={k:SOURCE[k][1] for k in paths}
 p=modules['assembler'].assemble(base,base_raw,test_raw,source_paths=paths,source_hashes=hashes,seed_cases_path=str(seeds_path),seed_cases_sha=sha(seeds_path),mapping_path=str(frozen_path),mapping_sha=frozen_sha,x_provenance_path=str(X/'provenance.json'),x_provenance_sha=sha(X/'provenance.json'),x_npz_path=str(X/'x_populations.npz'),x_npz_sha=sha(X/'x_populations.npz'),evidence_paths={str(E/name):h for name,h in EVIDENCE.items()},bootstrap_indices=chosen)
 for key in ('validator','assembler','prequeue','evaluation','rf_execution','alpha_inference','rf_stage','ablation','summary','metrics','cold_timing','rf_cold_timing','warm_timing_candidate'):p['source_sha256'][str(S/SOURCE[key][0])]=SOURCE[key][1]
 p['input_sha256'][str(method_policy)]=sha(method_policy);p['stages']['testbank']['input_paths'].append(str(method_policy))
 p['artifact_sha256'][str(chosen_path)]=sha(chosen_path);p['artifact_sha256'][str(selection)]=sha(selection);p['model_assets'].update({'chosen':str(chosen_path),'selection':str(selection)})
 p['methods_95']=policy['methods_95'];p['methods_levels']=policy['level_methods'];p['allow_methods']=p['methods_95']+p['methods_levels'];p['rf_settings']=policy['rf']['settings']
 training=B/'capacity_4m_v10_20261028/robust_retrain_train_seedfix_v2.py'
 if sha(training)!='91d2cbb5a6f9f09a37b5d9b23a78c806c30030bc67e0382f579f2c26e0d759d4':raise RuntimeError('95% head source drift')
 p['source_sha256'][str(training)]=sha(training)
 p['project_import_paths'].update({'robust_retrain_train_seedfix_v2':str(training),'ablation95':str(S/SOURCE['ablation'][0]),'alpha_inference':str(S/SOURCE['alpha_inference'][0]),'rf_method_execution_v2':str(S/SOURCE['rf_execution'][0]),'rf_seed_binding_v1':str(S/SOURCE['rf_binding'][0])})
 p['source_sha256'][str(Path(__file__).resolve(strict=True))]=sha(Path(__file__))
 p['input_sha256'].update(audit);p['input_sha256'].update(history);p['stages']['testbank']['input_paths']+=sorted(history);p['status']='FROZEN_CPU_STAGE';p['authorization']=False
 p['scientific_scope']['testbank_prequeue']={'scope_review_sha256':EVIDENCE['cpu_test_rf_scoped_seed_source_local_review_v1.json'],'candidate_inventory':inventory,'writer_observation':writer,'limitations':['Prequeue only; independent review and both required scientific guards precede release.','Historical e65 and retained actor spool not recovered; no all-history claim.','Current writer observation is not continuous isolation.','X-only private research; five sources have uncertain upstream redistribution terms.','CPU/GPU interval equivalence is not established.']}
 wf=modules['workflow'];wf.validate_case_plans(p,R,'testbank');wf.require_stage_predecessors(p,'testbank');wf.check_predecessors(p['stages']['testbank'])
 if live_writer_check()!=writer:raise RuntimeError('active writer changed during prequeue freeze')
 out.mkdir();(R/'outputs/testbank').mkdir(exist_ok=False)
 with frozen_path.open('xb') as f:f.write(frozen_bytes);f.flush();os.fsync(f.fileno())
 with target.open('x') as f:json.dump(p,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 h=sha(target)
 with digest.open('x') as f:f.write(h+'\n');f.flush();os.fsync(f.fileno())
 return {'status':'PREQUEUE_PROTOCOL_FROZEN_NOT_AUTHORIZATION','protocol':str(target),'protocol_sha256':h,'mapping_sha256':frozen_sha,'cases':2952,'bootstrap_references':261,'writer_observation':writer}

if __name__=='__main__':
 raise RuntimeError('requires independently reviewed held-prequeue invocation; no automatic release')
