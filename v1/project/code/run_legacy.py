"""Matched p3 legacy baseline CLI, separately source/asset frozen and CPU-only."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,uuid
import numpy as np
from evaluate import EvaluationGuard,sha,clean,canonical_score,interval_stats,failed_stats,metrics

ASSETS=('code/model.py','code/nuisance_model.py','base.pt','v1_head.pt')


def verify_registration(protocol,registration,folder):
    value=json.loads(Path(registration).read_text(encoding='utf-8'))
    if value.get('frozen') is not True or value['protocol_sha256']!=sha(protocol):raise ValueError('Separate frozen legacy registration required')
    for name in ('legacy_worker.py','run_legacy.py'):
        if value['source_hashes'][name]!=sha(Path(__file__).with_name(name)):raise ValueError('Legacy source changed')
    for relative in ASSETS:
        if value['assets'][relative]!=sha(Path(folder)/relative):raise ValueError('Frozen legacy asset changed '+relative)
    return value


def run_cell(group,protocol,registration,legacy_dir,output,authorization=None,threads=None):
    if group['p']!=3 or group.get('bank') not in ('synthetic','semisynthetic'):raise ValueError('Only prespecified p3 main/semisynthetic subset')
    EvaluationGuard(protocol,authorization or Path(protocol).resolve().parent/'audit/test_authorization.json').check('test' if group['split']=='test' else group['split'])
    binding=verify_registration(protocol,registration,legacy_dir)
    if group['protocol_sha256']!=sha(protocol) or group['raw_sha256']!=sha(group['raw_path']) or group['metadata_sha256']!=sha(group['metadata_path']):raise ValueError('Legacy raw binding changed')
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True);path=folder/(group['name']+'.legacy.npz');ledger=folder/(group['name']+'.legacy.metadata.json')
    if path.exists() and ledger.exists():
        old=json.loads(ledger.read_text())
        if old['predictions_sha256']!=sha(path) or old['registration_sha256']!=sha(registration) or old['raw_sha256']!=group['raw_sha256']:raise ValueError('Existing legacy artifact drift')
        return dict(group=group['name'],resumed=True,path=str(path))
    if path.exists() or ledger.exists():raise FileExistsError('Incomplete legacy output preserved')
    temporary=folder/(group['name']+'.worker-'+uuid.uuid4().hex+'.npz');stderr=folder/(group['name']+'.worker.stderr.log');start=time.perf_counter()
    command=[sys.executable,'-B',str(Path(__file__).with_name('legacy_worker.py')),'--legacy-dir',str(Path(legacy_dir).resolve())]
    if threads is not None:command+=['--threads',str(threads)]
    with stderr.open('xb') as errorlog:
        worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errorlog,text=True)
        try:
            ready=json.loads(worker.stdout.readline())
            if ready.get('ready') is not True:raise RuntimeError('Legacy worker did not initialize')
            assert ready['base_sha256']==binding['assets']['base.pt'] and ready['head_sha256']==binding['assets']['v1_head.pt']
            worker.stdin.write(json.dumps(dict(raw_path=group['raw_path'],output_path=str(temporary)))+'\n');worker.stdin.flush();finished=json.loads(worker.stdout.readline())
            if finished.get('complete') is not True or finished['sha256']!=sha(temporary):raise RuntimeError('Legacy worker incomplete')
            worker.stdin.write(json.dumps(dict(command='stop'))+'\n');worker.stdin.flush();worker.wait(timeout=60)
            if worker.returncode:raise RuntimeError('Legacy worker exit failure')
        finally:
            if worker.poll() is None:worker.terminate();worker.wait(timeout=30)
    with np.load(temporary,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
    with np.load(group['raw_path'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    rawmeta=json.loads(Path(group['metadata_path']).read_text());count=len(raw['truth']);stats={m:{a:[] for a in (.1,.05,.01)} for m in ('old_v1','native_fsp')};tables=[]
    for i in range(count):
        table={k:raw[k][i] for k in ('x','a','m','y')};summary=canonical_score(table,pred['nuisance'][i]);old_error=str(pred['old_errors'][i]) or summary['error'];native_error=str(pred['native_errors'][i])
        with np.errstate(over='ignore',invalid='ignore'):native_se=float(np.exp(pred['native_logV'][i]/2)/np.sqrt(group['n']))
        native=dict(estimate=float(pred['native_estimate'][i]),se=native_se,clipping_rate=0.,failed=bool(native_error) or not(np.isfinite([pred['native_estimate'][i],native_se]).all() and native_se>0))
        for alpha in (.1,.05,.01):stats['old_v1'][alpha].append(interval_stats(summary,alpha));stats['native_fsp'][alpha].append(interval_stats(native,alpha))
        tables.append(dict(rawmeta['tables'][i],methods=dict(old_v1=dict(error=old_error,interval_evidence=dict(kind='wald')),native_fsp=dict(error=native_error,score_kind='direct_native',interval_evidence=dict(kind='wald'),variance_definition='exp(logV) coefficient; SE=sqrt(V/n), no clipping/inflation'))))
    arrays=dict(old_v1__nuisance=pred['nuisance'],native_fsp__native_estimate=pred['native_estimate'],native_fsp__native_logV=pred['native_logV'],**{k:raw[k] for k in ('truth','fold_ids','mechanism_id')})
    for method in stats:
        for alpha in (.1,.05,.01):arrays[method+'__stats__'+str(alpha)]=np.asarray(stats[method][alpha])
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    report=dict(raw_sha256=group['raw_sha256'],predictions_sha256=sha(path),registration_sha256=sha(registration),protocol_sha256=sha(protocol),methods=['old_v1','native_fsp'],tables=tables,
        assets=binding['assets'],source_hashes=binding['source_hashes'],device='cpu',seconds_including_worker_load=time.perf_counter()-start,
        old_v1='Exact old frozen encoder+v1 nuisance decoder, same prescribed folds, raw64full-ncanonicalscore',native_fsp='Original full-table direct effect/logV heads, independent95Waldformulaextended90/99, populationVcoefficient/n',
        no_old_assets_modified=True,no_truth_input_to_worker=True,failures_preserved=True,
        metrics=[dict(method=m,**metrics(arrays[m+'__stats__'+str(a)],raw['truth'],a)) for m in stats for a in (.1,.05,.01)])
    with ledger.open('x',encoding='utf-8') as f:json.dump(clean(report),f,indent=2,allow_nan=False)
    temporary.unlink()
    return dict(group=group['name'],path=str(path),metadata_path=str(ledger),methods=report['methods'],attempts=count)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--raw-manifest',required=True);p.add_argument('--registration',required=True);p.add_argument('--legacy-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--authorization');p.add_argument('--threads',type=int)
    p.add_argument('--task-index',type=int,default=int(os.environ.get('SGE_TASK_ID','1'))-1);p.add_argument('--task-count',type=int,default=1);args=p.parse_args()
    if not 0<=args.task_index<args.task_count:raise ValueError('Invalid zero-based array shard')
    manifest=json.loads(Path(args.raw_manifest).read_text());groups=[g for g in manifest['groups'] if g['p']==3 and g.get('bank') in ('synthetic','semisynthetic')]
    for group in groups[args.task_index::args.task_count]:print(json.dumps(run_cell(group,args.protocol,args.registration,args.legacy_dir,args.output,args.authorization,args.threads)),flush=True)


if __name__=='__main__':main()
