"""CPU per-cell matched nuisance baselines, immutable array-job outputs."""
from pathlib import Path
import argparse,json,os,time,sys,importlib.metadata
import numpy as np
from baselines import crossfit_nuisance
from data import sample_mechanism,derived_seed
from evaluate import EvaluationGuard,inference_from_nuisance,metrics,clean,sha,canonical_score,interval_stats,failed_stats,bootstrap_score_interval,bootstrap_refit_interval


def run_cell(group,kind,protocol_path,output_root,authorization=None,estimator_jobs=1):
    split=group['split'];stage='dev' if split in ('train','dev','calibration','cal','capacity_train','capacity_dev') else 'test'
    EvaluationGuard(protocol_path,authorization or Path(protocol_path).resolve().parent/'audit/test_authorization.json').check(stage)
    if group['protocol_sha256']!=sha(protocol_path) or group['raw_sha256']!=sha(group['raw_path']) or group['metadata_sha256']!=sha(group['metadata_path']):raise ValueError('Raw manifest stale')
    root=Path(output_root);folder=root/kind;folder.mkdir(parents=True,exist_ok=True);name=group['name'];out=folder/(name+'.npz');metaout=folder/(name+'.metadata.json')
    if out.exists() and metaout.exists():
        previous=json.loads(metaout.read_text(encoding='utf-8'))
        if previous['predictions_sha256']!=sha(out) or previous['raw_sha256']!=group['raw_sha256'] or previous['protocol_sha256']!=sha(protocol_path):raise ValueError('Existing baseline binding changed')
        for file,digest in previous['source_hashes'].items():
            if sha(Path(__file__).with_name(file))!=digest:raise ValueError('Existing baseline source changed')
        return dict(group=name,kind=kind,prediction_path=str(out),metadata_path=str(metaout),attempts=group['count'],resumed=True)
    if out.exists() or metaout.exists():raise FileExistsError('Incomplete baseline output preserved: '+name)
    metadata=json.loads(Path(group['metadata_path']).read_text(encoding='utf-8'));populations={};start=time.perf_counter();nuisances=[];summaries=[];errors=[];seconds=[];tables=[]
    with np.load(group['raw_path'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    for i,record in enumerate(metadata['tables']):
        scm=None;oracle_delta=None
        if kind=='oracle':
            population=None
            if 'population_path' in record:
                path=record['population_path']
                if path not in populations:
                    if sha(path)!=record['population_sha256']:raise ValueError('Empirical population changed')
                    populations[path]=np.load(path,allow_pickle=False)
                population=populations[path]
            law=record['mechanism']
            scm=sample_mechanism(law['seed'],law['p'],law['family'],law['population_size'],population)
            if scm.population_hash!=law['population_hash']:raise ValueError('Oracle covariate population hash differs')
            oracle_delta=float(scm.theta_NIE-law['theta_NIE'])
            # Same finite functional; a fixed arithmetic tolerance accommodates
            # cross-node summation/BLAS roundoff, never Monte Carlo uncertainty.
            np.testing.assert_allclose(scm.theta_NIE,law['theta_NIE'],rtol=0,atol=1e-12,err_msg='Oracle finite-law truth regeneration')
        table={k:raw[k][i] for k in ('x','a','m','y')};t=time.perf_counter()
        result=crossfit_nuisance(table,raw['fold_ids'][i],kind=kind,scm=scm,seed=int(raw['observation_seed'][i])%(2**31-1),n_jobs=estimator_jobs)
        inference=inference_from_nuisance(table,result['nuisance']);seconds.append(time.perf_counter()-t);nuisances.append(result['nuisance']);summaries.append(inference)
        error=result['error'] or inference['summary']['error'];errors.append(error)
        tables.append(dict(mechanism=record['mechanism'],mechanism_id=record['mechanism_id'],n=record['n'],draw_seed=record['draw_seed'],fold_seed=record['fold_seed'],
            **{k:v for k,v in record.items() if k.startswith('population_')},methods={kind:dict(error=error,interval_evidence=dict(kind='wald'),fit_metadata=result['metadata'],oracle_theta_regeneration_delta=oracle_delta,oracle_truth_tolerance=1e-12 if kind=='oracle' else None)}))
    payload={kind+'__nuisance':np.asarray(nuisances),**{k:raw[k] for k in ('truth','fold_ids','mechanism_id')}}
    levels=(.1,.05,.01)
    for alpha in levels:payload[kind+'__stats__'+str(alpha)]=np.asarray([r['stats'][str(alpha)] for r in summaries])
    with out.open('xb') as f:np.savez_compressed(f,**payload)
    fixed=[]
    if group.get('bank')=='fixed_mechanism':
        for mid in np.unique(raw['mechanism_id']):
            selected=raw['mechanism_id']==mid
            for alpha in levels:fixed.append(dict(mechanism_id=str(mid),method=kind,**metrics(payload[kind+'__stats__'+str(alpha)][selected],raw['truth'][selected],alpha,fixed_mechanism=True)))
    report=dict(raw_sha256=group['raw_sha256'],predictions_sha256=sha(out),raw_metadata_sha256=group['metadata_sha256'],protocol_sha256=sha(protocol_path),
        source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('baselines.py'),Path(__file__).with_name('evaluate.py'))},
        methods=[kind],tables=tables,seconds=time.perf_counter()-start,per_table_fit_and_score_seconds=seconds,oracle=kind=='oracle',
        no_rows_or_failures_dropped=True,fixed_mechanism_diagnostics=fixed,
        runtime=dict(numpy=np.__version__,scikit_learn=importlib.metadata.version('scikit-learn'),python=sys.version.split()[0],estimator_jobs=estimator_jobs,allocated_SGE_slots=os.environ.get('NSLOTS')),
        metrics=[dict(method=kind,**metrics(payload[kind+'__stats__'+str(alpha)],raw['truth'],alpha,fixed_mechanism=group['mechanisms']==1)) for alpha in levels])
    with metaout.open('x',encoding='utf-8') as f:json.dump(clean(report),f,indent=2,allow_nan=False)
    return dict(group=group['name'],kind=kind,prediction_path=str(out),metadata_path=str(metaout),attempts=len(raw['truth']),failures=sum(e is not None for e in errors),seconds=report['seconds'])


def bootstrap_eligible(group):
    """Prespecified index subset, never selected by observed coverage/outcomes."""
    return group['split']=='test' and group['bank']=='synthetic' and group['population_kind']=='synthetic' and group['family'] in ('linear','weak_overlap','combinations') and group['p'] in (3,10,30) and group['n'] in (600,1000)


def run_bootstrap_cell(group,kind,protocol_path,output_root,authorization=None,estimator_jobs=1):
    if not bootstrap_eligible(group):raise ValueError('Cell is outside prespecified bootstrap subset')
    EvaluationGuard(protocol_path,authorization or Path(protocol_path).resolve().parent/'audit/test_authorization.json').check('test')
    protocol=json.loads(Path(protocol_path).read_text(encoding='utf-8'));setting=protocol['bootstrap'];repeats=int(setting['repeats']);count=min(group['count'],int(setting['tables_per_selected_cell']))
    base=run_cell(group,kind,protocol_path,output_root,authorization,estimator_jobs);folder=Path(output_root)/kind;name=group['name'];out=folder/(name+'.bootstrap.npz');metaout=folder/(name+'.bootstrap.metadata.json');subset=folder/(name+'.bootstrap.raw.npz')
    if out.exists() and metaout.exists():
        prior=json.loads(metaout.read_text(encoding='utf-8'))
        if prior['predictions_sha256']!=sha(out) or prior['raw_sha256']!=sha(subset) or prior['protocol_sha256']!=sha(protocol_path):raise ValueError('Existing bootstrap binding changed')
        for filename,digest in prior['source_hashes'].items():
            if sha(Path(__file__).with_name(filename))!=digest:raise ValueError('Bootstrap source changed')
        return dict(group=name,kind=kind,bootstrap=True,attempts=count,resumed=True)
    if out.exists() or metaout.exists() or subset.exists():raise FileExistsError('Incomplete bootstrap artifact preserved')
    with np.load(group['raw_path'],allow_pickle=False) as z:raw={k:z[k][:count] for k in z.files}
    with np.load(base['prediction_path'],allow_pickle=False) as z:nuisances=z[kind+'__nuisance'][:count]
    original=json.loads(Path(group['metadata_path']).read_text(encoding='utf-8'));metadata=original['tables'][:count];methods=(kind+'_score_bootstrap',kind+'_refit_bootstrap');stats={m:{a:[] for a in (.1,.05,.01)} for m in methods};tables=[];ticks=[];populations={}
    for i in range(count):
        table={k:raw[k][i] for k in ('x','a','m','y')};summary=canonical_score(table,nuisances[i]);record=metadata[i];scm=None
        if kind=='oracle':
            law=record['mechanism'];population=None
            if 'population_path' in record:
                path=record['population_path']
                if path not in populations:
                    if sha(path)!=record['population_sha256']:raise ValueError('Empirical X drift')
                    populations[path]=np.load(path,allow_pickle=False)
                population=populations[path]
            scm=sample_mechanism(law['seed'],law['p'],law['family'],law['population_size'],population)
        seed=derived_seed(int(raw['observation_seed'][i]),202610260401);start=time.perf_counter()
        def predictor(observed,fold_ids,seed):return crossfit_nuisance(observed,fold_ids,kind=kind,scm=scm,seed=seed,n_jobs=estimator_jobs)
        if summary['failed']:refit=dict(interval_evidence=dict(kind='refit_bootstrap',estimates=[None]*repeats,seed=seed),failures=repeats)
        else:refit=bootstrap_refit_interval(table,predictor,.05,repeats,seed)
        evidence=refit['interval_evidence'];values=evidence['estimates'];lower=np.sort([-np.inf if v is None else v for v in values]);upper=np.sort([np.inf if v is None else v for v in values]);errors={}
        for alpha in (.1,.05,.01):
            conditional=interval_stats(summary,alpha,endpoints=bootstrap_score_interval(summary,alpha,repeats,seed))
            lo=lower[max(0,int(np.floor(repeats*alpha/2))-1)];hi=upper[min(repeats-1,int(np.ceil(repeats*(1-alpha/2)))-1)]
            full=interval_stats(summary,alpha,endpoints=(lo,hi));stats[methods[0]][alpha].append(conditional);stats[methods[1]][alpha].append(full)
            if conditional[5]:errors[methods[0]]=summary['error'] or 'Invalid retained score-bootstrap interval'
            if full[5]:errors[methods[1]]=summary['error'] or 'Unbounded refit-bootstrap interval after retained failed replicates'
        ticks.append(time.perf_counter()-start)
        tables.append(dict(record,methods={methods[0]:dict(error=errors.get(methods[0]),interval_evidence=dict(kind='score_bootstrap',repeats=repeats,seed=seed)),methods[1]:dict(error=errors.get(methods[1]),interval_evidence=evidence,failed_bootstrap_replicates=refit['failures'])}))
    payload={m+'__nuisance':nuisances for m in methods}
    for m in methods:
        for alpha in (.1,.05,.01):payload[m+'__stats__'+str(alpha)]=np.asarray(stats[m][alpha])
    payload.update({k:raw[k] for k in ('truth','fold_ids','mechanism_id')})
    with subset.open('xb') as f:np.savez_compressed(f,**raw)
    with out.open('xb') as f:np.savez_compressed(f,**payload)
    report=dict(raw_sha256=sha(subset),predictions_sha256=sha(out),protocol_sha256=sha(protocol_path),full_raw_sha256=group['raw_sha256'],selected_table_indices=list(range(count)),
        source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('baselines.py'),Path(__file__).with_name('evaluate.py'))},methods=methods,tables=tables,
        repeats=repeats,per_table_diagnostic_seconds=ticks,all_attempts_and_failed_replicates_retained=True,
        limitations=['First fixed table indices, not outcome selected. Score bootstrap conditions on saved OOF nuisances; full refit repeats nuisance fitting and outer folds.',
            'Finite B limits extreme-tail quantile resolution, especially99percent intervals; no finite-sample bootstrap coverage guarantee.'],
        metrics=[dict(method=m,**metrics(payload[m+'__stats__'+str(alpha)],raw['truth'],alpha)) for m in methods for alpha in (.1,.05,.01)])
    with metaout.open('x',encoding='utf-8') as f:json.dump(clean(report),f,indent=2,allow_nan=False)
    return dict(group=name,kind=kind,bootstrap=True,attempts=count,repeats=repeats,prediction_path=str(out),raw_path=str(subset),metadata_path=str(metaout))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--raw-manifest',required=True);p.add_argument('--kind',choices=('linear','gbm','rf','oracle','all'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--authorization');p.add_argument('--task-index',type=int,default=int(os.environ.get('SGE_TASK_ID','1'))-1);p.add_argument('--task-count',type=int,default=1);p.add_argument('--estimator-jobs',type=int,default=0,help='0 uses actual NSLOTS, otherwise explicit estimator jobs')
    p.add_argument('--split');p.add_argument('--bank');p.add_argument('--bootstrap-subset',action='store_true',help='Only prespecified18synthetic cells, firstfixed30tables, separate score/refit artifacts');args=p.parse_args()
    manifest=json.loads(Path(args.raw_manifest).read_text(encoding='utf-8'))
    if manifest['protocol_sha256']!=sha(args.protocol):raise ValueError('Manifest protocol binding')
    groups=[g for g in manifest['groups'] if (args.split is None or g['split']==args.split) and (args.bank is None or g['bank']==args.bank)]
    if args.bootstrap_subset:groups=[g for g in groups if bootstrap_eligible(g)]
    if not 0<=args.task_index<args.task_count:raise ValueError('Task index is zero-based and below task count')
    # Process-level job parallelism is supplied by the real SGE array/NSLOTS.
    # A caller may instead explicitly allocate all estimator jobs to one RF fit.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=max(1,int(os.environ.get('NSLOTS','1')))):
        kinds=('linear','gbm','rf','oracle') if args.kind=='all' else (args.kind,)
        jobs=[(group,kind) for group in groups for kind in kinds]
        workers=args.estimator_jobs if args.estimator_jobs else int(os.environ.get('NSLOTS','1'))
        for group,kind in jobs[args.task_index::args.task_count]:
            result=run_bootstrap_cell(group,kind,args.protocol,args.output,args.authorization,workers) if args.bootstrap_subset else run_cell(group,kind,args.protocol,args.output,args.authorization,workers)
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
