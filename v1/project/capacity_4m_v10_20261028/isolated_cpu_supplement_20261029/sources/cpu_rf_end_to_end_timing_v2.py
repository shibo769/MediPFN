"""RF cold timing candidate on a pre-existing registered observed table.

This is a non-authorizing helper. A separately reviewed stage must bind the
table, case, seed reservation, dependency sources and hardware. Each measured
call starts with the table read and ends after interval JSON serialization.
No simulator, fitting selection, scheduler operation or file write is invoked.
"""
import hashlib,io,json,time
from pathlib import Path
import numpy as np

METHODS=('rf_dml_90','rf_dml_95','rf_dml_99','rf_bootstrap_90','rf_bootstrap_95','rf_bootstrap_99')


def measure_one(case,method,table_path,table_sha256,truth,initial_seed,bootstrap_seed,slots,deps):
    """Measure one fixed case/method; caller records exceptions as failures.

    ``deps`` must contain the already source-bound data, baselines, evaluate,
    and RF core modules. All bootstrap levels use the same registered root.
    """
    if method not in METHODS or type(slots) is not int or slots<1:raise ValueError('invalid timing method/resources')
    if not np.isfinite(float(truth)):raise ValueError('nonfinite truth')
    level={'90':.90,'95':.95,'99':.99}[method[-2:]]
    started=time.perf_counter()
    raw=Path(table_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=table_sha256:raise ValueError('observed table SHA mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        if set(z.files)!={'observed'}:raise ValueError('table schema')
        observed=np.array(z['observed'],copy=True)
    n,p=int(case['n']),int(case['p'])
    if observed.shape!=(n,p+3) or not np.isfinite(observed).all():raise ValueError('invalid observed table')
    fold=np.ascontiguousarray(deps['data'].make_folds(n,int(case['fold_seed'])))
    table={'x':observed[:,:p],'a':observed[:,p],'m':observed[:,p+1],'y':observed[:,p+2]}
    baseline,ev,core=deps['baselines'],deps['evaluate'],deps['rf_supplement_core']
    if method.startswith('rf_dml_'):
        fit=baseline.crossfit_nuisance(table,fold,kind='rf',seed=int(initial_seed),n_jobs=slots)
        summary=ev.canonical_score(table,fit['nuisance'])
        if fit.get('error') or summary['failed']:raise ValueError(fit.get('error') or summary['error'])
        endpoints=core.normal_interval(summary['estimate'],summary['se'],level)
    else:
        def predictor(sampled,fold_ids,seed):
            return baseline.crossfit_nuisance(sampled,fold_ids,kind='rf',seed=seed,n_jobs=slots)
        result=ev.bootstrap_refit_interval(table,predictor,.05,100,int(bootstrap_seed))
        estimates=result['interval_evidence']['estimates']
        if len(estimates)!=100 or result['failures']!=sum(v is None for v in estimates):raise ValueError('bootstrap accounting')
        if tuple(result['endpoints'])!=core.endpoints_from_replicates(estimates,.95):raise ValueError('bootstrap parity')
        endpoints=core.endpoints_from_replicates(estimates,level)
    lo,hi=map(float,endpoints)
    if not np.isfinite([lo,hi]).all() or lo>hi:raise ValueError('RF interval failure')
    encoded=json.dumps({'lo':lo,'hi':hi},allow_nan=False,separators=(',',':')).encode()
    seconds=time.perf_counter()-started
    covered=bool(lo<=float(truth)<=hi)  # evaluation is outside the inference timer
    identity=lambda a:{'sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),'dtype':str(a.dtype),'shape':list(a.shape)}
    return {'ok':True,'case_id':case['case_id'],'method':method,'seconds':seconds,'interval_json':encoded.decode(),
            'covered':covered,'table_sha256':table_sha256,'observed':identity(observed),'fold':identity(fold),'device':'cpu','slots':slots,
            'scope':'table read and SHA, fold/data preparation, RF fitting and interval JSON; excludes process/module startup, simulation and output file writes; OS cache not controlled; each nominal level reruns the same fixed B100 root, so per-level costs are not additive shared-batch cost'}


def production_entry():raise RuntimeError('requires independently reviewed protocol and registered timing stage')
if __name__=='__main__':production_entry()
