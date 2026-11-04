"""Independent initial RF and full bootstrap execution for one table.

The bootstrap helper does not depend on an initial fit. All levels inherit
the same 100 resamples, but a level-specific encoding failure is isolated.
"""
import time

def run(table,fold,truth,initial_seed,bootstrap_seed,slots,core,
        crossfit_nuisance,canonical_score,bootstrap_refit_interval,bootstrap_planned):
    row={'methods':{},'baseline_seed':int(initial_seed)}
    start=time.perf_counter()
    try:
        fit=crossfit_nuisance(table,fold,kind='rf',seed=initial_seed,n_jobs=slots)
        summary=canonical_score(table,fit['nuisance'])
        if fit.get('error') or summary['failed']:raise ValueError(fit.get('error') or summary['error'])
        row['estimate']=float(summary['estimate']);row['se']=float(summary['se'])
        for level in core.LEVELS:
            try:
                ep=core.normal_interval(summary['estimate'],summary['se'],level)
                row['methods']['rf_dml_'+str(level)]=core.encode_interval(*ep,truth)
            except Exception as ex:
                row['methods']['rf_dml_'+str(level)]={'ok':False,'error':repr(ex)}
    except Exception as ex:
        row['dml_error']=repr(ex)
        for level in core.LEVELS:row['methods']['rf_dml_'+str(level)]={'ok':False,'error':repr(ex)}
    row['initial_fit_seconds']=time.perf_counter()-start
    if bootstrap_planned:
        row['bootstrap_seed']=int(bootstrap_seed);start=time.perf_counter()
        try:
            def predictor(sampled,fold_ids,seed):
                return crossfit_nuisance(sampled,fold_ids,kind='rf',seed=seed,n_jobs=slots)
            result=bootstrap_refit_interval(table,predictor,.05,100,bootstrap_seed)
            values=result['interval_evidence']['estimates']
            if len(values)!=100 or result['failures']!=sum(v is None for v in values):raise RuntimeError('bootstrap replicate accounting')
            if tuple(result['endpoints'])!=core.endpoints_from_replicates(values,.95):raise RuntimeError('bootstrap rank parity')
            row['bootstrap_estimates']=values;row['bootstrap_failures']=result['failures']
            for level in core.LEVELS:
                try:
                    ep=core.endpoints_from_replicates(values,level)
                    row['methods']['rf_bootstrap_'+str(level)]=core.encode_interval(*ep,truth)
                except Exception as ex:
                    row['methods']['rf_bootstrap_'+str(level)]={'ok':False,'error':repr(ex)}
        except Exception as ex:
            row['bootstrap_error']=repr(ex)
            for level in core.LEVELS:row['methods']['rf_bootstrap_'+str(level)]={'ok':False,'error':repr(ex)}
        row['bootstrap_seconds_excluding_initial_fit']=time.perf_counter()-start
    return row
