"""Independent canonical evaluation, interval calibration and saved-bank gates.

Truth is an evaluation label only, never included in predictor payloads.
Every attempted table is retained at every nominal level.
"""
from pathlib import Path
from statistics import NormalDist
import hashlib,json,math,os,time
import numpy as np
from baselines import unpack

ALPHAS=(.10,.05,.01)
COLUMNS=('estimate','se','lo','hi','clipping_rate','failed')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value


def canonical_nuisance(nuisance):
    if isinstance(nuisance,dict):
        return np.column_stack([nuisance[k] if k in nuisance else nuisance[alias] for k,alias in
                                [('g','g'),('q0','q0'),('q1','q1'),('u0','mu10'),('u1','mu11')]])
    values=np.asarray(nuisance,float)
    if values.ndim!=2:raise ValueError('Nuisance matrix')
    if values.shape[1]==5:return values
    if values.shape[1]==7:
        from scipy.special import expit
        return np.column_stack((expit(values[:,:3]),values[:,5:7]))
    raise ValueError('Expected five canonical nuisances or seven logits/means')


def canonical_score(table,nuisance,clip=.01):
    """Full-n NIE=S11-S10; finite active-arm arithmetic; variance ddof=1."""
    try:
        x,a,m,y=unpack(table);nu=canonical_nuisance(nuisance);n=len(x)
        if n<2 or nu.shape!=(n,5) or not np.isfinite(nu).all():raise ValueError('Invalid nuisance/data shape')
        if not 0<=clip<.5 or np.any((nu[:,:3]<0)|(nu[:,:3]>1)):raise ValueError('Invalid probabilities/clip')
        prob=np.clip(nu[:,:3],clip,1-clip) if clip else nu[:,:3].copy()
        if np.any((prob<=0)|(prob>=1)):raise ValueError('Noninterior untrimmed probabilities')
        g,q0,q1=prob.T;u0,u1=nu[:,3:].T
        eta0=u0+q0*(u1-u0);eta1=u0+q1*(u1-u0);um=np.where(m==1,u1,u0)
        ratio=np.where(m==1,q0/q1,(1-q0)/(1-q1));treated=a==1;control=~treated
        s11,s10=eta1.copy(),eta0.copy()
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            s11[treated]+=(y[treated]-eta1[treated])/g[treated]
            s10[treated]+=ratio[treated]*(y[treated]-um[treated])/g[treated]
            s10[control]+=(um[control]-eta0[control])/(1-g[control])
            score=s11-s10;estimate=float(score.mean());se=float(np.std(score,ddof=1)/np.sqrt(n))
        if not np.isfinite(score).all() or not se>0:raise ValueError('Nonfinite/zero score variance')
        return dict(estimate=estimate,se=se,score=score,clipping_rate=float(np.any(prob!=nu[:,:3],1).mean()),failed=False,error=None,n=n)
    except Exception as exc:
        return dict(estimate=np.nan,se=np.nan,score=None,clipping_rate=np.nan,failed=True,error=repr(exc),n=len(table['a']) if isinstance(table,dict) else len(table))


def failed_stats():return np.array([np.nan,np.nan,np.inf,-np.inf,np.nan,1.])


def interval_stats(summary,alpha=.05,quantiles=None,endpoints=None):
    if summary['failed']:return failed_stats()
    if not 0<alpha<1:raise ValueError('Invalid alpha')
    if endpoints is not None:lo,hi=map(float,endpoints)
    else:
        qlo,qhi=(-NormalDist().inv_cdf(1-alpha/2),NormalDist().inv_cdf(1-alpha/2)) if quantiles is None else map(float,quantiles)
        if not qlo<=qhi:return failed_stats()
        lo,hi=summary['estimate']-summary['se']*qhi,summary['estimate']-summary['se']*qlo
    if not np.isfinite([lo,hi]).all() or lo>hi:return failed_stats()
    return np.array([summary['estimate'],summary['se'],lo,hi,summary['clipping_rate'],0.])


def inference_from_nuisance(table,nuisance,alphas=ALPHAS,clip=.01):
    summary=canonical_score(table,nuisance,clip)
    return dict(summary=summary,stats={str(float(a)):interval_stats(summary,a) for a in alphas})


def score_vectors(stats,truth,alpha=.05):
    stats=np.asarray(stats,float);truth=np.asarray(truth,float)
    if stats.shape!=(len(truth),6) or not np.isfinite(truth).all():raise ValueError('Stats/truth shapes')
    if not np.isin(stats[:,5],(0,1)).all():raise ValueError('Failure flag must be binary')
    valid=(stats[:,5]==0)&np.isfinite(stats[:,:4]).all(1)&(stats[:,1]>0)&(stats[:,3]>=stats[:,2])
    covered=valid&(stats[:,2]<=truth)&(truth<=stats[:,3]);length=np.full(len(truth),np.inf);loss=length.copy()
    length[valid]=stats[valid,3]-stats[valid,2]
    loss[valid]=length[valid]+2/alpha*(np.maximum(stats[valid,2]-truth[valid],0)+np.maximum(truth[valid]-stats[valid,3],0))
    return dict(valid=valid,coverage=covered.astype(float),lower_miss=(valid&(truth<stats[:,2])).astype(float),
                upper_miss=(valid&(truth>stats[:,3])).astype(float),length=length,interval_score=loss)


def tail_summary(values):
    values=np.sort(np.asarray(values,float));n=len(values)
    if not n:raise ValueError('Empty tail summary')
    result=dict(mean=float(values.mean()),maximum=float(values[-1]),infinite=int(np.isinf(values).sum()))
    for q in (50,90,95,99):result['p'+str(q)]=float(values[max(0,int(np.ceil(n*q/100))-1)])
    k=max(1,int(np.ceil(n*.01)));total=float(values.sum())
    result['top_one_percent_contribution']=float(values[-k:].sum()/total) if np.isfinite(total) and total>0 else None
    return result


def metrics(stats,truth,alpha=.05,fixed_mechanism=False):
    stats=np.asarray(stats,float);truth=np.asarray(truth,float);v=score_vectors(stats,truth,alpha);valid=v['valid'];n=len(truth)
    p=float(v['coverage'].mean());z=NormalDist().inv_cdf(.975);den=1+z*z/n;center=(p+z*z/(2*n))/den;rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    errors=stats[valid,0]-truth[valid];mean=lambda x:float(np.mean(x)) if len(x) else np.nan
    result=dict(attempts=n,valid_attempts=int(valid.sum()),failures=int((~valid).sum()),alpha=float(alpha),coverage=p,
        coverage_low=float(center-rad),coverage_high=float(center+rad),lower_miss=float(v['lower_miss'].mean()),upper_miss=float(v['upper_miss'].mean()),
        failure_rate=float((~valid).mean()),tail_denominator='All attempted tables; failures separately accounted, not assigned a directional miss',mean_length=float(v['length'].mean()),interval_score=float(v['interval_score'].mean()),
        bias=mean(errors),rmse=float(np.sqrt(mean(errors**2))),mean_se=mean(stats[valid,1]),length_tails=tail_summary(v['length']),IS_tails=tail_summary(v['interval_score']))
    if fixed_mechanism:
        if not np.all(truth==truth[0]):raise ValueError('Fixed-mechanism SD cannot pool different truths')
        sd=float(np.std(stats[valid,0],ddof=1)) if valid.sum()>1 else np.nan
        result.update(empirical_sd=sd,mean_se_over_empirical_sd=result['mean_se']/sd if sd>0 else np.nan,
            rms_se_over_empirical_sd=float(np.sqrt(mean(stats[valid,1]**2)))/sd if sd>0 else np.nan,sd_valid_attempts=int(valid.sum()))
    return result


def paired_difference(left,right,truth,alpha=.05):
    a,b=score_vectors(left,truth,alpha),score_vectors(right,truth,alpha);result={}
    for key in ('coverage','lower_miss','upper_miss','length','interval_score'):
        if not (np.isfinite(a[key]).all() and np.isfinite(b[key]).all()):result[key]=dict(available=False,difference=None,reason='All failures retained; infinite loss difference undefined')
        else:
            d=a[key]-b[key];se=float(np.std(d,ddof=1)/np.sqrt(len(d)));z=NormalDist().inv_cdf(.975)
            result[key]=dict(available=True,difference=float(d.mean()),mc_se=se,mc95_low=float(d.mean()-z*se),mc95_high=float(d.mean()+z*se))
    return result


class EmpiricalErrorCalibration:
    """Independent calibration; failed calibration attempts occupy infinite tails.

    Finite-sample order ranks calibrate exchangeable simulated task errors,
    not a claim of fixed-mechanism coverage or ordinary predictive conformal.
    """
    def __init__(self,group_keys=()):
        if any(k not in ('p','n') for k in group_keys):raise ValueError('Only observable p/n calibration groups')
        self.group_keys=tuple(group_keys);self.groups={};self.ids=set()
    def fit(self,records,blocked_mechanism_ids=()):
        if self.groups:raise ValueError('Calibration immutable after fit')
        blocked=set(map(str,blocked_mechanism_ids))
        for r in records:
            if r['split'] not in ('calibration','calibrate'):raise ValueError('Independent calibration split required')
            mid=str(r['mechanism_id'])
            if mid in blocked:raise ValueError('Calibration mechanism overlaps fitting/evaluation')
            self.ids.add(mid);key=tuple(r[k] for k in self.group_keys);ss=r['summary']
            value=(ss['estimate']-float(r['truth']))/ss['se'] if not ss['failed'] and ss['se']>0 else None
            self.groups.setdefault(key,[]).append(float(value) if value is not None and np.isfinite(value) else None)
        if not self.groups:raise ValueError('Empty calibration')
        return self
    def quantiles(self,alpha=.05,**observables):
        values=self.groups[tuple(observables[k] for k in self.group_keys)];n=len(values)
        low=np.sort([-np.inf if v is None else v for v in values]);high=np.sort([np.inf if v is None else v for v in values])
        li=int(np.floor((n+1)*alpha/2))-1;ui=int(np.ceil((n+1)*(1-alpha/2)))-1
        return (-np.inf if li<0 else float(low[li]),np.inf if ui>=n else float(high[ui]))
    def state(self):return clean(dict(group_keys=self.group_keys,groups=[dict(key=k,values=v) for k,v in self.groups.items()],mechanism_ids=sorted(self.ids),failed_calibration_retained=True))


def bootstrap_score_interval(summary,alpha=.05,repeats=499,seed=0):
    """Conditional OOF-score multiplier/resampling diagnostic; no nuisance refit.

    This is deliberately labeled score bootstrap, distinct from a full-pipeline
    nonparametric refit bootstrap. Never advertised as accounting for selection.
    """
    if summary['failed']:return (np.nan,np.nan)
    rng=np.random.default_rng(seed);s=summary['score'];n=len(s);means=np.array([s[rng.integers(n,size=n)].mean() for _ in range(repeats)])
    lower,upper=np.quantile(means,[alpha/2,1-alpha/2]);return (float(2*summary['estimate']-upper),float(2*summary['estimate']-lower))


def bootstrap_refit_interval(table,predictor,alpha=.05,repeats=199,seed=0):
    """Full table resampling/refit subset; any failed replicate stays unbounded.

    predictor receives observed resample only and fresh label-independent folds.
    Caller must prespecify table subset and repeats before test access.
    """
    x,a,m,y=unpack(table);rng=np.random.default_rng(seed);values=[];failures=0;n=len(x)
    for b in range(repeats):
        ids=rng.integers(n,size=n);fold=np.empty(n,np.int8);fold[rng.permutation(n)]=np.arange(n)%2
        sampled=dict(x=x[ids],a=a[ids],m=m[ids],y=y[ids])
        try:
            result=predictor(sampled,fold_ids=fold,seed=int(rng.integers(2**31-1)));ss=canonical_score(sampled,result['nuisance'])
            if ss['failed']:raise ValueError(ss['error'])
            values.append(ss['estimate'])
        except Exception:failures+=1;values.append(None)
    lo=np.sort([-np.inf if v is None else v for v in values]);hi=np.sort([np.inf if v is None else v for v in values])
    return dict(endpoints=(float(lo[max(0,int(np.floor(repeats*alpha/2))-1)]),float(hi[min(repeats-1,int(np.ceil(repeats*(1-alpha/2)))-1)])),repeats=repeats,failures=failures,all_replicates_retained=True,interval_evidence=dict(kind='refit_bootstrap',estimates=values,seed=seed))


def evaluate_table(table,fold_ids,predictors,alphas=ALPHAS,seed=0,calibrators=None,bootstrap_repeats=0):
    """One matched table. Predictor sees no truth, mechanism ID or family label."""
    x,a,m,y=unpack(table);observed=dict(x=x,a=a,m=m,y=y);truth=float(table['theta_NIE']);records={}
    for method,predictor in predictors.items():
        start=time.perf_counter()
        try:
            output=predictor(observed,fold_ids=np.asarray(fold_ids),seed=seed)
            if output.get('error'):raise ValueError(output['error'])
            nuisance=canonical_nuisance(output['nuisance']);summary=canonical_score(observed,nuisance)
            statistics={}
            for alpha in alphas:
                key=str(float(alpha));statistics[key]=interval_stats(summary,alpha)
                if 'intervals' in output:statistics[key]=interval_stats(summary,alpha,endpoints=output['intervals'][key])
            records[method]=dict(nuisance=nuisance,summary=summary,stats=statistics,seconds=time.perf_counter()-start,metadata=output.get('metadata',{}),error=summary['error'],interval_evidence=output.get('interval_evidence'))
            if calibrators and method in calibrators:
                cal=calibrators[method]
                if str(table.get('mechanism_id','')) in cal.ids:raise ValueError('Calibration/test mechanism leakage')
                values=cal.groups[tuple(dict(p=x.shape[1],n=len(x))[k] for k in cal.group_keys)]
                records[method+'_calibrated']=dict(summary=summary,nuisance=nuisance,stats={str(float(q)):interval_stats(summary,q,quantiles=cal.quantiles(q,p=x.shape[1],n=len(x))) for q in alphas},metadata=dict(calibration=cal.state()),error=summary['error'],interval_evidence=dict(kind='calibration',values=values))
            if bootstrap_repeats:
                records[method+'_score_bootstrap']=dict(summary=summary,nuisance=nuisance,stats={str(float(q)):interval_stats(summary,q,endpoints=bootstrap_score_interval(summary,q,bootstrap_repeats,seed)) for q in alphas},metadata=dict(repeats=bootstrap_repeats,conditional_score_only=True),error=summary['error'],interval_evidence=dict(kind='score_bootstrap',repeats=bootstrap_repeats,seed=seed))
        except Exception as exc:
            records[method]=dict(nuisance=np.full((len(x),5),np.nan),summary=canonical_score(observed,np.full((len(x),5),np.nan)),stats={str(float(q)):failed_stats() for q in alphas},seconds=time.perf_counter()-start,error=repr(exc),metadata=dict(whole_table_failure=True))
            for suffix,enabled in (('_calibrated',calibrators and method in calibrators),('_score_bootstrap',bootstrap_repeats)):
                if enabled:records[method+suffix]=dict(records[method],metadata=dict(whole_table_failure=True,derived_from=method))
    return dict(truth=truth,records=records,alphas=list(alphas),columns=COLUMNS,all_attempts_retained=True)


class EvaluationGuard:
    def __init__(self,protocol_path,authorization_path=None):
        self.path=Path(protocol_path);self.protocol=json.loads(self.path.read_text(encoding='utf-8'));self.authorization=Path(authorization_path) if authorization_path else None
    def check(self,split):
        p=self.protocol
        if not (p.get('frozen') is True or 'frozen' in p.get('status','')):raise ValueError('Frozen protocol required before evaluation')
        for relative,digest in p['source_hashes'].items():
            path=self.path.parent/relative
            if not path.exists():path=self.path.parent/'code'/relative
            if sha(path)!=digest:raise ValueError('Frozen source changed: '+relative)
        if split in ('test','fixed_test','semisynthetic_test'):
            if self.authorization is None:raise ValueError('Explicit test authorization required before test read')
            a=json.loads(self.authorization.read_text(encoding='utf-8'))
            if a.get('authorized') is not True or a['protocol_sha256']!=sha(self.path):raise ValueError('Test release binding failed')
        return True


def available_workers(memory_available_bytes=None,per_worker_bytes=None):
    """Respect real scheduler slots and caller memory estimate, no global cap."""
    slots=int(os.environ.get('NSLOTS',os.cpu_count() or 1))
    if memory_available_bytes is not None and per_worker_bytes is not None:slots=min(slots,max(1,int(memory_available_bytes)//int(per_worker_bytes)))
    return max(1,slots)


def benchmark_loaded(table,fold_ids,predictors,alphas=ALPHAS,warmups=3,repeats=10,seed=0,synchronize=None):
    """Loaded-model full callback timing: fresh encodings/fits, never saved cache.

    Caller supplies a CUDA synchronization hook when predictors use a GPU.
    All timings including failed attempts retained; methods rotate each repeat.
    """
    methods=tuple(predictors);times={m:[] for m in methods};failures={m:[] for m in methods}
    sync=synchronize or (lambda:None)
    for rep in range(warmups+repeats):
        order=methods[rep%len(methods):]+methods[:rep%len(methods)]
        for method in order:
            sync();start=time.perf_counter()
            result=evaluate_table(table,fold_ids,{method:predictors[method]},alphas,seed+rep)['records'][method]
            sync();elapsed=1000*(time.perf_counter()-start)
            failures[method].append(bool(result['summary']['failed'] or any(s[5]!=0 for s in result['stats'].values())))
            if rep>=warmups:times[method].append(elapsed)
    return dict(protocol='Loaded models, independent full table callback each attempt; no prediction cache, rotated order',
        measurements=[dict(method=m,raw_ms=times[m],median_ms=float(np.median(times[m])),mean_ms=float(np.mean(times[m])),
            warmup_failures=sum(failures[m][:warmups]),timed_failures=sum(failures[m][warmups:]),failure_flags=failures[m]) for m in methods],warmups=warmups,repeats=repeats)
