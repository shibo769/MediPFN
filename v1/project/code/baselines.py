"""Prespecified CPU nuisance baselines; every outer query is out of fold.

No oracle is used by deployable fits. Oracle is an explicitly privileged
diagnostic. All rows and genuine fit failures are retained by the evaluator.
"""
import math
import warnings
import numpy as np

KINDS=('linear','gbm','rf','oracle')
DEFAULTS={'linear':dict(C=1000.,max_iter=2000),
          'gbm':dict(max_iter=150,max_leaf_nodes=15,l2_regularization=1.,learning_rate=.05),
          'rf':dict(n_estimators=200,min_samples_leaf=5,max_features=1.)}


def unpack(table):
    if isinstance(table,dict):
        x=np.asarray(table['x'],float);a,m,y=(np.asarray(table[k],float) for k in ('a','m','y'))
    else:
        rows=np.asarray(table,float);x,a,m,y=rows[:,:-3],rows[:,-3],rows[:,-2],rows[:,-1]
    if x.ndim!=2 or not all(v.shape==(len(x),) for v in (a,m,y)):raise ValueError('Table shape')
    if not all(np.isfinite(v).all() for v in (x,a,m,y)):raise ValueError('Nonfinite table, no row deletion')
    if not np.isin(a,(0,1)).all() or not np.isin(m,(0,1)).all():raise ValueError('Binary A/M required')
    return x,a,m,y


def _learner(kind,classification,seed,n_jobs,options):
    from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor,RandomForestClassifier,RandomForestRegressor
    from sklearn.linear_model import LogisticRegression,LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    settings=dict(DEFAULTS[kind]);settings.update(options or {})
    if kind=='linear':
        model=LogisticRegression(**settings,random_state=seed) if classification else LinearRegression()
        return make_pipeline(StandardScaler(),model)
    if kind=='gbm':
        cls=HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        return cls(**settings,random_state=seed)
    cls=RandomForestClassifier if classification else RandomForestRegressor
    return cls(**settings,random_state=seed,n_jobs=n_jobs)


def crossfit_nuisance(table,fold_ids,kind='linear',scm=None,seed=0,n_jobs=1,options=None):
    """Return canonical g,q0,q1,u0,u1, diagnostics and an explicit failure.

    g fits X->A; q fits (X,A)->M; outcome fits (X,A,M,A*M)->Y.
    Linear uses this fixed ordinary specification; GBM/RF use the same observed
    inputs. No correct-specification claim is made for arbitrary mechanisms.
    n_jobs is per estimator; use 1 with table-level allocated-slot parallelism.
    """
    x,a,m,y=unpack(table);fold_ids=np.asarray(fold_ids)
    if fold_ids.shape!=(len(x),) or set(np.unique(fold_ids))!={0,1}:raise ValueError('Two nonempty prescribed folds required')
    if kind not in KINDS:raise ValueError('Unregistered baseline')
    out=np.full((len(x),5),np.nan);diagnostics=[]
    try:
        if kind=='oracle':
            if scm is None:raise ValueError('Oracle requires explicitly supplied SCM')
            nu=scm.nuisance(x)
            out=np.column_stack([nu[k] for k in ('g','q0','q1','u0','u1')])
        else:
            from sklearn.exceptions import ConvergenceWarning
            for f in (0,1):
                train=fold_ids!=f;query=~train;xx=x[train];aa,mm,yy=a[train],m[train],y[train];qx=x[query]
                if len(np.unique(aa))!=2 or len(np.unique(mm))!=2:raise ValueError('One-class support A/M')
                classifiers=[_learner(kind,True,seed+10*f+j,n_jobs,options) for j in (0,1)]
                outcome=_learner(kind,False,seed+10*f+2,n_jobs,options)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    classifiers[0].fit(xx,aa);classifiers[1].fit(np.column_stack((xx,aa)),mm)
                    features=np.column_stack((xx,aa,mm,aa*mm));outcome.fit(features,yy)
                if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise ArithmeticError('Baseline nonconvergence')
                if kind=='linear' and outcome[-1].rank_<features.shape[1]:raise ArithmeticError('Rank-deficient linear outcome fit')
                zero,one=np.zeros(len(qx)),np.ones(len(qx))
                out[query,0]=classifiers[0].predict_proba(qx)[:,1]
                for av in (0,1):out[query,1+av]=classifiers[1].predict_proba(np.column_stack((qx,av*one)))[:,1]
                for mv in (0,1):out[query,3+mv]=outcome.predict(np.column_stack((qx,one,mv*one,mv*one)))
                diagnostics.append(dict(fold=f,support_rows=int(train.sum()),query_rows=int(query.sum())))
        if out.shape!=(len(x),5) or not np.isfinite(out).all():raise ArithmeticError('Invalid baseline output')
        return dict(nuisance=out,error=None,metadata=dict(kind=kind,oracle=kind=='oracle',seed=int(seed),folds=diagnostics,
            settings=dict(DEFAULTS.get(kind,{}),**(options or {})),per_estimator_jobs=n_jobs,all_rows_retained=True))
    except Exception as exc:
        return dict(nuisance=np.full((len(x),5),np.nan),error=repr(exc),metadata=dict(kind=kind,oracle=kind=='oracle',seed=int(seed),folds=diagnostics,whole_table_failure=True))


def legacy_eligible(table):
    """Old frozen FSP/v1 accept exactly three X; never pad/truncate to fake it."""
    return unpack(table)[0].shape[1]==3
