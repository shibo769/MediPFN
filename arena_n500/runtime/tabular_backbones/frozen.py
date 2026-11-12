"""Frozen TabPFN nuisance models for a vector-mediator NIE score.

This is a mediation adaptation using a TabPFN checkpoint, NOT original TabCon.
No truth labels are accepted by this module. Every predictive label comes only
from outer support data; nested eta0 targets use inner out-of-fold mu predictions.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
from pathlib import Path
from functools import lru_cache
from statistics import NormalDist
import time
import numpy as np


@dataclass(frozen=True)
class BackboneConfig:
    model_version: str
    package_version: str
    classifier_path: str
    classifier_sha256: str
    regressor_path: str
    regressor_sha256: str
    device: str = 'cpu'
    n_estimators: int = 1
    n_preprocessing_jobs: int = 1
    probability_floor: float = .01


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return h.hexdigest()


def official_factory(config, classification, seed):
    """Explicit local checkpoint and version only; no automatic download/fallback."""
    import inspect
    from tabpfn import TabPFNClassifier,TabPFNRegressor
    from tabpfn.constants import ModelVersion
    if importlib.metadata.version('tabpfn')!=config.package_version:
        raise ValueError('TabPFN package version differs from frozen configuration')
    version=getattr(ModelVersion,{'v2':'V2','v3':'V3'}[config.model_version],None)
    if version is None:raise RuntimeError('Installed TabPFN does not support '+config.model_version)
    path=config.classifier_path if classification else config.regressor_path
    expected=config.classifier_sha256 if classification else config.regressor_sha256
    if not Path(path).is_file() or len(expected)!=64 or sha256(path)!=expected:
        raise ValueError('Missing or changed checkpoint; automatic download prohibited')
    cls=TabPFNClassifier if classification else TabPFNRegressor
    kwargs=dict(model_path=path,device=config.device,n_estimators=config.n_estimators,
                random_state=int(seed),n_preprocessing_jobs=config.n_preprocessing_jobs,
                ignore_pretraining_limits=False)
    if not set(kwargs).issubset(inspect.signature(cls.__init__).parameters):
        raise RuntimeError('Unsupported frozen TabPFN API')
    return cls.create_default_for_version(version,**kwargs)


def make_shared_frozen_v3_factory(config):
    """Load/hash two immutable V3 networks once; create new contexts each fit.

    Only neural weights and static architecture/inference metadata are shared.
    No observed labels, preprocessing, support cache or estimator state is reused.
    The first initialization consumes the first caller's support rows only.
    """
    if config.model_version!='v3':raise ValueError('Shared factory is for actual V3')
    cache={};counts={'checkpoint_initializations':0,'fresh_context_fits':0}

    class SharedContextEstimator:
        def __init__(self,classification,seed):
            self.classification=classification;self.seed=int(seed);self.estimator=None
        def fit(self,x,y):
            from tabpfn import TabPFNClassifier,TabPFNRegressor
            from tabpfn.base import ModelSpecs
            if self.classification not in cache:
                first=official_factory(config,self.classification,self.seed)
                first.fit(x,y)
                if len(first.models_)!=1:raise RuntimeError('Expected one checkpoint network')
                model=first.models_[0];model.eval()
                for p in model.parameters():p.requires_grad_(False)
                cache[self.classification]=ModelSpecs(model=model,
                    architecture_config=first.configs_[0],inference_config=first.inference_config_,
                    norm_criterion=None if self.classification else first.znorm_space_bardist_)
                counts['checkpoint_initializations']+=1
                # The fitted estimator and its observed-data preprocessing/cache
                # are discarded; only the static ModelSpecs remains.
                del first
            cls=TabPFNClassifier if self.classification else TabPFNRegressor
            self.estimator=cls(model_path=cache[self.classification],device=config.device,
                n_estimators=config.n_estimators,n_preprocessing_jobs=config.n_preprocessing_jobs,
                random_state=self.seed,ignore_pretraining_limits=False)
            self.estimator.fit(x,y);counts['fresh_context_fits']+=1
            return self
        def predict(self,q):return self.estimator.predict(q)
        def predict_proba(self,q):return self.estimator.predict_proba(q)
        def get_embeddings(self,q,data_source='test'):return self.estimator.get_embeddings(q,data_source=data_source)
        @property
        def classes_(self):return self.estimator.classes_
    def factory(actual_config,classification,seed):
        if actual_config!=config:raise ValueError('Different shared-backbone configuration')
        return SharedContextEstimator(classification,seed)
    factory.counts=counts
    factory.static_specs=cache
    return factory


@lru_cache(4)
def shared_frozen_factory(config):
    return make_shared_frozen_v3_factory(config) if config.model_version=='v3' else official_factory


def _x(x):
    # A constant input column supports source designs with no baseline covariates.
    return x if x.shape[1] else np.ones((len(x),1))


def _assignment(a,seed):
    r=np.random.default_rng(seed);out=np.empty(len(a),int)
    for value in (0,1):
        ix=np.flatnonzero(a==value)
        if len(ix)<4:raise ValueError('Insufficient treatment support for nested two-fold score')
        r.shuffle(ix);out[ix]=np.arange(len(ix))%2
    return out


def _predict(factory,config,classification,seed,x,y,q,embeddings=False):
    if len(x)!=len(y) or not len(x):raise ValueError('Invalid support labels')
    if (x.shape[1]==0 or np.all(np.std(x,axis=0)<1e-12)) and not getattr(factory,'learns_constant_calibration',False):
        # No-X designs have constant pi/eta functions; empirical means avoid
        # sending a zero-information predictor matrix through a preprocessor.
        if embeddings:raise RuntimeError('Constant-context task has no learned query embedding')
        return np.full(len(q),float(np.mean(y))),None
    if not classification and np.std(y)<1e-10:
        return np.full(len(q),float(np.mean(y))),None
    model=factory(config,classification,seed);model.fit(x,y)
    if classification:
        classes=np.asarray(model.classes_)
        if set(classes.tolist())!={0,1}:raise ValueError('Need both treatment classes in support')
        p=np.asarray(model.predict_proba(q),float)
        prediction=p[:,int(np.flatnonzero(classes==1)[0])]
        if p.shape!=(len(q),2) or not np.isfinite(p).all():raise ValueError('Invalid classification output')
    else:prediction=np.asarray(model.predict(q),float).reshape(-1)
    embed=None
    if embeddings:
        if not hasattr(model,'get_embeddings'):raise RuntimeError('Backbone does not expose embeddings')
        raw=np.asarray(model.get_embeddings(q,data_source='test'),float)
        if raw.ndim==2:raw=raw[None]
        if raw.ndim!=3 or raw.shape[1]!=len(q):raise RuntimeError('Unexpected backbone embedding shape')
        embed=np.concatenate([raw.mean((0,1)),raw.std((0,1))])
    if prediction.shape!=(len(q),) or not np.isfinite(prediction).all():raise ValueError('Invalid prediction')
    return prediction,embed


def estimate_table(table,config,seed=18031,factory=None,return_scores=False,
                   collect_embeddings=False):
    start=time.perf_counter()
    factory=shared_frozen_factory(config) if factory is None else factory
    if not 0<config.probability_floor<.5:raise ValueError('Invalid probability floor')
    x=np.asarray(table['x'],float);m=np.asarray(table['m'],float)
    a=np.asarray(table['a'],float).reshape(-1);y=np.asarray(table['y'],float).reshape(-1)
    n=len(a)
    if x.ndim!=2 or m.ndim!=2 or any(len(v)!=n for v in (x,m,y)):
        raise ValueError('Invalid table shape')
    if not all(np.isfinite(v).all() for v in (x,m,a,y)) or not np.isin(a,[0,1]).all():
        raise ValueError('Invalid observed data')
    folds=_assignment(a,seed);scores=np.full(n,np.nan);details=[];pooled=[]
    predictions={k:np.full(n,np.nan) for k in ('pi','g','mu1','eta0','eta1')}
    for fold in (0,1):
        si=np.flatnonzero(folds!=fold);qi=np.flatnonzero(folds==fold)
        sx,sm,sa,sy=x[si],m[si],a[si],y[si];qx,qm=x[qi],m[qi]
        base=int(seed)+100*fold
        pi,_=_predict(factory,config,True,base,_x(sx),sa,_x(qx))
        gp,_=_predict(factory,config,True,base+1,np.column_stack([sx,sm]),sa,np.column_stack([qx,qm]))
        mu,embed=_predict(factory,config,False,base+2,np.column_stack([sx,sa,sm]),sy,
                          np.column_stack([qx,np.ones(len(qi)),qm]),collect_embeddings)
        if embed is not None:pooled.append(embed)
        # Fit eta0 on support A=0 only, using inner out-of-fold mu1 pseudo labels.
        inner=_assignment(sa,base+3);pseudo=np.full(len(si),np.nan)
        for infold in (0,1):
            inner_train=inner!=infold;inner_query=(inner==infold)&(sa==0)
            v,_=_predict(factory,config,False,base+4+infold,
                          np.column_stack([sx[inner_train],sa[inner_train],sm[inner_train]]),sy[inner_train],
                          np.column_stack([sx[inner_query],np.ones(inner_query.sum()),sm[inner_query]]))
            pseudo[inner_query]=v
        zero=sa==0;one=sa==1
        eta0,_=_predict(factory,config,False,base+6,_x(sx[zero]),pseudo[zero],_x(qx))
        # eta1=E[Y|A=1,X] follows directly from iterated expectation.
        eta1,_=_predict(factory,config,False,base+7,_x(sx[one]),sy[one],_x(qx))
        floor=config.probability_floor;pc=np.clip(pi,floor,1-floor);gc=np.clip(gp,floor,1-floor)
        ratio=(1-gc)/gc*pc/(1-pc)
        aq,yq=a[qi],y[qi]
        psi11=aq/pc*(yq-eta1)+eta1
        psi10=aq/pc*ratio*(yq-mu)+(1-aq)/(1-pc)*(mu-eta0)+eta0
        scores[qi]=psi11-psi10
        for k,v in zip(('pi','g','mu1','eta0','eta1'),(pc,gc,mu,eta0,eta1)):predictions[k][qi]=v
        details.append(dict(fold=fold,support_rows=len(si),query_rows=len(qi),
                            pi_clipped=float(np.mean(pc!=pi)),g_clipped=float(np.mean(gc!=gp)),
                            ratio_max=float(ratio.max())))
    if not np.isfinite(scores).all():raise FloatingPointError('Incomplete/nonfinite score')
    estimate=float(scores.mean());se=float(scores.std(ddof=1)/np.sqrt(n))
    out=dict(estimate=estimate,standard_error=se,n=n,seconds=time.perf_counter()-start,
             model_version=config.model_version,method='frozen-TabPFN mediation Bayes-score; not TabCon',
             diagnostics=details,backbone_config=asdict(config),interval_type='uncalibrated score-Wald')
    for level in (.90,.95,.99):
        radius=NormalDist().inv_cdf((1+level)/2)*se
        out['interval_'+str(round(100*level))]=[estimate-radius,estimate+radius]
    if pooled:out['pooled_context']=np.mean(pooled,axis=0).tolist()
    if return_scores:
        out.update(scores=scores.tolist(),fold_assignment=folds.tolist(),
                   nuisances={k:v.tolist() for k,v in predictions.items()})
    return out
