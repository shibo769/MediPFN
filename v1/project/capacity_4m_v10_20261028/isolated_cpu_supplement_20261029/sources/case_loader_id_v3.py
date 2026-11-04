"""Case reconstruction for preregistered 4M real-X and stress evaluations.

No test identities are invented here: callers supply frozen protocol cases.
The model and simulation labels are never supplied to PFN at inference.
"""
from functools import lru_cache
from pathlib import Path
import hashlib, json, os, sys
import numpy as np
from scipy.special import expit

B = Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9')
D = B/'capacity_4m_v10_20261028'/'supplemental_wave_20261028'
sys.path.insert(0,str(B/'code'))
from data import sample_mechanism, make_folds

SCENARIOS = ('control','high_noise','heavy_tail','weak_overlap','near_zero')
REAL_FAMILIES = ('smooth','tree','rff','mlp','combinations')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return h.hexdigest()

@lru_cache(maxsize=1)
def populations():
    root=Path(os.environ.get('NIE_WAVE_XDIR',str(D/'frozen_covariates')))
    meta=json.loads((root/'provenance.json').read_text(encoding='utf8'))
    if sha(root/'x_populations.npz') != meta['npz_sha256']:
        raise RuntimeError('real X artifact hash mismatch')
    with np.load(root/'x_populations.npz',allow_pickle=False) as f:
        result={name:np.ascontiguousarray(f[name],dtype=np.float32) for name in f.files}
    if set(result)!=set(meta['populations']):raise RuntimeError('real X source mismatch')
    for name,x in result.items():
        row=meta['populations'][name]
        if (list(x.shape)!=row['shape'] or not np.isfinite(x).all() or
            hashlib.sha256(x.tobytes()).hexdigest()!=row['population_sha256']):
            raise RuntimeError('real X population validation failed: '+name)
        x.flags.writeable=False
    return result

def mechanism_for_case(case):
    kind=case['kind']; seed=int(case['mechanism_seed']); p=int(case['p'])
    if kind=='id':
        if case['family'] not in ('linear','smooth','heterosked','weak_overlap','covariate_shift'):
            raise ValueError('unregistered ID family')
        s=sample_mechanism(seed,p,case['family'])
        if case.get('population_rng_used') is not True or int(case.get('population_seed_derived',-1))!=s.population_seed:
            raise ValueError('ID population RNG does not match registered identity')
    elif kind=='stress':
        scenario=case['scenario']
        if scenario not in SCENARIOS or p!=10:raise ValueError('unregistered stress case')
        s=sample_mechanism(seed,p,'smooth')
        if scenario=='high_noise':s.parameters['noise_scale']*=3.0
        elif scenario=='heavy_tail':s.parameters['noise_df']=3
        elif scenario=='weak_overlap':s.parameters['floor']=.002;s.parameters['propensity_scale']=5.0
        elif scenario=='near_zero':s.parameters['gamma']=.01;s.parameters['weights'][2,:]=0.0
    elif kind=='realx':
        source=case['source']; family=case['family']
        if family not in REAL_FAMILIES:raise ValueError('unregistered real-X family')
        pop=populations()[source]
        if p!=pop.shape[1]:raise ValueError('real X dimensions mismatch')
        s=sample_mechanism(seed,p,family,population_size=len(pop),population=pop)
    else:raise ValueError('unregistered case kind')
    if s._theta_NIE is not None:raise RuntimeError('truth was accessed before stress modification')
    return s

def independent_truth(s):
    """Recompute finite-X NIE from a separate algebraic implementation.

    It intentionally does not call s.nuisance or s.theta_NIE.  This audits
    the finite population estimand, not discovery of a real-world NIE.
    """
    x=np.asarray(s.population,dtype=np.float64); z=(x-x.mean(0))/np.maximum(x.std(0),1e-6)
    par=s.parameters; linear=z@par['weights'].T
    pairs=par['pairs']; interaction=np.zeros(len(z))
    if len(pairs):interaction=np.mean(np.tanh(z[:,pairs[:,0]])*np.tanh(z[:,pairs[:,1]]),axis=1)
    nonlinear=np.zeros(len(z))
    if s.family in ('smooth','heterosked','covariate_shift'):
        nonlinear=.45*(np.sin(z@par['smooth_direction'])+.4*np.tanh(z[:,par['active'][0]]))+.3*interaction
    if s.family in ('tree','combinations'):
        nonlinear+=1.4*np.mean(np.where(z[:,par['tree_columns']]>par['tree_thresholds'],par['tree_values'],-par['tree_values']),axis=1)
    if s.family in ('rff','combinations'):
        nonlinear+=np.sin(z@par['rff_frequencies'].T+par['rff_phase'])@par['rff_weights']/np.sqrt(8)
    if s.family in ('mlp','combinations'):
        nonlinear+=np.tanh(z@par['mlp_weights'].T+par['mlp_bias'])@par['mlp_output']/np.sqrt(12)
    floor=par['floor']; varying=s.family!='linear'
    l0=par['intercepts'][1]+linear[:,1]+.8*nonlinear+(.2*interaction if varying else 0.)
    delta=par['gamma']+(.35*np.tanh(linear[:,2]) if varying else 0.)
    q0=floor+(1-2*floor)*expit(l0);q1=floor+(1-2*floor)*expit(l0+delta)
    # E[Y(1,1)-Y(1,0)|X], the treated mediator contrast.
    effect=par['mediator_effect']+par['am_effect']+(.35*np.tanh(linear[:,2])+.25*nonlinear+.25*np.tanh(linear[:,4]) if varying else 0.)
    return float(np.mean((q1-q0)*effect))

def make_case(case):
    for k in ('index','kind','scenario','p','n','rep','mechanism_seed','observation_seed','fold_seed','view_seed','cluster_id'):
        if k not in case:raise KeyError('missing preregistered case field '+k)
    if int(case['n'])<4:raise ValueError('n too small')
    s=mechanism_for_case(case)
    truth=float(s.theta_NIE); separate=independent_truth(s)
    if not np.isfinite(truth) or not np.isclose(truth,separate,rtol=1e-12,atol=1e-12):
        raise RuntimeError('independent NIE reconstruction mismatch')
    table=s.draw(int(case['n']),int(case['observation_seed']),include_truth=False)
    observed=np.ascontiguousarray(np.column_stack([table['x'],table['a'],table['m'],table['y']]),dtype=np.float32)
    fold=np.ascontiguousarray(make_folds(int(case['n']),int(case['fold_seed'])))
    if observed.shape!=(int(case['n']),int(case['p'])+3) or not np.isfinite(observed).all():
        raise RuntimeError('invalid generated table')
    return observed,fold,truth,s.population_hash
