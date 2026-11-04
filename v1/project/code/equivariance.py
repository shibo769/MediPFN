"""Pure causal-equivalence transforms for binary-mediator NIE at A=1.

M'=1-M swaps mediator labels, not treatment labels. All X columns can be
permuted when the oracle query applies the inverse permutation. Neither
operation changes patient identities, Y, folds, or the population NIE.
"""
from __future__ import annotations
import hashlib
import numpy as np

TRANSFORMS=('m_flip','x_permutation')

def checked_permutation(permutation,p):
    value=np.asarray(permutation)
    if value.shape!=(p,) or not np.issubdtype(value.dtype,np.integer) or not np.array_equal(np.sort(value),np.arange(p)):
        raise ValueError('Permutation must retain every observed X column exactly once')
    return value.astype(np.int64,copy=True)

def fixed_permutation(p,seed,group=''):
    tag=int.from_bytes(hashlib.blake2b(str(group).encode('utf-8'),digest_size=8).digest(),'little')
    result=np.random.default_rng(np.random.SeedSequence([int(seed),int(p),tag])).permutation(p)
    if p>1 and np.array_equal(result,np.arange(p)):result=np.roll(result,1)
    return result

def transform_rows(rows,kind,permutation=None):
    rows=np.asarray(rows)
    if rows.ndim!=2 or rows.shape[1]<4 or kind not in TRANSFORMS:raise ValueError('Expected [X...,A,M,Y] and supported causal-equivalence transform')
    result=rows.copy();p=rows.shape[1]-3
    if kind=='m_flip':
        if not np.isin(rows[:,-2],(0,1)).all():raise ValueError('Mediator must be binary')
        result[:,-2]=1-rows[:,-2]
    else:result[:,:p]=rows[:,checked_permutation(permutation,p)]
    return result

def transform_table(table,kind,permutation=None):
    if kind not in TRANSFORMS:raise ValueError('No treatment-flip or affine-Y heuristic is defined')
    result=dict(table)
    for key in ('x','a','m','y'):result[key]=np.asarray(table[key]).copy()
    if kind=='m_flip':
        if not np.isin(result['m'],(0,1)).all():raise ValueError('Mediator must be binary')
        result['m']=1-result['m']
    else:result['x']=result['x'][:,checked_permutation(permutation,result['x'].shape[1])]
    return result

def transform_nuisance(nuisance,kind,raw_logits=None):
    """Map canonical5, raw7, or named nuisances at corresponding patients.

    raw7 order: g-logit,q0-logit,q1-logit,mu00,mu01,mu10,mu11.
    canonical5: g,q0,q1,u0=mu10,u1=mu11. Dicts may expose either/both aliases.
    """
    if kind not in TRANSFORMS:raise ValueError('Unsupported causal-equivalence transform')
    if isinstance(nuisance,dict):
        out={key:np.asarray(value).copy() if isinstance(value,(np.ndarray,list,tuple)) else value for key,value in nuisance.items()}
        if kind=='m_flip':
            for key in ('q0','q1'):out[key]=1-np.asarray(nuisance[key])
            for left,right in (('mu00','mu01'),('mu10','mu11'),('u0','u1')):
                if left in nuisance and right in nuisance:out[left],out[right]=np.asarray(nuisance[right]).copy(),np.asarray(nuisance[left]).copy()
        return out
    values=np.asarray(nuisance)
    if values.ndim<2 or values.shape[-1] not in (5,7):raise ValueError('Expected canonical5 or raw7 nuisance columns')
    result=values.copy()
    if kind=='x_permutation':return result
    if raw_logits is None:raw_logits=values.shape[-1]==7
    result[...,1:3]=-values[...,1:3] if raw_logits else 1-values[...,1:3]
    if values.shape[-1]==5:result[...,3:5]=values[...,[4,3]]
    else:result[...,3:7]=values[...,[4,3,6,5]]
    return result

def equivalent_oracle(mechanism,query_x,kind,permutation=None):
    x=np.asarray(query_x)
    if kind=='x_permutation':
        permutation=checked_permutation(permutation,mechanism.p)
        x=x[:,np.argsort(permutation)]
    elif kind!='m_flip':raise ValueError('Unsupported causal-equivalence transform')
    return transform_nuisance(mechanism.nuisance(x),kind)

def implied_theta_moments(estimate,se,error_mean,error_variance):
    """Compare paired density heads in theta space, not raw Z space.

    Z=(estimate-theta)/SE, so E(theta)=estimate-SE*E(Z) and
    Var(theta)=SE^2 Var(Z). Pair estimates and SEs may differ.
    """
    estimate,se,error_mean,error_variance=np.broadcast_arrays(*map(np.asarray,(estimate,se,error_mean,error_variance)))
    if np.any(se<=0) or np.any(error_variance<0) or not all(np.isfinite(v).all() for v in (estimate,se,error_mean,error_variance)):raise ValueError('Invalid error/point moments')
    return estimate-se*error_mean,se**2*error_variance
