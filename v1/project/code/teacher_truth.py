"""Exact finite-population oracle NIE teacher and population EIF variance.

Enumerate all A/M cells and integrate Y analytically using its conditional mean
and variance. There is no Monte Carlo approximation to either label.
"""
import numpy as np


def oracle_score(table,nu):
    a,m,y=(np.asarray(table[k],float) for k in ('a','m','y'))
    g,q0,q1,u0,u1=(np.asarray(nu[k],float) for k in ('g','q0','q1','u0','u1'))
    if not all(np.isfinite(v).all() for v in (a,m,y,g,q0,q1,u0,u1)):raise ValueError('Nonfinite oracle teacher inputs')
    if not all(((v>0)&(v<1)).all() for v in (g,q0,q1)):raise ValueError('Oracle teacher is unclipped and requires positivity')
    e0=u0+q0*(u1-u0);e1=u0+q1*(u1-u0);um=np.where(m==1,u1,u0);ratio=np.where(m==1,q0/q1,(1-q0)/(1-q1))
    result=e1-e0;treated=a==1;control=~treated
    result[treated]+=((y[treated]-e1[treated])-ratio[treated]*(y[treated]-um[treated]))/g[treated]
    result[control]-=(um[control]-e0[control])/(1-g[control])
    return result


def exact_teacher_labels(scm):
    """V=E[(S-theta)^2], the variance coefficient; sampling Var(mean S)=V/n."""
    x=scm.population;nu=scm.nuisance(x);g,q0,q1=(np.asarray(nu[k],float) for k in ('g','q0','q1'));theta=float(np.mean((q1-q0)*(nu['mu11']-nu['mu10'])))
    conditional_mean=np.zeros(len(x));second=np.zeros(len(x));weight_total=np.zeros(len(x))
    for a in (0,1):
        qa=q1 if a else q0
        for m in (0,1):
            probability=(g if a else 1-g)*(qa if m else 1-qa)
            table=dict(a=np.full(len(x),a),m=np.full(len(x),m),y=nu['mu'+str(a)+str(m)])
            at_mean=oracle_score(table,nu);ratio=(q0/q1) if m else (1-q0)/(1-q1)
            coefficient=(1-ratio)/g if a else np.zeros(len(x))
            conditional_mean+=probability*at_mean
            second+=probability*((at_mean-theta)**2+coefficient**2*nu['conditional_variance'])
            weight_total+=probability
    target_x=(q1-q0)*(nu['mu11']-nu['mu10'])
    np.testing.assert_allclose(weight_total,1,atol=2e-14,rtol=0)
    np.testing.assert_allclose(conditional_mean,target_x,atol=2e-12,rtol=2e-12)
    variance=float(second.mean())
    if not np.isfinite(variance) or not variance>0:raise ValueError('Nonregular population EIF variance')
    return dict(theta=theta,variance_coefficient=variance,log_variance_coefficient=float(np.log(variance)),
        maximum_conditional_mean_identity_error=float(abs(conditional_mean-target_x).max()),
        population_hash=scm.population_hash,population_rows=len(x),method='Exact finite-X sum, four A/M cells, analytic conditional Y variance; unclipped oracle')


def table_teacher_labels(scm,table,mode='FSP_oracle_score',population_labels=None):
    labels=exact_teacher_labels(scm) if population_labels is None else population_labels
    if mode=='latent_theta':effect=labels['theta']
    elif mode=='FSP_oracle_score':effect=float(oracle_score(table,scm.nuisance(table['x'])).mean())
    else:raise ValueError('Unknown teacher label mode')
    return dict(effect_target=effect,log_variance_coefficient_target=labels['log_variance_coefficient'],theta=labels['theta'])
