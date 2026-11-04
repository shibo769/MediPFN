"""Final article §5 Simulation B, with paper/code discrepancy recorded."""
import numpy as np

def covariance(c):
    p=c['p']; r=c['rho_within']; b=c['rho_bridge']
    s=np.zeros((p,p),dtype=float)
    s[:2,:2]=r; np.fill_diagonal(s[:2,:2],1.)
    s[2:,2:]=r; np.fill_diagonal(s[2:,2:],1.)
    s[2:,:2]=b; s[:2,2:]=b
    return s

def generate(config,n,seed):
    if isinstance(n,bool) or not isinstance(n,int) or n<1: raise ValueError('n must be positive int')
    rng=np.random.default_rng(seed)
    p=config['p']; a=rng.binomial(1,config['p_treatment'],n)
    errors=rng.multivariate_normal(np.zeros(p),covariance(config),size=n,method='cholesky')
    m=errors.copy()
    for j in config['a_to_m_indices']: m[:,j-1]+=config['a_to_m_coefficient']*a
    y=config['direct_effect']*a.astype(float)
    for j in config['m_to_y_indices']: y+=config['m_to_y_coefficient']*m[:,j-1]
    y+=rng.normal(0,config['outcome_error_sd'],n)
    out={'A':a.tolist()}
    out.update({f'M{j:03d}':m[:,j-1].tolist() for j in range(1,p+1)})
    out['Y']=y.tolist()
    return out
