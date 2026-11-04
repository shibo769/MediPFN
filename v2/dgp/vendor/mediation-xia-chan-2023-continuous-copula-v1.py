"""§4.1 continuous C,M; Gaussian copula realized by correlated normals."""
import math,random
def expit(z):
    if z>=0: return 1/(1+math.exp(-z))
    e=math.exp(z); return e/(1+e)
def generate(config,n,seed):
    if isinstance(n,bool) or not isinstance(n,int) or n<1: raise ValueError('n must be positive int')
    r=random.Random(seed); o={k:[] for k in ('X','A','C','M','Y')}
    for _ in range(n):
        x=r.gauss(0,config['x_sd']); a=int(r.random()<expit(config['a_logit_intercept']+config['a_logit_x']*x))
        em=r.gauss(0,1); ec=config['copula_rho']*em+math.sqrt(1-config['copula_rho']**2)*r.gauss(0,1)
        m=config['m_intercept']+config['m_a']*a+config['m_x']*x+config['m_sd']*em
        c=config['c_intercept']+config['c_a']*a+config['c_x']*x+config['c_sd']*ec
        y=(config['y_intercept']+config['y_a']*a+config['y_m']*m+config['y_c']*c+config['y_x']*x+config['y_ac']*a*c+config['y_am']*a*m+r.gauss(0,config['y_sd']))
        for k,v in zip(o,(x,a,c,m,y)): o[k].append(v)
    return o
