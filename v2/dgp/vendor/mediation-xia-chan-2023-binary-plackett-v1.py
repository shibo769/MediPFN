"""Exact Bernoulli pair from Plackett cross-ratio and Bernoulli margins."""
import random,math
def expit(z):
    if z>=0:return 1/(1+math.exp(-z))
    e=math.exp(z);return e/(1+e)
def margins(c,a,x):
    pm=expit(c['m_logit_intercept']+c['m_logit_a']*a+c['m_logit_x']*x)
    pc=expit(c['c_logit_intercept']+c['c_logit_a']*a+c['c_logit_x']*x)
    theta=math.exp(c['copula_log_or_intercept']+c['copula_log_or_a']*a+c['copula_log_or_x']*x)
    return pm,pc,theta
def joint11(pm,pc,theta):
    lo=max(0.,pm+pc-1.);hi=min(pm,pc)
    if abs(theta-1)<1e-12:return pm*pc
    for _ in range(70):
        q=(lo+hi)/2
        # f(q)=p11*p00-theta*p10*p01 is increasing through the admissible interval.
        f=q*(1-pm-pc+q)-theta*(pm-q)*(pc-q)
        if f<0:lo=q
        else:hi=q
    return (lo+hi)/2
def generate(config,n,seed):
    if isinstance(n,bool) or not isinstance(n,int) or n<1:raise ValueError('n must be positive int')
    r=random.Random(seed);o={k:[] for k in ('X','A','C','M','Y')}
    for _ in range(n):
        x=r.gauss(0,config['x_sd']);a=int(r.random()<expit(config['a_logit_intercept']+config['a_logit_x']*x))
        pm,pc,theta=margins(config,a,x);q=joint11(pm,pc,theta);v=r.random()
        if v<q:m=1;cc=1
        elif v<pm:m=1;cc=0
        elif v<pm+pc-q:m=0;cc=1
        else:m=0;cc=0
        y=config['y_intercept']+config['y_a']*a+config['y_m']*m+config['y_c']*cc+config['y_x']*x+config['y_ac']*a*cc+config['y_am']*a*m+r.gauss(0,config['y_sd'])
        for k,val in zip(o,(x,a,cc,m,y)):o[k].append(val)
    return o
