"""Section 5 sequential binary generator, independent of the source R package."""
import math
import random

def expit(x):
    if x >= 0: return 1/(1+math.exp(-x))
    v=math.exp(x); return v/(1+v)

def probabilities(c,w,a,z=None,m=None):
    w1,w2,w3=w; sw=sum(w)
    pa=expit(c["treatment_intercept"]+c["treatment_w_sum"]*sw+c["treatment_w1w2"]*w1*w2)
    pz=expit(c["z_intercept"]+c["z_w_sum"]*sw+c["z_a"]*a+c["z_aw3"]*a*w3)
    pm=None if z is None else expit(c["m_w1"]*w1+c["m_w2"]*w2+c["m_a"]*a+c["m_z"]*z+c["m_az"]*a*z+c["m_aw2"]*a*w2)
    py=None if m is None or z is None else expit((c["y_a"]*a+c["y_z"]*z+c["y_m"]*m+c["y_az"]*a*z)/(sw+c["y_denominator_intercept"]))
    return pa,pz,pm,py

def generate(config,n,seed):
    if isinstance(n,bool) or not isinstance(n,int) or n<1: raise ValueError("n must be positive int")
    rng=random.Random(seed)
    out={k:[] for k in ("W1","W2","W3","A","Z","M","Y")}
    for _ in range(n):
        w1=int(rng.random()<config["p_w1"])
        w2=int(rng.random()<config["p_w2"])
        w3=int(rng.random()<config["p_w3_intercept"]+config["p_w3_slope"]*(w1+w2))
        w=(w1,w2,w3)
        pa,_,_,_=probabilities(config,w,0)
        a=int(rng.random()<pa)
        _,pz,_,_=probabilities(config,w,a)
        z=int(rng.random()<pz)
        _,_,pm,_=probabilities(config,w,a,z)
        m=int(rng.random()<pm)
        _,_,_,py=probabilities(config,w,a,z,m)
        y=int(rng.random()<py)
        for k,v in zip(out,(*w,a,z,m,y)): out[k].append(v)
    return out
