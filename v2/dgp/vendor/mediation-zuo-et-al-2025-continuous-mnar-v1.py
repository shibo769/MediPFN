"""Setting C.III complete-data NPSEM followed by MNAR observation indicators."""
import math
import random

def expit(v):
    if v>=0: return 1/(1+math.exp(-v))
    e=math.exp(v); return e/(1+e)

def generate(config,n,seed):
    if isinstance(n,bool) or not isinstance(n,int) or n<1: raise ValueError("n must be positive integer")
    r=random.Random(seed)
    out={k:[] for k in ("X","T","R_M","M_obs0","R_Y","Y_obs0")}
    for _ in range(n):
        x=r.gauss(config["x_mean"],config["x_sd"])
        t=int(r.random()<config["p_t"])
        m=r.gauss(config["alpha_0"]+config["alpha_t"]*t+config["alpha_x"]*x,config["m_sd"])
        y=r.gauss(config["beta_0"]+config["beta_m"]*m+config["beta_t"]*t+config["beta_mt"]*m*t+config["beta_x"]*x,config["y_sd"])
        rm=int(r.random()<expit(config["lambda_0"]+config["lambda_m"]*m+config["lambda_t"]*t+config["lambda_x"]*x))
        ry=int(r.random()<expit(config["gamma_0"]+config["gamma_y"]*y+config["gamma_t"]*t+config["gamma_x"]*x))
        for k,v in zip(out,(x,t,rm,m if rm else 0.,ry,y if ry else 0.)): out[k].append(v)
    return out
