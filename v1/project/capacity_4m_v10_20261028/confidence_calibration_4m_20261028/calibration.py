"""Explicit interval-coverage confidence: isolated 13M mediation experiment.

Inspired by confidence elicitation, not a reproduction of verbal prompting.
All learnable decisions use disjoint training, development and calibration laws.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE=Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9')
XL=BASE/'capacity_4m_v10_20261028'
ROOT=XL/'confidence_calibration_4m_20261028'
sys.path.insert(0,str(BASE/'posttrain_v10_20261026'/'protocol'))
sys.path.insert(0,str(BASE/'posttrain_v10_20261026'/'tabcon'))
sys.path.insert(0,str(BASE/'code'))
import mixture
from posttrain import Cache,quantile
import evaluate_pairs as evaluation
from evaluate import EvaluationGuard,clean

SCALES=np.asarray([.4,.6,.8,1.,1.25,1.6,2.,2.5,3.5],dtype=np.float32)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def check_registration():
    r=json.loads((ROOT/'registration.json').read_text())
    assert sha(__file__)==r['code_sha256']
    assert sha(ROOT/'independent_test'/'protocol.json')==r['test_protocol_sha256']
    return r

class Confidence(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,x,s):
        v=self.net(x);location=F.softplus(v[...,0]);slope=.05+F.softplus(v[...,1]);return slope*(s-location)

@torch.no_grad()
def make_features(a,heads,router,mean,std,batch=256):
    """Return outputs based exclusively on observed table features and score."""
    n=len(a['estimates']);q=np.full((n,3),np.nan,dtype=np.float64)
    x=np.asarray(a['x_multi']); est=np.asarray(a['estimates'])[:,1];se=np.asarray(a['ses'])[:,1]
    valid=np.isfinite(x).all(1)&np.isfinite(est)&np.isfinite(se)&(se>0)
    if 'cache_failed' in a:valid &= ~np.asarray(a['cache_failed'],bool)
    idx=np.flatnonzero(valid)
    for part in np.array_split(idx,max(1,int(np.ceil(len(idx)/batch)))):
        if not len(part):continue
        z=torch.as_tensor(np.nan_to_num((x[part]-mean)/std),dtype=torch.float32)
        pred=mixture.mixture(mixture.parts_at(heads,z),router(z),True)
        q[part]=np.stack([quantile(pred,v).numpy() for v in (.025,.5,.975)],1)
    valid &= np.isfinite(q).all(1)&(q[:,0]<=q[:,1])&(q[:,1]<=q[:,2])
    # Features are observable, normalized on original head training, plus model interval geometry.
    f=np.concatenate((np.nan_to_num((x-mean)/std,nan=0.,posinf=0.,neginf=0.).clip(-8,8),
                      np.nan_to_num(np.stack((q[:,1]-q[:,0],q[:,2]-q[:,1]),1),nan=0.).clip(0,20)),1).astype(np.float32)
    return f,q,est,se,valid

def labels(q,est,se,truth):
    z=(est-truth)/se
    lo=q[:,1,None]+SCALES[None,:]*(q[:,0,None]-q[:,1,None])
    hi=q[:,1,None]+SCALES[None,:]*(q[:,2,None]-q[:,1,None])
    return ((lo<=z[:,None])&(z[:,None]<=hi)).astype(np.float32)

def masks(a):
    role=a['bank'].astype(str);split=a['split'].astype(str);laws=a['mechanism_id'].astype(str)
    train=(split=='train')&(role=='dpo');pool=(split=='train')&(role=='rlvr')
    # Independent deterministic law split, fixed before labels are inspected.
    available=sorted(set(laws[pool])); assert len(available)>=20
    dev_laws=set(available[::2]);cal_laws=set(available[1::2]);dev=pool&np.isin(laws,list(dev_laws));cal=pool&np.isin(laws,list(cal_laws))
    ignored=set(laws[(split=='train')&np.isin(role,['head','sft'])])|set(laws[split=='dev'])
    groups=[set(laws[m]) for m in (train,dev,cal)]
    assert all(len(g)>0 for g in groups)
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    assert all(not g&ignored for g in groups)
    assert not (split=='test').any()
    return train,dev,cal

def loss(net,x,y,ids):
    scales=torch.as_tensor(SCALES).view(1,-1)
    logits=net(x[ids,None,:].expand(-1,len(SCALES),-1),scales)
    return F.binary_cross_entropy_with_logits(logits,y[ids])

def fit():
    reg=check_registration();torch.set_num_threads(4);torch.manual_seed(reg['seed']);np.random.seed(reg['seed'])
    base=XL/'mixture_top2'/'mixture.pt';assert base.is_file()
    heads,routers,mean,std,meta=mixture.load(base,'cpu');assert meta['arguments']['top2']
    assert sha(meta['arguments']['cache'])==meta['cache_sha256']
    cache=Cache(XL/'fit.npz');a=cache.a;roles=masks(a)
    x,q,est,se,valid=make_features(a,heads,routers['sft'],mean,std)
    masks_good=[m&valid&cache.good for m in roles];train,dev,cal=[np.flatnonzero(m) for m in masks_good]
    assert min(map(len,(train,dev,cal)))>100
    y=labels(q,est,se,a['truth']);xt=torch.from_numpy(x);yt=torch.from_numpy(y)
    net=Confidence(x.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=0.0003,weight_decay=.01)
    best=float('inf');state=None;chosen=0;logs=[]
    for epoch in range(41):
        if epoch:
            for ids in torch.randperm(len(train)).split(256):
                ix=torch.as_tensor(train)[ids];objective=loss(net,xt,yt,ix)
                opt.zero_grad();objective.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        net.eval()
        with torch.no_grad(): v=float(loss(net,xt,yt,torch.as_tensor(dev)))
        logs.append(dict(epoch=epoch,dev_bce=v))
        if v<best:best=v;state={k:v.detach().clone() for k,v in net.state_dict().items()};chosen=epoch
        net.train()
    net.load_state_dict(state);net.eval()
    with torch.no_grad():
        sl=torch.as_tensor(SCALES).view(1,-1); raw=net(xt[cal,None,:].expand(-1,len(SCALES),-1),sl)
        temperatures=np.exp(np.linspace(-1.5,1.5,121));bce=[float(F.binary_cross_entropy_with_logits(raw/float(t),yt[cal])) for t in temperatures]
    temperature=float(temperatures[int(np.argmin(bce))])
    z=(est[cal]-a['truth'][cal])/se[cal];left=np.maximum(q[cal,1]-q[cal,0],1e-8);right=np.maximum(q[cal,2]-q[cal,1],1e-8)
    required=np.where(z<q[cal,1],(q[cal,1]-z)/left,(z-q[cal,1])/right)
    global_scale=float(np.quantile(required,.95,method='higher'))
    assert np.isfinite(global_scale) and global_scale>=0
    model={'state':state,'dim':x.shape[1],'temperature':temperature,'global_scale':global_scale,
           'base_sha256':sha(base),'fit_sha256':sha(XL/'fit.npz'),'code_sha256':sha(__file__),
           'registration_sha256':sha(ROOT/'registration.json'),'selected_epoch':chosen,'roles':{k:int(len(v)) for k,v in zip(('train','dev','calibration'),(train,dev,cal))},'seed':reg['seed']}
    out=ROOT/'fitted';out.mkdir(exist_ok=False)
    torch.save(model,out/'confidence.pt')
    (out/'fit_report.json').write_text(json.dumps(clean(dict(roles=model['roles'],epoch=chosen,dev_bce=best,temperature=temperature,global_scale=global_scale,logs=logs)),indent=2,allow_nan=False)+'\n')
    with (out/'freeze.json').open('x') as f:
        json.dump(dict(model_sha256=sha(out/'confidence.pt'),registration_sha256=sha(ROOT/'registration.json'),base_sha256=sha(base),time_utc=time.time()),f,indent=2)
    print('FITTED confidence',sha(out/'confidence.pt'),flush=True)

def scale_choice(net,x,valid,temp):
    out=np.full(len(x),float(SCALES[-1]));idx=np.flatnonzero(valid)
    with torch.no_grad():
        z=torch.from_numpy(x[idx]);sc=torch.as_tensor(SCALES).view(1,-1)
        prob=torch.sigmoid(net(z[:,None,:].expand(-1,len(SCALES),-1),sc)/temp).numpy()
    meets=prob>=.95;first=np.argmax(meets,axis=1);out[idx]=np.where(meets.any(1),SCALES[first],SCALES[-1])
    return out

def stats(est,se,q,sc,valid):
    low=est-se*(q[:,1]+sc*(q[:,2]-q[:,1]));high=est-se*(q[:,1]+sc*(q[:,0]-q[:,1]))
    a=np.full((len(est),6),np.nan);a[:,0]=est;a[:,1]=se;a[:,2]=low;a[:,3]=high;a[:,4]=0;a[:,5]=(~valid).astype(float)
    return a

def evaluate():
    check_registration();torch.set_num_threads(4)
    fitted=ROOT/'fitted'/'confidence.pt';model=torch.load(fitted,map_location='cpu',weights_only=False)
    freeze=json.loads((ROOT/'fitted'/'freeze.json').read_text());assert sha(fitted)==freeze['model_sha256']
    assert sha(__file__)==model['code_sha256'] and sha(ROOT/'registration.json')==model['registration_sha256']
    test=ROOT/'independent_test';gate=json.loads((test/'audit'/'test_authorization.json').read_text())
    EvaluationGuard(test/'protocol.json',test/'audit'/'test_authorization.json').check('test')
    assert gate['confidence_model_sha256']==sha(fitted)
    assert sha(gate['base_mixture_path'])==model['base_sha256']==gate['base_mixture_sha256']
    a,manifest=evaluation.read_manifest(test/'cache'/'xl'/'manifest.json',test/'protocol.json','test')
    assert manifest['checkpoint_sha256']==gate['checkpoint_sha256']
    heads,routers,mean,std,meta=mixture.load(XL/'mixture_top2'/'mixture.pt','cpu')
    x,q,est,se,valid=make_features(a,heads,routers['sft'],mean,std)
    net=Confidence(model['dim']);net.load_state_dict(model['state']);net.eval()
    predictions={'original_sft':stats(est,se,q,np.ones(len(est)),valid),
      'global_calibration':stats(est,se,q,np.full(len(est),model['global_scale']),valid),
      'confidence_raw':stats(est,se,q,scale_choice(net,x,valid,1.),valid),
      'confidence_temperature':stats(est,se,q,scale_choice(net,x,valid,model['temperature']),valid)}
    obs={k:a[k] for k in ('x_multi','estimates','ses','cache_failed')}
    reference=evaluation.predict(obs,heads,routers,mean,std,True,device='cpu')['sft']
    assert np.array_equal(predictions['original_sft'][:,5],reference[:,5])
    # Invalid intervals use different sentinel values; compare endpoints only on valid rows.
    assert np.allclose(predictions['original_sft'][valid,2:4],reference[valid,2:4],rtol=1e-5,atol=1e-7)
    for pred in predictions.values():pred[:,0]=reference[:,0]
    f=a['family'].astype(str);groups={'ID':np.isin(f,['linear','smooth','heterosked','weak_overlap','covariate_shift']),
      'selection_shift':np.isin(f,['tree','rff']),'blind_shift':np.isin(f,['mlp','combinations'])}
    rows=[];paired=[]
    for g,m in groups.items():
        ids=np.flatnonzero(m)
        for name,pred in predictions.items():
            rows.append(dict(group=g,method=name,**evaluation.metric(pred,a['truth'],ids)))
            paired.append(dict(group=g,method=name,**evaluation.paired_coverage(predictions['original_sft'],pred,a['truth'],ids,a['mechanism_id'].astype(str))))
    # Additional decision latency only: feature and base-mixture quantile computation excluded.
    ids=np.flatnonzero(valid)[:13];assert len(ids)==13;times=[]
    for count,i in enumerate(ids):
        t=time.perf_counter();scale_choice(net,x[i:i+1],valid[i:i+1],model['temperature']);delta=time.perf_counter()-t
        if count>=3:times.append(delta)
    out=ROOT/'test';out.mkdir(exist_ok=False)
    np.savez_compressed(out/'per_table.npz',**predictions,truth=a['truth'],family=a['family'],mechanism_id=a['mechanism_id'])
    report={'registration_sha256':sha(ROOT/'registration.json'),'fitted_sha256':sha(fitted),'models':rows,'paired_original_minus_method':paired,
       'no_finite_sample_ood_guarantee':True,'valid_rows':int(valid.sum()),'failures':int((~valid).sum()),
       'additional_cpu_decision_seconds':float(np.median(times)),
       'timing_scope':'median over 10 single tables following 3 warmups; classifier/scale only, excludes feature construction, all quantiles, base model, and I/O; not end-to-end latency'}
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    print('EVALUATION COMPLETE',len(est),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('fit','evaluate'));a=p.parse_args();{'fit':fit,'evaluate':evaluate}[a.action]()
