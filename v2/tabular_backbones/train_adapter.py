"""Offline cross-DGP mediation adaptation of real frozen TabPFN-v3 weights."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .adapter import ColumnAdapter,FrozenV3Backend,save_adapter
from .frozen import BackboneConfig,_x
from .train_backbone import fixed_dev_tasks
from .run_state import initialize_output,allocation_threads
from dataclasses import asdict


def context_loss(backend,adapter,t,seed,kind):
    a=np.asarray(t['a']);r=np.random.default_rng(int(seed));si=[];qi=[]
    for arm in (0,1):
        ix=np.flatnonzero(a==arm);r.shuffle(ix)
        if len(ix)<4:raise ValueError('Insufficient factual treatment support')
        cut=max(2,int(.65*len(ix)));si.extend(ix[:cut]);qi.extend(ix[cut:])
    si=np.asarray(si);qi=np.asarray(qi)
    x=t['x'];m=t['m'];y=t['y'];device=backend.config.device
    if kind=='pi' and x.shape[1]==0:
        # With no X, pi is constant; retain its own objective and let only the
        # shared classification output calibration receive gradients.
        support=torch.tensor(a[si],device=device,dtype=torch.float32)
        target=torch.tensor(a[qi],device=device,dtype=torch.float32)
        raw=support.mean().expand(len(qi))
        pred=adapter.adjust(raw,'classification')
        return torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6,1-1e-6),target)
    features=_x(x) if kind=='pi' else np.column_stack([x,m]) if kind=='g' else np.column_stack([x,a,m])
    target=a if kind in ('pi','g') else y
    role='classification' if kind in ('pi','g') else 'regression'
    sx=torch.as_tensor(features[si],device=device,dtype=torch.float32)
    sy=torch.as_tensor(target[si],device=device,dtype=torch.float32)
    qx=torch.as_tensor(features[qi],device=device,dtype=torch.float32)
    qy=torch.as_tensor(target[qi],device=device,dtype=torch.float32)
    pred=backend.predict_tensor(adapter,sx,sy,qx,role)
    if role=='classification':return torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6,1-1e-6),qy)
    scale=sy.std(correction=0).clamp_min(1e-3)
    return ((pred-qy)/scale).square().mean()


def dev_loss(backend,adapter,tasks,seed):
    rows=[];adapter.eval()
    with torch.no_grad():
        for j,t in enumerate(tasks):
            values=[float(context_loss(backend,adapter,t,seed+j,kind)) for kind in ('pi','g','mu')]
            rows.append(dict(source=t['metadata']['source_id'],family=t['source_group'],
                             pi_nll=values[0],g_nll=values[1],normalized_mu_mse=values[2],combined=float(np.mean(values))))
    value=float(np.mean([v['combined'] for v in rows]))
    if not np.isfinite(value):raise FloatingPointError('Nonfinite adapter DEV loss')
    return value,rows


def run(config,output,steps=2000,n=128,seed=92061103,accumulation=4,lr=1e-4,dev_every=250):
    from dgp import sample_random_task,available_sources
    torch.manual_seed(seed);torch.set_num_threads(allocation_threads())
    backend=FrozenV3Backend(config,seed=seed)
    adapter=ColumnAdapter().to(config.device)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=lr,weight_decay=1e-4)
    dev=fixed_dev_tasks(n,seed+800000000);dev_ids={t['mechanism_id'] for t in dev}
    best=float('inf');best_step=0;history=[]
    out=initialize_output(output,dict(method='V3 frozen-backbone cross-task adapter',base_config=asdict(config),
             steps=steps,n=n,seed=seed,accumulation=accumulation,lr=lr,dev_every=dev_every,
             objectives={'pi':1/3,'g':1/3,'mu':1/3},no_X_pi='empirical support propensity with output calibration'),__file__)
    started=time.perf_counter()
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True);adapter.train();loss_sum=0.
        for mini in range(accumulation):
            t=sample_random_task(n,seed+step*1009+mini,split='train',compute_truth=False)
            if t['metadata']['source_id'] not in available_sources('train') or t['mechanism_id'] in dev_ids:raise RuntimeError('Split leakage')
            kind=('pi','g','mu')[(step+mini)%3]
            loss=context_loss(backend,adapter,t,seed+step+mini,kind)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite adapter loss')
            (loss/accumulation).backward();loss_sum+=float(loss.detach())/accumulation
        norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.))
        if not np.isfinite(norm) or norm<=0:raise RuntimeError('Missing adapter gradient')
        for base in backend.base.values():
            if any(p.grad is not None for m in base.models_ for p in m.parameters()):raise RuntimeError('Frozen backbone received gradients')
        optimizer.step()
        rec=dict(step=step,loss=loss_sum,gradient_norm=norm,seconds=time.perf_counter()-started)
        if step==steps or step%dev_every==0:
            value,details=dev_loss(backend,adapter,dev,seed+90000000)
            rec['dev']=value
            if value<best:
                best=value;best_step=step
                save_adapter(out/'best.pt',adapter,config,dict(selection='independent family-balanced factual DEV',
                             dev_value=best,step=step,dev_mechanism_ids=sorted(dev_ids)))
            (out/'dev_latest.json').write_text(json.dumps(dict(step=step,value=value,tasks=details,best_step=best_step,best=best),indent=2)+'\n')
        history.append(rec)
        if step==1 or step%10==0:print(json.dumps(rec),flush=True)
        if step==steps or step%100==0:
            save_adapter(out/'last.pt',adapter,config,dict(step=step,seed=seed,history=history,
                         no_test_time_sgd=True,independent_contexts=True))
    return history


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--n',type=int,default=128);p.add_argument('--seed',type=int,default=92061103)
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--dev-every',type=int,default=250)
    a=p.parse_args();c=BackboneConfig(**json.loads(Path(a.config).read_text()));run(c,a.output,a.steps,a.n,a.seed,a.accumulation,a.lr,a.dev_every)
