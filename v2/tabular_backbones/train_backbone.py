"""Optional genuine cross-DGP TabPFN-v2 fine-tuning via its pinned official API.

Each optimizer item is ONE independent table/context, never a concatenation of
unrelated DGP observations. Weights are updated only during this offline program;
deployment uses frozen weights and table-specific in-context fitting only.
No job is submitted by this module. Not an implementation of original TabCon.
"""
from dataclasses import asdict
from functools import partial
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .frozen import BackboneConfig,official_factory,sha256,_x
from .run_state import initialize_output,allocation_threads

FORMAT='medipfn-tabpfn-v2-cross-task-finetune-1'


def make_training_dataset(estimator,x,y,seed):
    from sklearn.model_selection import train_test_split
    # The lists explicitly preserve task boundaries; batch_size=1 is mandated by
    # this official TabPFN release's meta_dataset_collator.
    splitter=partial(train_test_split,test_size=.35,random_state=int(seed))
    data=estimator.get_preprocessed_datasets([x],[y],splitter,max_data_size=None)
    if len(data)!=1:raise RuntimeError('One task must remain exactly one preprocessed dataset')
    return data


def train_one_batch(estimator,data,optimizer,classification,device,scale=1.):
    import torch
    from tabpfn.utils import meta_dataset_collator
    batch=meta_dataset_collator([data[0]])
    if classification:
        # Official ClassifierDatasetConfig yields six collated components.
        xtr,xq,ytr,yq,cat,confs=batch[:6]
        estimator.fit_from_preprocessed(xtr,ytr,cat,confs,no_refit=True)
        logits=estimator.forward(xq,return_logits=True)
        if logits.ndim!=3:raise RuntimeError('Unexpected classifier fine-tuning logit shape')
        loss=torch.nn.functional.cross_entropy(logits,yq.to(device).long())
    else:
        if len(batch)!=10:raise RuntimeError('Unsupported pinned regressor batch schema')
        xtr,xq,ytr,yq,cat,confs,raw_bar,zbar,_,_=batch
        estimator.raw_space_bardist_=raw_bar[0]
        estimator.znorm_space_bardist_=zbar[0]
        estimator.fit_from_preprocessed(xtr,ytr,cat,confs,no_refit=True)
        logits,_,_=estimator.forward(xq)
        loss=zbar[0](logits,yq.to(device)).mean()
    actual_ids={id(p) for p in estimator.models_[0].parameters() if p.requires_grad}
    optimizer_ids={id(p) for group in optimizer.param_groups for p in group['params']}
    if actual_ids!=optimizer_ids:raise RuntimeError('Forward changed optimizer parameter identities')
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite fine-tuning loss')
    (loss/scale).backward()
    params=[p for p in estimator.models_[0].parameters() if p.requires_grad]
    finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
    norm=float(torch.sqrt(sum((p.grad.detach().float().square().sum() for p in params if p.grad is not None),torch.tensor(0.,device=device))))
    if not finite or norm<=0:raise RuntimeError('No valid backbone gradient')
    return float(loss.detach()),norm


def save_checkpoint(path,classifier,regressor,config,steps,log,optimizers=None,selection=None):
    import torch
    payload={'format':FORMAT,'base_config':asdict(config),'steps':int(steps),
             'classifier_state':classifier.models_[0].state_dict(),
             'regressor_state':regressor.models_[0].state_dict(),
             'training_log':log,'deployment_requires_grad':False,
             'context_policy':'independent DGP tables; no per-test gradient updates'}
    if selection is not None:payload['selection']=selection
    if optimizers:payload['optimizer_states']=[o.state_dict() for o in optimizers]
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    torch.save(payload,tmp);tmp.replace(p)
    return sha256(p)


def fixed_dev_tasks(n,seed):
    from dgp import available_sources,registry,sample_task
    rows=registry();ids=set(available_sources('train'))
    groups=sorted(set(v['source_group'] for v in rows if v['source_id'] in ids))
    tasks=[]
    for j,group in enumerate(groups):
        candidates=[v['source_id'] for v in rows if v['source_id'] in ids and v['source_group']==group]
        for k in range(2):
            source=candidates[k%len(candidates)]
            tasks.append(sample_task(source,n,seed+1000+10*j+k,mechanism_seed=seed+2000+10*j+k,compute_truth=False))
    return tasks


def dev_loss(classifier,regressor,config,tasks):
    from tabpfn import TabPFNClassifier,TabPFNRegressor
    from tabpfn.finetune_utils import clone_model_for_evaluation
    args=dict(device=config.device,n_estimators=config.n_estimators,
              n_preprocessing_jobs=1,ignore_pretraining_limits=False)
    records=[]
    for j,t in enumerate(tasks):
        a=np.asarray(t['a']);y=np.asarray(t['y']);x=t['x'];m=t['m']
        r=np.random.default_rng(910000+j);train=[];query=[]
        for arm in (0,1):
            ix=np.flatnonzero(a==arm);r.shuffle(ix);cut=max(1,len(ix)//2)
            train.extend(ix[:cut]);query.extend(ix[cut:])
        tr=np.array(train);qu=np.array(query)
        losses=[]
        for features in (_x(x),np.column_stack([x,m])):
            if np.all(np.std(features[tr],axis=0)<1e-12):probs=np.full(len(qu),a[tr].mean())
            else:
                est=clone_model_for_evaluation(classifier,dict(args,random_state=110+j),TabPFNClassifier)
                est.fit(features[tr],a[tr]);raw=est.predict_proba(features[qu]);classes=np.asarray(est.classes_)
                probs=raw[:,int(np.flatnonzero(classes==1)[0])]
                del est
            p=np.clip(probs,1e-6,1-1e-6);losses.append(float(np.mean(-a[qu]*np.log(p)-(1-a[qu])*np.log(1-p))))
        est=clone_model_for_evaluation(regressor,dict(args,random_state=710+j),TabPFNRegressor)
        features=np.column_stack([x,a,m]);est.fit(features[tr],y[tr]);pred=est.predict(features[qu]);del est
        scale=max(float(y[tr].std()),1e-3)
        mse=float(np.mean(((pred-y[qu])/scale)**2))
        records.append(dict(source=t['metadata']['source_id'],family=t['metadata']['source_group'],
                            classifier_nll=float(np.mean(losses)),normalized_outcome_mse=mse,
                            combined=float(np.mean(losses))+mse))
    value=float(np.mean([r['combined'] for r in records]))
    if not np.isfinite(value):raise FloatingPointError('Nonfinite DEV loss')
    return value,records


def run(config,output,steps=2000,seed=271831,n=256,lr=1e-6,accumulation=4,dev_every=250):
    import torch
    from dgp import sample_random_task,available_sources
    if config.model_version!='v2' or config.package_version!='6.0.6':
        raise ValueError('This training implementation is pinned to actual installed v2/6.0.6 API')
    if steps<1 or accumulation<1:raise ValueError('Invalid training budget')
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(allocation_threads())
    classifier=official_factory(config,True,seed)
    regressor=official_factory(config,False,seed+1)
    for est in (classifier,regressor):est.set_params(inference_precision=torch.float32,fit_mode='batched')
    # Preprocessing initializes each official model once; later task calls keep it.
    first=sample_random_task(n,seed,split='train',compute_truth=False)
    make_training_dataset(classifier,_x(first['x']),first['a'],seed)
    make_training_dataset(regressor,np.column_stack([first['x'],first['a'],first['m']]),first['y'],seed)
    for est in (classifier,regressor):
        if len(est.models_)!=1:raise RuntimeError('Fine-tuning requires a single underlying model')
        est.models_[0].to(config.device);est.models_[0].train()
        for p in est.models_[0].parameters():p.requires_grad_(True)
    optimizers=[torch.optim.AdamW(est.models_[0].parameters(),lr=lr,weight_decay=1e-4) for est in (classifier,regressor)]
    for opt in optimizers:opt.zero_grad(set_to_none=True)
    records=[];started=time.perf_counter()
    out=initialize_output(output,dict(method='legacy V2 cross-task backbone fine-tune',base_config=asdict(config),
            steps=steps,n=n,seed=seed,accumulation=accumulation,lr=lr,dev_every=dev_every),__file__)
    dev=fixed_dev_tasks(n,seed+900000000)
    dev_ids={t['mechanism_id'] for t in dev};best=float('inf');best_step=0;dev_record=[]
    for step in range(1,steps+1):
        csum=rsum=0.
        for mini in range(accumulation):
            task_seed=seed+step*1009+mini
            t=sample_random_task(n,task_seed,split='train',compute_truth=False)
            if t['metadata']['source_id'] not in available_sources('train'):raise RuntimeError('Source leakage')
            if t['mechanism_id'] in dev_ids:raise RuntimeError('Training/DEV mechanism collision')
            # Alternate pi(X) and g(X,M) factual classification contexts.
            cx=_x(t['x']) if t['x'].shape[1]>0 and (step+mini)%2==0 else np.column_stack([t['x'],t['m']])
            cd=make_training_dataset(classifier,cx,t['a'],task_seed)
            cl,cnorm=train_one_batch(classifier,cd,optimizers[0],True,config.device,accumulation)
            rd=make_training_dataset(regressor,np.column_stack([t['x'],t['a'],t['m']]),t['y'],task_seed+1)
            rl,rnorm=train_one_batch(regressor,rd,optimizers[1],False,config.device,accumulation)
            csum+=cl/accumulation;rsum+=rl/accumulation
        for est,opt in zip((classifier,regressor),optimizers):
            torch.nn.utils.clip_grad_norm_(est.models_[0].parameters(),1.)
            opt.step();opt.zero_grad(set_to_none=True)
        rec=dict(step=step,classifier_loss=csum,regressor_loss=rsum,
                 classifier_gradient_norm=cnorm,regressor_gradient_norm=rnorm,
                 seconds=time.perf_counter()-started)
        records.append(rec)
        if step==steps or step%dev_every==0:
            value,details=dev_loss(classifier,regressor,config,dev)
            dev_record.append(dict(step=step,value=value,tasks=details))
            if value<best:
                best=value;best_step=step
                save_checkpoint(out/'best.pt',classifier,regressor,config,step,records,selection=dict(criterion='independent family-balanced factual DEV',value=best,step=step))
            (out/'dev_selection.json').write_text(json.dumps(dict(best_value=best,best_step=best_step,
                 dev_seed=seed+900000000,dev_mechanism_ids=sorted(dev_ids),history=dev_record),indent=2)+'\n')
            for est in (classifier,regressor):est.models_[0].train()
        if step==1 or step%10==0:print(json.dumps(rec),flush=True)
        if step==steps or step%100==0:
            digest=save_checkpoint(out/'last.pt',classifier,regressor,config,step,records,optimizers)
            (out/'checkpoint_manifest.json').write_text(json.dumps({'sha256':digest,'steps':step,'format':FORMAT},indent=2)+'\n')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=271831)
    p.add_argument('--n',type=int,default=256);p.add_argument('--lr',type=float,default=1e-6);p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--dev-every',type=int,default=250)
    a=p.parse_args();c=BackboneConfig(**json.loads(Path(a.config).read_text()))
    run(c,a.output,a.steps,a.seed,a.n,a.lr,a.accumulation,a.dev_every)
