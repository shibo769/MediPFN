"""Real installed-checkpoint gradient/roundtrip fixture; not a benchmark.

Run only in a compute allocation with --config explicit local weights. It loads
two small, independent fictional contexts and verifies actual backbone updates.
It never reads the scientific OOD/test bank or submits a job.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from .frozen import BackboneConfig,official_factory,sha256
from .train_backbone import make_training_dataset,train_one_batch,save_checkpoint
from .error_head import NIEErrorHead,save_head,load_head
from .finetuned import make_finetuned_factory
from .run_state import initialize_output,allocation_threads


def main(c,out):
    torch.set_num_threads(allocation_threads());torch.manual_seed(154)
    r=np.random.default_rng(15);n=48
    x=r.normal(size=(n,4));a=np.tile([0,1],n//2);r.shuffle(a)
    y=.5*x[:,0]+.3*a+r.normal(size=n)
    classifier=official_factory(c,True,154);regressor=official_factory(c,False,155)
    for est in (classifier,regressor):est.set_params(inference_precision=torch.float32,fit_mode='batched')
    report={'kind':'fictional real checkpoint gradient fixture','config':asdict(c),'checks':{}}
    opts=[]
    for name,est,classification,target in [('classifier',classifier,True,a),('regressor',regressor,False,y)]:
        data=make_training_dataset(est,x,target,129)
        model=est.models_[0];model.to(c.device);model.train()
        for p in model.parameters():p.requires_grad_(True)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-6);opts.append(opt)
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        opt.zero_grad(set_to_none=True)
        loss,norm=train_one_batch(est,data,opt,classification,c.device)
        opt.step()
        changed=sum(int(not torch.equal(before[k],v)) for k,v in model.named_parameters())
        if changed<1:raise RuntimeError('No actual weight update: '+name)
        # A second independent dataset must preserve parameter object identities.
        ids=[id(p) for p in model.parameters()]
        second=make_training_dataset(est,x*.7,target[::-1].copy(),130)
        if ids!=[id(p) for p in est.models_[0].parameters()]:raise RuntimeError('Weights reinitialized between tasks')
        second_before={k:v.detach().clone() for k,v in model.named_parameters()}
        opt.zero_grad(set_to_none=True)
        second_loss,second_norm=train_one_batch(est,second,opt,classification,c.device)
        opt.step()
        second_changed=sum(int(not torch.equal(second_before[k],v)) for k,v in model.named_parameters())
        if second_changed<1:raise RuntimeError('Independent second task failed to update backbone')
        if {id(p) for group in opt.param_groups for p in group['params']}!={id(p) for p in model.parameters() if p.requires_grad}:
            raise RuntimeError('Optimizer parameter identity changed across contexts')
        report['checks'][name]=dict(loss=loss,gradient_norm=norm,changed_parameters=changed,
            second_loss=second_loss,second_gradient_norm=second_norm,second_changed_parameters=second_changed,
            independent_table_parameters_and_optimizer_retained=True)
    output=initialize_output(out,dict(fixture='actual legacy checkpoint gradients',config=asdict(c)),__file__)
    digest=save_checkpoint(output/'fixture_finetuned.pt',classifier,regressor,c,1,[],opts)
    checkpoint=torch.load(output/'fixture_finetuned.pt',map_location='cpu',weights_only=False)
    for key,est in [('classifier_state',classifier),('regressor_state',regressor)]:
        for name,value in est.models_[0].state_dict().items():
            if not torch.equal(value.detach().cpu(),checkpoint[key][name]):raise RuntimeError('Checkpoint roundtrip mismatch')
    report['checkpoint_sha256']=digest;report['checks']['backbone_checkpoint_roundtrip']=True
    frozen_factory=make_finetuned_factory(c,output/'fixture_finetuned.pt',digest)
    for classification,target,key in [(True,a,'classifier_state'),(False,y,'regressor_state')]:
        deployed=frozen_factory(c,classification,72);deployed.fit(x,target)
        for name,value in deployed.models_[0].named_parameters():
            if not torch.equal(value.detach().cpu(),checkpoint[key][name]):
                raise RuntimeError('Normal deployment fit reset learned weight: '+name)
    report['checks']['frozen_deployment_keeps_finetuned_weights']=True
    head=NIEErrorHead(24,width=32).to(c.device);features=torch.randn(8,24,device=c.device);target=torch.randn(8,device=c.device)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3);before=[p.detach().clone() for p in head.parameters()]
    loss=head.loss(features,target);loss.backward();opt.step()
    if not any(not torch.equal(a,b) for a,b in zip(before,head.parameters())):raise RuntimeError('Error head unchanged')
    manifest={'model_version':c.model_version,'classifier_sha256':c.classifier_sha256,'regressor_sha256':c.regressor_sha256}
    save_head(output/'fixture_error_head.pt',head,manifest,{'dimension':24},{'fixture':True})
    restored,_=load_head(output/'fixture_error_head.pt',manifest,map_location=c.device);restored=restored.to(c.device)
    with torch.no_grad():
        if not torch.allclose(head(features),restored(features)):raise RuntimeError('Error-head roundtrip mismatch')
    report['checks']['error_head_gradient_and_roundtrip']=True;report['status']='PASS'
    (output/'fixture_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(BackboneConfig(**json.loads(Path(a.config).read_text())),a.output)
