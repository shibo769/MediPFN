"""Actual V3 checkpoint/API and adapter-gradient fixture, inside compute allocation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .adapter import ColumnAdapter,FrozenV3Backend,save_adapter,load_adapter,adapted_factory
from .frozen import BackboneConfig,official_factory,estimate_table,sha256,make_shared_frozen_v3_factory
from .run_state import initialize_output,allocation_threads
from .test_embedding_contract import EmbeddingContract
import unittest


def main(c,output):
    if c.model_version!='v3':raise ValueError('Actual V3 required')
    torch.manual_seed(193);torch.set_num_threads(allocation_threads())
    out=initialize_output(output,dict(fixture='actual V3 gradients and NIE',config=asdict(c)),__file__)
    r=np.random.default_rng(193);report=dict(kind='actual V3 fictional fixture',config=asdict(c),checks={})
    contracts=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EmbeddingContract))
    if not contracts.wasSuccessful():raise RuntimeError('Differentiable embedding Tensor regression contract failed')
    report['checks']['embedding_tensor_contract']=dict(tests=contracts.testsRun,passed=True,input_dtype='torch.float32',output_dtype='float32')
    shared=make_shared_frozen_v3_factory(c)
    comparisons=[]
    for classification in (True,False):
        for j,p in enumerate((4,7)):
            sx=r.normal(size=(32,p));qx=r.normal(size=(8,p))
            sy=np.tile([0,1],16) if classification else r.normal(size=32)
            direct=official_factory(c,classification,193+j);direct.fit(sx,sy)
            cached=shared(c,classification,193+j);cached.fit(sx,sy)
            a=direct.predict_proba(qx) if classification else direct.predict(qx)
            b=cached.predict_proba(qx) if classification else cached.predict(qx)
            if not np.allclose(a,b,rtol=1e-5,atol=1e-6):raise RuntimeError('Shared frozen factory changed reference predictions')
            comparisons.append(dict(classification=classification,columns=p,max_difference=float(np.max(abs(a-b)))))
    if shared.counts['checkpoint_initializations']!=2 or shared.counts['fresh_context_fits']!=4:
        raise RuntimeError('Shared factory did not initialize exactly two frozen networks')
    report['checks']['shared_factory_reference_agreement']=dict(comparisons=comparisons,counts=dict(shared.counts))
    backend=FrozenV3Backend(c,seed=193);adapter=ColumnAdapter().to(c.device)
    opt=torch.optim.AdamW(adapter.parameters(),lr=1e-3)
    for classification in (True,False):
        role='classification' if classification else 'regression'
        parameter_ids={id(p) for p in adapter.parameters()}
        rows=[]
        for j,p in enumerate((4,7)):
            x=torch.tensor(r.normal(size=(32,p)),device=c.device,dtype=torch.float32)
            q=torch.tensor(r.normal(size=(8,p)),device=c.device,dtype=torch.float32)
            y=torch.tensor(np.tile([0,1],16) if classification else r.normal(size=32),device=c.device,dtype=torch.float32)
            target=torch.tensor(np.tile([0,1],4) if classification else r.normal(size=8),device=c.device,dtype=torch.float32)
            opt.zero_grad(set_to_none=True);before={k:v.detach().clone() for k,v in adapter.named_parameters()}
            pred=backend.predict_tensor(adapter,x,y,q,role)
            if not pred.requires_grad:raise RuntimeError('Forward blocked required adapter gradient path')
            bases={k:v.detach().clone() for k,v in backend.base[classification].models_[0].named_parameters()}
            loss=torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6,1-1e-6),target) if classification else (pred-target).square().mean()
            loss.backward()
            map_norm=float(sum(p.grad.detach().square().sum() for p in adapter.map.parameters() if p.grad is not None).sqrt())
            if map_norm<=0 or not np.isfinite(map_norm):raise RuntimeError('No gradient through actual frozen V3 into nonlinear adapter')
            if any(p.grad is not None for p in backend.base[classification].models_[0].parameters()):raise RuntimeError('Frozen V3 has gradients')
            opt.step()
            changed=sum(int(not torch.equal(before[k],v)) for k,v in adapter.named_parameters())
            if changed<1 or parameter_ids!={id(p) for p in adapter.parameters()}:raise RuntimeError('Adapter did not update persistently')
            if any(not torch.equal(bases[k],v) for k,v in backend.base[classification].models_[0].named_parameters()):raise RuntimeError('Frozen backbone weight changed')
            rows.append(dict(columns=p,loss=float(loss.detach()),adapter_map_gradient_norm=map_norm,changed_parameters=changed))
        report['checks'][role]=rows
    save_adapter(out/'fixture_adapter.pt',adapter,c,dict(fixture=True,independent_contexts=4))
    restored,_=load_adapter(out/'fixture_adapter.pt',c)
    with torch.no_grad():
        a=backend.predict_tensor(adapter,x,y,q,'regression');b=backend.predict_tensor(restored,x,y,q,'regression')
        if not torch.allclose(a,b,rtol=1e-5,atol=1e-6):raise RuntimeError('Adapter prediction roundtrip mismatch')
    report['checks']['adapter_checkpoint_roundtrip']=True
    report['adapter_sha256']=sha256(out/'fixture_adapter.pt')
    # Full vector-mediator score and pooled-embedding contract after freezing.
    from dgp import sample_task
    t=sample_task('jones_correlated',64,201,randomize=False,compute_truth=False)
    factory=adapted_factory(c,restored,backend)
    result=estimate_table(t,c,seed=211,factory=factory,collect_embeddings=True,return_scores=True)
    embedding=np.asarray(result['pooled_context'])
    if embedding.ndim!=1 or len(embedding)<2 or not np.isfinite(embedding).all():raise RuntimeError('Invalid pooled V3 embedding')
    if not np.isfinite(result['estimate']) or len(result['scores'])!=64:raise RuntimeError('Invalid vector NIE fixture')
    report['checks']['no_X_vector_NIE_and_embeddings']=dict(n=64,p=0,q=100,
                embedding_dimension=len(embedding),successful_scores=len(result['scores']),finite=True,
                embedding_input_dtype='torch.float32',raw_embedding_output_dtype='float32',
                pooled_embedding_dtype=str(embedding.dtype))
    frozen=estimate_table(t,c,seed=211,factory=shared,collect_embeddings=True,return_scores=True)
    if len(frozen['pooled_context'])!=len(embedding) or len(frozen['scores'])!=64:raise RuntimeError('Frozen V3 embedding/NIE contract differs')
    if shared.counts['checkpoint_initializations']!=2:raise RuntimeError('Frozen networks reloaded per nuisance fit')
    report['checks']['shared_frozen_vector_NIE']=dict(n=64,p=0,q=100,successful_scores=64,
                     embedding_dimension=len(frozen['pooled_context']),
                     pooled_embedding_dtype=str(np.asarray(frozen['pooled_context']).dtype),counts=dict(shared.counts))
    report['status']='PASS';(out/'fixture_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(BackboneConfig(**json.loads(Path(a.config).read_text())),a.output)
