"""Cross-task nonlinear input/output adaptation of a frozen TabPFN-v3 backbone.

Adapters are trained offline on factual labels from separate DGP contexts.
Fresh wrappers share the frozen underlying networks by reference, allowing new
table dimensions without reusing another table's preprocessing or observations.
"""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .frozen import official_factory

FORMAT='medipfn-tabpfn-v3-cross-task-adapter-1'


def differentiable_query_embeddings(estimator,query,classification=False):
    """Mirror V3's embedding executor loop without converting adapted X to numpy.

    The public raw-data helper runs sklearn cleaning first. Here query already
    passed through the adapter and the executor uses differentiable torch-based
    preprocessing, whose learned means/stds remain tensors on the model device.
    """
    if not isinstance(query,torch.Tensor):raise TypeError('Differentiable embedding query must remain a torch Tensor')
    query=query.detach()
    rows=[]
    with torch.no_grad():
        for output,_ in estimator.executor_.iter_outputs(query,autocast=estimator.use_autocast_,
                    task_type='multiclass' if classification else 'regression',only_return_standard_out=False):
            value=output['test_embeddings']
            if not isinstance(value,torch.Tensor):raise TypeError('Backbone embedding must be a torch Tensor')
            value=value.squeeze(1)
            if value.ndim!=2 or value.shape[0]!=len(query):raise RuntimeError('Unexpected differentiable query embedding shape')
            if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite differentiable query embedding')
            rows.append(value.detach().cpu().numpy())
    if not rows:raise RuntimeError('No differentiable query embeddings returned')
    return np.stack(rows)


class ColumnAdapter(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.width=width
        self.map=nn.Sequential(nn.Linear(4,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.map[-1].weight);nn.init.zeros_(self.map[-1].bias)
        self.output_scale=nn.Parameter(torch.zeros(2));self.output_bias=nn.Parameter(torch.zeros(2))

    def transform(self,features,role):
        # Support-standardized observed columns only; no DGP identity/true parameter input.
        classification=int(role=='classification')
        z=features.clamp(-30,30)
        role_value=torch.full_like(z,float(classification))
        inp=torch.stack([z/5,(z/5).square().clamp(max=36),role_value,torch.ones_like(z)],-1)
        return z+.1*torch.tanh(self.map(inp).squeeze(-1))

    def adjust(self,prediction,role,mean=0.,scale=1.):
        index=int(role=='classification')
        slope=F.softplus(self.output_scale[index])/np.log(2.)
        if index:
            logits=torch.logit(prediction.clamp(1e-6,1-1e-6))
            return torch.sigmoid(slope*logits+self.output_bias[index])
        return mean+scale*(slope*(prediction-mean)/scale+self.output_bias[index])


class FrozenV3Backend:
    def __init__(self,config,seed=219):
        if config.model_version!='v3':raise ValueError('This adapter requires actual TabPFN-v3')
        self.config=config;self.seed=int(seed);self.base={}

    def _spec(self,classification,x,y):
        from tabpfn.base import ModelSpecs
        if classification not in self.base:
            base=official_factory(self.config,classification,self.seed+int(classification))
            base.set_params(inference_precision=torch.float32)
            # Initialization sees support observations only; query labels never enter.
            base.fit(x.detach().cpu().numpy(),y.detach().cpu().numpy())
            for model in base.models_:
                model.to(self.config.device);model.eval()
                for p in model.parameters():p.requires_grad_(False)
            self.base[classification]=base
        b=self.base[classification]
        return ModelSpecs(model=b.models_[0],architecture_config=b.configs_[0],
                          inference_config=b.inference_config_,
                          norm_criterion=None if classification else b.znorm_space_bardist_)

    def predict_tensor(self,adapter,x,y,q,role,return_model=False):
        from tabpfn import TabPFNClassifier,TabPFNRegressor
        classification=role=='classification'
        if x.ndim!=2 or q.ndim!=2 or x.shape[1]!=q.shape[1] or len(x)!=len(y):raise ValueError('Context shape mismatch')
        mean=x.detach().mean(0,keepdim=True);scale=x.detach().std(0,correction=0,keepdim=True).clamp_min(1e-3)
        sx=adapter.transform((x-mean)/scale,role);qx=adapter.transform((q-mean)/scale,role)
        spec=self._spec(classification,x,y)
        cls=TabPFNClassifier if classification else TabPFNRegressor
        est=cls(model_path=spec,device=self.config.device,n_estimators=1,
                n_preprocessing_jobs=1,random_state=self.seed,differentiable_input=True,
                inference_precision=torch.float32,fit_mode='fit_preprocessors',ignore_pretraining_limits=False)
        est.fit_with_differentiable_input(sx,y)
        if classification:
            p=est.forward(qx,use_inference_mode=True)
            if p.ndim!=2 or p.shape[1]!=2:raise RuntimeError('Unexpected differentiable classification output')
            raw=p[:,1];prediction=adapter.adjust(raw,role)
        else:
            logits,_,_=est.forward(qx,use_inference_mode=True)
            if logits is None:raise RuntimeError('Missing differentiable regression logits')
            raw=est.znorm_space_bardist_.mean(logits.transpose(0,1))*float(est.y_train_std_)+float(est.y_train_mean_)
            prediction=adapter.adjust(raw,role,float(est.y_train_mean_),float(est.y_train_std_))
        if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('Nonfinite adapted prediction')
        return (prediction,est,qx) if return_model else prediction


class AdaptedEstimator:
    """scikit-style wrapper for frozen deployment; fit() never invokes an optimizer."""
    def __init__(self,backend,adapter,classification):
        self.backend=backend;self.adapter=adapter;self.classification=classification;self.classes_=np.array([0,1])
    def fit(self,x,y):
        self.x=torch.as_tensor(x,dtype=torch.float32,device=self.backend.config.device)
        self.y=torch.as_tensor(y,dtype=torch.float32,device=self.backend.config.device)
        return self
    def _predict(self,q,model=False):
        with torch.no_grad():
            if self.x.shape[1]==0 or bool((self.x.std(0,correction=0)<1e-12).all()):
                if model:raise RuntimeError('No learned query embedding for constant features')
                role='classification' if self.classification else 'regression'
                return self.adapter.adjust(self.y.mean().expand(len(q)),role,
                           float(self.y.mean()),max(float(self.y.std(correction=0)),1e-3))
            return self.backend.predict_tensor(self.adapter,self.x,self.y,
                    torch.as_tensor(q,dtype=torch.float32,device=self.backend.config.device),
                    'classification' if self.classification else 'regression',return_model=model)
    def predict(self,q):return self._predict(q).detach().cpu().numpy()
    def predict_proba(self,q):
        p=self.predict(q);return np.column_stack([1-p,p])
    def get_embeddings(self,q,data_source='test'):
        if data_source!='test':raise ValueError('Only query embeddings are used')
        _,est,adapted_query=self._predict(q,model=True)
        # Error-head feature extraction is deliberately detached from adapter SGD.
        adapted_query=adapted_query.detach().to(device=self.backend.config.device,dtype=torch.float32)
        return differentiable_query_embeddings(est,adapted_query,self.classification)


def save_adapter(path,adapter,config,metadata):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(format=FORMAT,state_dict=adapter.state_dict(),width=adapter.width,
                    base_config=asdict(config),training_metadata=metadata),p)


def load_adapter(path,config,device=None):
    c=torch.load(path,map_location=device or config.device,weights_only=False)
    if c.get('format')!=FORMAT:raise ValueError('Wrong adapter format')
    for key in ('model_version','package_version','classifier_sha256','regressor_sha256'):
        if c['base_config'][key]!=getattr(config,key):raise ValueError('Different adapter base: '+key)
    a=ColumnAdapter(c['width']);a.load_state_dict(c['state_dict']);a.to(device or config.device);a.eval()
    return a,c


def adapted_factory(config,adapter,backend=None):
    backend=backend or FrozenV3Backend(config)
    adapter.eval()
    for p in adapter.parameters():p.requires_grad_(False)
    def factory(actual_config,classification,seed):
        if actual_config!=config:raise ValueError('Different backbone configuration')
        return AdaptedEstimator(backend,adapter,classification)
    factory.learns_constant_calibration=True
    return factory
