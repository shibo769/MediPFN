"""Ordered NIE-error head and confidence/group-balanced post-training geometry."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from common import MODEL_FORMAT,VARIANTS,PAIR_INDICES,TAUS,torch_save_new,sha256

SCALE_GRID=(.4,.6,.8,1.,1.25,1.6,2.,2.5,3.5)

class NIEErrorHead(nn.Module):
    """Exact architecture of the existing 335,394-parameter V3 error head."""
    def __init__(self,input_dim=1038,width=256):
        super().__init__();self.input_dim=input_dim;self.width=width
        self.network=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,width),nn.GELU(),
          nn.Linear(width,width),nn.GELU(),nn.Linear(width,6))
    def forward(self,features):
        raw=self.network(features)
        return torch.cat((raw[:,:1],raw[:,:1]+(F.softplus(raw[:,1:])+1e-4).cumsum(-1)),dim=-1)

def ordered_pairs(quantiles):return torch.stack([quantiles[:,[low,high]] for low,high in PAIR_INDICES],dim=1)

def pinball(pairs,response):
    tau=pairs.new_tensor([[.05,.95],[.025,.975],[.005,.995]])
    residual=response[:,None,None]-pairs
    return torch.maximum(tau*residual,(tau-1)*residual).mean((1,2))

def geometry(quantiles):
    center=(quantiles[:,2]+quantiles[:,3])[:,None]/2
    left=center-quantiles[:,[2,1,0]];right=quantiles[:,[3,4,5]]-center
    return center,left,right

class CoverageConfidence(nn.Module):
    def __init__(self,features=1038,width=32):
        super().__init__();self.width=width
        self.net=nn.Sequential(nn.Linear(features+4,width),nn.Tanh(),nn.Linear(width,2))
        self.register_buffer('scale_grid',torch.tensor(SCALE_GRID));self.register_buffer('temperature',torch.tensor(1.))
    def logits(self,x,center,left,right):
        levels=left.shape[1]
        geo=torch.stack((center.expand(-1,levels),left.clamp_min(1e-5).log(),right.clamp_min(1e-5).log(),
          x.new_tensor([.1,.05,.01]).log()[None,:].expand(len(x),-1)),dim=-1)
        raw=self.net(torch.cat((x[:,None,:].expand(-1,levels,-1),geo.clamp(-20,20)),dim=-1))
        return raw[:,:,0,None]+(F.softplus(raw[:,:,1])+1e-4)[:,:,None]*self.scale_grid.log()[None,None,:]
    def forward(self,x,center,left,right):
        probabilities=torch.sigmoid(self.logits(x,center,left,right)/self.temperature)
        qualifies=probabilities>=x.new_tensor([.9,.95,.99])[None,:,None]
        chosen=qualifies.long().argmax(-1)
        chosen=torch.where(qualifies.any(-1),chosen,torch.full_like(chosen,len(SCALE_GRID)-1))
        return self.scale_grid[chosen],probabilities

class QuantileCorrection(nn.Module):
    def __init__(self,features=1038,width=16):
        super().__init__();self.width=width
        self.net=nn.Sequential(nn.Linear(features,width),nn.Tanh(),nn.Linear(width,7))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,center,left,right):
        raw=self.net(x);offset=raw[:,:1]
        low=(left.clamp_min(1e-5)*raw[:,1:4].clamp(-3,3).exp()).cummax(1).values
        high=(right.clamp_min(1e-5)*raw[:,4:7].clamp(-3,3).exp()).cummax(1).values
        return torch.stack((center+offset-low,center+offset+high),dim=-1),raw

class PosttrainedV3Error(nn.Module):
    def __init__(self,variant='head_only',input_dim=1038,width=256,confidence_width=32,correction_width=16):
        super().__init__()
        if variant not in VARIANTS:raise ValueError('Unregistered post-training variant')
        self.variant=variant;self.input_dim=input_dim;self.width=width
        self.base=NIEErrorHead(input_dim,width)
        self.confidence=CoverageConfidence(input_dim,confidence_width) if variant in ('confidence_only','full','full_unbalanced') else None
        self.correction=QuantileCorrection(input_dim,correction_width) if variant in ('balanced_correction_only','full','full_unbalanced') else None
    def reference_geometry(self,x):
        center,left,right=geometry(self.base(x))
        if self.confidence is not None:
            scale,_=self.confidence(x,center,left,right)
            left=(left*scale).cummax(1).values;right=(right*scale).cummax(1).values
        return center,left,right
    def forward(self,x):
        if self.variant=='head_only':return ordered_pairs(self.base(x))
        center,left,right=self.reference_geometry(x)
        if self.correction is not None:return self.correction(x,center,left,right)[0]
        return torch.stack((center-left,center+right),dim=-1)

def make_document(model,metadata):
    return dict(format=MODEL_FORMAT,variant=model.variant,input_dim=model.input_dim,width=model.width,
      confidence_width=32 if model.confidence is None else model.confidence.width,
      correction_width=16 if model.correction is None else model.correction.width,
      state_dict=model.state_dict(),metadata=metadata)

def save_model(path,model,metadata):torch_save_new(path,make_document(model,metadata))

def load_model(path,device='cpu'):
    doc=torch.load(path,map_location='cpu',weights_only=False)
    if doc.get('format')!=MODEL_FORMAT:raise ValueError('Wrong post-training checkpoint format')
    source=doc['metadata'].get('source_sha256',{})
    for name in ('heads.py','common.py'):
        if source.get(name)!=sha256(Path(__file__).with_name(name)):raise ValueError('Selected head architecture/contract source changed: '+name)
    model=PosttrainedV3Error(doc['variant'],doc['input_dim'],doc['width'],doc['confidence_width'],doc['correction_width'])
    model.load_state_dict(doc['state_dict'])
    return model.to(device).eval(),doc
