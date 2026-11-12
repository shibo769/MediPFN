"""Learned NIE standardized-error quantiles on frozen-backbone task summaries.

The head learns a task-level estimator error, not outcome predictive uncertainty.
It is a mediation-specific adaptation; it is not the released/original TabCon.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

LEVELS=(.005,.025,.05,.95,.975,.995)
FORMAT='medipfn-frozen-tabpfn-nie-error-head-1'


class NIEErrorHead(nn.Module):
    def __init__(self,input_dim,width=256):
        super().__init__();self.input_dim=int(input_dim);self.width=int(width)
        self.network=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,width),
                                   nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,len(LEVELS)))

    def forward(self,features):
        raw=self.network(features)
        # Monotone quantiles without a fixed finite tail range.
        return torch.cat([raw[...,:1],raw[...,:1]+torch.cumsum(F.softplus(raw[...,1:])+1e-4,dim=-1)],dim=-1)

    def loss(self,features,error):
        q=self(features);residual=error[...,None]-q
        levels=torch.as_tensor(LEVELS,device=q.device,dtype=q.dtype)
        return torch.maximum(levels*residual,(levels-1)*residual).mean()


def save_head(path,head,backbone_manifest,feature_schema,training_metadata):
    payload=dict(format=FORMAT,state_dict=head.state_dict(),input_dim=head.input_dim,width=head.width,
                 levels=LEVELS,backbone_manifest=backbone_manifest,feature_schema=feature_schema,
                 training_metadata=training_metadata)
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,p)


def load_head(path,expected_backbone_manifest,map_location='cpu'):
    c=torch.load(path,map_location=map_location,weights_only=False)
    if c.get('format')!=FORMAT or c.get('backbone_manifest')!=expected_backbone_manifest:
        raise ValueError('Different backbone checkpoint/configuration')
    h=NIEErrorHead(c['input_dim'],c['width']);h.load_state_dict(c['state_dict']);h.eval()
    return h,c
