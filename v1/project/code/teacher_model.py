"""Separate direct-teacher baseline, same causal-role support encoder config.

Retain but freeze unused nuisance-query modules for exact encoder compatibility.
The effect head restores an outcome contrast using Yscale, without Ycenter.
The variance head sees a detached encoder representation and returns log V,
not log(V/n). Deployment SE is exp(logV/2)/sqrt(n).
"""
from pathlib import Path
import math
import torch
from torch import nn
from model import MediationPFN


class DirectTeacherPFN(nn.Module):
    def __init__(self,size='medium',config=None):
        super().__init__();self.encoder=MediationPFN(config=config,size=size);w=self.encoder.config.width
        self.effect_head=nn.Sequential(nn.Linear(w,w),nn.GELU(),nn.Linear(w,1))
        self.variance_head=nn.Sequential(nn.Linear(w,w),nn.GELU(),nn.Linear(w,1))
        for name in ('type_embedding','query_attention','query_norm','query_ff','output_norm','output'):
            for parameter in getattr(self.encoder,name).parameters():parameter.requires_grad_(False)
    @property
    def parameter_counts(self):
        return dict(retained=sum(p.numel() for p in self.parameters()),active=sum(p.numel() for p in self.parameters() if p.requires_grad),
            inactive_nuisance_query=sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad),
            active_encoder=sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            direct_effect_head=sum(p.numel() for p in self.effect_head.parameters()),direct_variance_head=sum(p.numel() for p in self.variance_head.parameters()))
    def forward(self,rows,feature_mask=None):
        context=self.encoder.encode_support(rows,feature_mask=feature_mask);pooled=context['pooled'];scale=context['y_scale'].squeeze(-1)
        effect=self.effect_head(pooled).squeeze(-1)*scale
        logV=self.variance_head(pooled.detach()).squeeze(-1)+2*scale.log()
        return dict(estimate=effect,log_variance_coefficient=logV,outcome_scale=scale)


def teacher_loss(prediction,effect_target,logV_target,variance_weight=1.):
    effect=((prediction['estimate']-effect_target)/prediction['outcome_scale']).square().mean()
    variance=(prediction['log_variance_coefficient']-logV_target).square().mean()
    return effect+variance_weight*variance,dict(normalized_effect_MSE=effect.detach(),log_population_V_MSE=variance.detach())


def save_teacher(path,model,**metadata):
    path=Path(path)
    if path.exists():raise FileExistsError('Teacher checkpoint immutable')
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(format='v9-direct-teacher-1',config=model.encoder.config_dict(),state_dict=model.state_dict(),parameter_counts=model.parameter_counts,
        variance_definition='Population EIF variance coefficient V, never V/n',standard_error='exp(logV/2)/sqrt(n)',**metadata),path)


def load_teacher(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint['format']!='v9-direct-teacher-1':raise ValueError('Unknown teacher checkpoint')
    model=DirectTeacherPFN(config=checkpoint['config']);model.load_state_dict(checkpoint['state_dict']);model.to(device).eval()
    if model.parameter_counts!=checkpoint['parameter_counts']:raise ValueError('Teacher active/retained parameter count mismatch')
    return model
