"""Nested signed-error quantiles and consistent interval construction for v2.

The response is e=(theta_hat-theta)/scale. Truth appears only in the supervised
response or calibration score. Deployment features use observed data and frozen
nuisance predictions. Cumulative positive widths guarantee nested intervals for
the registered confidence levels before calibration.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ERROR_FORMAT = "medipfn-v2-signed-error-quantiles-1"
DEFAULT_ALPHAS = (.10, .05, .01)
SUMMARY_FEATURES = ("log_n", "estimate_over_y_sd", "log_scale_over_y_sd", "score_skew",
                    "log_one_plus_score_kurtosis", "score_q25_over_sd", "score_q75_over_sd",
                    "pi_clip_fraction", "g_clip_fraction", "log_one_plus_max_ratio",
                    "treatment_fraction", "log_raw_se_over_y_sd")


def uncertainty_features(result, table, relative_scale_floor=1e-3):
    """Return observed-only features and the *same* scale used everywhere else."""
    if not relative_scale_floor > 0:
        raise ValueError("The registered relative scale floor must be positive")
    y = np.asarray(table["y"], float).reshape(-1)
    score = np.asarray(result["scores"], float).reshape(-1)
    if len(score) != len(y) or len(y) < 2:
        raise ValueError("Score and observed-table denominators differ")
    y_sd = max(float(y.std(ddof=1)), 1e-6)
    raw_se = float(result["standard_error"])
    if not math.isfinite(raw_se) or raw_se < 0:
        raise ValueError("Raw standard error must be finite and nonnegative")
    scale = max(raw_se, relative_scale_floor*y_sd)
    score_sd = max(float(score.std(ddof=1)), relative_scale_floor*y_sd)
    standardized_score = (score-float(result["estimate"]))/score_sd
    diagnostics = result["diagnostics"]
    counts = np.asarray([d["query_rows"] for d in diagnostics], float)
    weights = counts/counts.sum()
    summary = [math.log(len(y)), float(result["estimate"])/y_sd,
               math.log(scale/y_sd), float(np.mean(standardized_score**3)),
               math.log1p(float(np.mean(standardized_score**4))),
               float(np.quantile(standardized_score,.25)), float(np.quantile(standardized_score,.75)),
               float(sum(w*d["pi_clipped"] for w,d in zip(weights,diagnostics))),
               float(sum(w*d["g_clipped"] for w,d in zip(weights,diagnostics))),
               math.log1p(max(d["ratio_max"] for d in diagnostics)),
               float(np.mean(table["a"])), math.log(max(raw_se/y_sd,1e-12))]
    pooled = np.asarray(result["pooled_context"], float).reshape(-1)
    features = np.concatenate((pooled,np.asarray(summary))).astype(np.float32)
    if not np.isfinite(features).all() or not np.isfinite(scale):
        raise FloatingPointError("Nonfinite observed uncertainty features")
    return features,scale


@dataclass
class ErrorConfig:
    feature_count: int
    width: int = 128
    depth: int = 2
    alphas: tuple = DEFAULT_ALPHAS
    relative_scale_floor: float = 1e-3


class SignedErrorQuantiles(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config,dict):
            config = dict(config)
            config["alphas"] = tuple(config["alphas"])
            config = ErrorConfig(**config)
        self.config = config
        alphas = config.alphas
        if any(not 0 < a < 1 for a in alphas) or any(a <= b for a,b in zip(alphas,alphas[1:])):
            raise ValueError("Alpha levels must decrease strictly, e.g. .10,.05,.01")
        self.register_buffer("feature_center",torch.zeros(config.feature_count))
        self.register_buffer("feature_scale",torch.ones(config.feature_count))
        layers = [nn.Linear(config.feature_count,config.width),nn.GELU()]
        for _ in range(config.depth-1):
            layers.extend([nn.Linear(config.width,config.width),nn.GELU()])
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(config.width,1+2*len(alphas))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        critical = np.asarray([NormalDist().inv_cdf(1-a/2) for a in alphas])
        increments = np.diff(np.concatenate(([0.],critical)))
        inverse_softplus = np.log(np.expm1(increments))
        self.register_buffer("width_offset",torch.tensor(inverse_softplus,dtype=torch.float32))

    def set_normalization(self, features):
        with torch.no_grad():
            self.feature_center.copy_(features.mean(0))
            self.feature_scale.copy_(features.std(0,unbiased=False).clamp_min(1e-3))

    def forward(self,features):
        standardized = ((features-self.feature_center)/self.feature_scale).clamp(-20,20)
        raw = self.output(self.body(standardized))
        levels = len(self.config.alphas)
        center = raw[:,0,None]
        left = F.softplus(raw[:,1:1+levels]+self.width_offset).cumsum(-1)
        right = F.softplus(raw[:,1+levels:]+self.width_offset).cumsum(-1)
        return torch.stack((center-left,center+right),-1)

    def pair(self,features,alpha):
        indices = [i for i,a in enumerate(self.config.alphas) if abs(a-alpha)<1e-10]
        if len(indices) != 1:
            raise ValueError("Requested alpha was not registered during training")
        return self(features)[:,indices[0],:]


def pinball_loss(predictions,response,alphas=DEFAULT_ALPHAS):
    if predictions.ndim != 3 or predictions.shape[1:] != (len(alphas),2):
        raise ValueError("Expected [tables, alpha levels, lower/upper quantile]")
    tau = predictions.new_tensor([[a/2,1-a/2] for a in alphas])
    residual = response[:,None,None]-predictions
    return torch.maximum(tau*residual,(tau-1)*residual).mean()


def interval_from_pair(estimate,scale,pair,expansion=0.):
    if not scale > 0 or not expansion >= 0:
        raise ValueError("Positive scale and nonnegative expansion are required")
    lower,upper = map(float,pair)
    if lower > upper:
        raise ValueError("Crossing signed-error quantiles")
    return [float(estimate-scale*(upper+expansion)),float(estimate-scale*(lower-expansion))]


def save_error_checkpoint(path,model,**metadata):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save(dict(format=ERROR_FORMAT,config=asdict(model.config),state_dict=model.state_dict(),**metadata),temporary)
    temporary.replace(path)


def load_error_checkpoint(path,device="cpu"):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint.get("format") != ERROR_FORMAT:
        raise ValueError("Not a MediPFN v2 signed-error checkpoint")
    model=SignedErrorQuantiles(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(),checkpoint
