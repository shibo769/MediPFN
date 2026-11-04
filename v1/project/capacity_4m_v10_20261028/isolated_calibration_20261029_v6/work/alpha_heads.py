"""Two fixed-level heads using the frozen 95%-confidence anchor.

The confidence network does not predict 90% or 99% coverage: its pre-existing
95% scale is only an input transform. These heads and their calibrations are
new, level-specific artifacts. No routine in this module accesses truth.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LEVELS = (0.90, 0.99)
PROBS = {0.90: (0.05, 0.95), 0.99: (0.005, 0.995)}


class QuantileHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 16), nn.Tanh(), nn.Linear(16, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def anchor(q95, scale):
    q = np.asarray(q95, dtype=np.float32)
    sc = np.asarray(scale, dtype=np.float32)
    assert q.ndim == 2 and q.shape[1] == 3 and sc.shape == (len(q),)
    return np.column_stack((q[:, 1] + sc * (q[:, 0] - q[:, 1]),
                            q[:, 1], q[:, 1] + sc * (q[:, 2] - q[:, 1]))).astype(np.float32)


def endpoints(raw, q):
    center = q[:, 1] + raw[:, 0]
    left = (q[:, 1] - q[:, 0]).clamp_min(1e-5) * torch.exp(raw[:, 1].clamp(-3, 3))
    right = (q[:, 2] - q[:, 1]).clamp_min(1e-5) * torch.exp(raw[:, 2].clamp(-3, 3))
    return center - left, center + right


def pinball(residual, probability):
    return torch.maximum(probability * residual, (probability - 1) * residual)


def loss(head, x, q, truth_z, group, level):
    raw = head(x)
    lo, hi = endpoints(raw, q)
    p_lo, p_hi = PROBS[level]
    values = pinball(truth_z - lo, p_lo) + pinball(truth_z - hi, p_hi)
    # The four scientific construction groups contribute equal weight.
    pieces = [values[group == j].mean() for j in range(4)]
    if any(not torch.isfinite(v) for v in pieces):
        raise ValueError('missing or nonfinite training group')
    return torch.stack(pieces).mean() + 0.0001 * (raw ** 2).mean()


@torch.no_grad()
def residual_endpoints(head, normalized_x, q95, scale):
    q = torch.from_numpy(anchor(q95, scale))
    raw = head(torch.from_numpy(np.asarray(normalized_x, dtype=np.float32)))
    lo, hi = endpoints(raw, q)
    return lo.numpy(), hi.numpy()


def expansion_score(z, lo, hi):
    w = np.maximum(np.asarray(hi) - np.asarray(lo), 1e-5)
    z = np.asarray(z)
    return np.maximum.reduce((np.zeros_like(z), (lo - z) / w, (z - hi) / w))


def relative_order_statistic(z, lo, hi, level):
    scores = expansion_score(z, lo, hi)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError('invalid calibration scores')
    # levels are rational decimals 0.90 and 0.99; avoid binary float ceil drift.
    numerator = {0.90: 90, 0.99: 99}[float(level)]
    k = ((len(scores) + 1) * numerator + 99) // 100
    if not 1 <= k <= len(scores):
        raise ValueError('insufficient calibration mechanisms for requested level')
    return float(np.partition(scores, k - 1)[k - 1]), int(k)


def prediction_intervals(head, x_norm, q95, scales, estimates, ses, expansion):
    lo, hi = residual_endpoints(head, x_norm, q95, scales)
    if not np.isfinite([expansion]).all() or expansion < 0:
        raise ValueError('invalid frozen calibration expansion')
    width = hi - lo
    lo = lo - expansion * width
    hi = hi + expansion * width
    estimates, ses = np.asarray(estimates), np.asarray(ses)
    if np.any(ses <= 0) or not np.isfinite([lo, hi, estimates, ses]).all():
        raise ValueError('invalid or failed interval')
    return estimates - ses * hi, estimates - ses * lo
