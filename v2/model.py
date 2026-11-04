"""Custom amortized nuisance network for variable-dimensional mediation tasks.

This is a new architecture/checkpoint format. It does not load v1 or TabPFN
weights. Only observed support data enter the encoder. A query's treatment,
mediator, and outcome are masked according to the requested nuisance function.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F

TASKS = {"pi": 0, "g": 1, "mu": 2, "eta0": 3, "eta1": 4}
FORMAT = "medipfn-v2-bayes-nuisance-1"


@dataclass
class ModelConfig:
    width: int = 256
    heads: int = 8
    depth: int = 4
    inducing_tokens: int = 32
    ff_multiplier: int = 3
    max_x: int = 32
    max_m: int = 128


def _pad(value, maximum):
    if value.ndim != 3 or value.shape[-1] > maximum:
        raise ValueError("Expected [batch, rows, features] within configured bounds")
    return F.pad(value, (0, maximum - value.shape[-1]))


def _stats(value, valid):
    weight = valid[:, :, None].to(value.dtype)
    count = weight.sum(1, keepdim=True).clamp_min(1)
    center = (value * weight).sum(1, keepdim=True) / count
    variance = ((value - center).square() * weight).sum(1, keepdim=True) / count
    return center, variance.sqrt().clamp_min(1e-3)


class MediationPFNV2(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = ModelConfig(**config) if isinstance(config, dict) else config or ModelConfig()
        c = self.config
        if c.width % c.heads:
            raise ValueError("Width must be divisible by number of heads")
        w = c.width
        self.x_map = nn.Sequential(nn.Linear(3 * c.max_x, w), nn.GELU(), nn.Linear(w, w))
        self.m_map = nn.Sequential(nn.Linear(3 * c.max_m, w), nn.GELU(), nn.Linear(w, w))
        self.a_map = nn.Linear(2, w)
        self.y_map = nn.Linear(1, w)
        self.row_fuse = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 3*w), nn.GELU(), nn.Linear(3*w, w))
        self.task_embedding = nn.Embedding(len(TASKS), w)
        self.inducing = nn.Parameter(torch.randn(1, c.inducing_tokens, w) * .02)
        self.size_map = nn.Linear(2, w)
        self.support_attention = nn.MultiheadAttention(w, c.heads, dropout=0, batch_first=True)
        self.support_norm = nn.LayerNorm(w)
        layer = nn.TransformerEncoderLayer(w, c.heads, c.ff_multiplier*w,
                                          dropout=0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.depth, enable_nested_tensor=False)
        self.query_attention = nn.MultiheadAttention(w, c.heads, dropout=0, batch_first=True)
        self.query_norm = nn.LayerNorm(w)
        self.query_ff = nn.Sequential(nn.Linear(w, 3*w), nn.GELU(), nn.Linear(3*w, w))
        self.output = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 1))
        nn.init.normal_(self.output[-1].weight, std=.02)
        nn.init.zeros_(self.output[-1].bias)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def _features(self, x, m, x_mask, m_mask, x_binary, m_binary, context=None,
                  include_m=True):
        x = _pad(x, self.config.max_x)
        m = _pad(m, self.config.max_m)
        if context is None:
            raise ValueError("Support normalization is required")
        xn = ((x - context["x_center"]) / context["x_scale"]).clamp(-30, 30)
        mn = ((m - context["m_center"]) / context["m_scale"]).clamp(-30, 30)
        xm = x_mask[:, None, :].expand(-1, x.shape[1], -1)
        mm = m_mask[:, None, :].expand(-1, m.shape[1], -1)
        x_role = self.x_map(torch.cat((xn*xm, xm, x_binary[:, None, :].expand_as(xm)*xm), -1))
        if include_m:
            m_role = self.m_map(torch.cat((mn*mm, mm, m_binary[:, None, :].expand_as(mm)*mm), -1))
        else:
            m_role = 0
        return x_role + m_role

    def encode(self, support, target_y=None, row_valid=None, kind="mu"):
        x, m, a = support["x"], support["m"], support["a"]
        y = support["y"] if target_y is None else target_y
        b, n = a.shape
        valid = torch.ones_like(a, dtype=torch.bool) if row_valid is None else row_valid.bool()
        if not bool(valid.any(1).all()):
            raise ValueError("Each context needs at least one valid support row")
        xp, mp = _pad(x, self.config.max_x), _pad(m, self.config.max_m)
        xc, xs = _stats(xp, valid)
        mc, ms = _stats(mp, valid)
        yc, ys = _stats(y[..., None], valid)
        context = dict(x_center=xc, x_scale=xs, m_center=mc, m_scale=ms,
                       y_center=yc.squeeze(-1), y_scale=ys.squeeze(-1))
        roles = self._features(x, m, support["x_mask"], support["m_mask"],
                               support["x_binary"], support["m_binary"], context,
                               include_m=kind == "mu")
        if kind == "mu":
            roles = roles + self.a_map(torch.stack((a, torch.ones_like(a)), -1))
        roles = roles + self.y_map(((y[..., None] - yc) / ys).clamp(-30, 30))
        roles = roles + self.task_embedding.weight[TASKS[kind]]
        rows = roles + self.row_fuse(roles)
        count = valid.sum(1).to(rows.dtype)
        sizes = torch.stack((count.log()/6, count.rsqrt()), -1)
        latent = self.inducing.expand(b, -1, -1) + self.size_map(sizes)[:, None, :]
        attended, _ = self.support_attention(latent, rows, rows,
                                              key_padding_mask=~valid, need_weights=False)
        tokens = self.encoder(self.support_norm(latent + attended))
        context.update(tokens=tokens, pooled=tokens.mean(1),
                       x_mask=support["x_mask"], m_mask=support["m_mask"],
                       x_binary=support["x_binary"], m_binary=support["m_binary"])
        return context

    def predict(self, context, x, m, task, a=None):
        if task not in TASKS:
            raise ValueError("Unknown nuisance task")
        roles = self._features(x, m, context["x_mask"], context["m_mask"],
                               context["x_binary"], context["m_binary"], context,
                               include_m=task in ("g", "mu"))
        if task == "mu":
            if a is None:
                raise ValueError("Outcome query needs its intervention treatment")
            roles = roles + self.a_map(torch.stack((a, torch.ones_like(a)), -1))
        query = roles + self.task_embedding.weight[TASKS[task]]
        attended, _ = self.query_attention(query, context["tokens"], context["tokens"], need_weights=False)
        hidden = self.query_norm(query + attended)
        raw = self.output(hidden + self.query_ff(hidden)).squeeze(-1)
        if task in ("pi", "g"):
            return raw
        return raw * context["y_scale"] + context["y_center"]


def subset(table, index):
    return {key: value[:, index] if key in ("x", "m", "a", "y") else value
            for key, value in table.items()}


def nested_contexts(model, support, teacher=None):
    """Inner out-of-fold mu1 pseudo-labels, using outer support rows only."""
    n = support["a"].shape[1]
    cut = n // 2
    if cut < 2:
        raise ValueError("Need at least four support rows for inner cross-fitting")
    first, second = subset(support, slice(None, cut)), subset(support, slice(cut, None))
    teacher = model if teacher is None else teacher
    with torch.no_grad():
        c0, c1 = teacher.encode(first), teacher.encode(second)
        p0 = teacher.predict(c1, first["x"], first["m"], "mu", torch.ones_like(first["a"]))
        p1 = teacher.predict(c0, second["x"], second["m"], "mu", torch.ones_like(second["a"]))
        pseudo = torch.cat((p0, p1), 1)
    eta0 = model.encode(support, target_y=pseudo, row_valid=support["a"] == 0, kind="eta0")
    eta1 = model.encode(support, target_y=pseudo, row_valid=support["a"] == 1, kind="eta1")
    return eta0, eta1


def save_checkpoint(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(format=FORMAT, config=asdict(model.config), state_dict=model.state_dict(),
                    parameter_count=model.parameter_count, **metadata), temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a MediPFN v2 checkpoint")
    model = MediationPFNV2(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint
