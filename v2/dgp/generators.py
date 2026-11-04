"""Vectorized full-data adaptations of fifteen frozen literature registry records.

Target: superpopulation NIE at A=1, E[Y(1,M(1))-Y(1,M(0))].
The original principal-stratum/path-specific/missing-data estimands are NOT reused.
All post-treatment confounder effects are removed in the structural equations.
Shan's two observation mechanisms collapse to one full-data mechanism.
"""
from copy import deepcopy
from functools import lru_cache
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_COMMIT = "8182caf7185845a31a2122e032811bb2254d6342"
_VENDOR = Path(__file__).resolve().parent / "vendor"
_ALIASES = {
    "cheng": "mediation-cheng-li-2025-principal-strata-kang-schafer-v1",
    "diaz_hejazi": "mediation-diaz-hejazi-2020-discrete-three-mediators-v1",
    "diaz_intermediate": "mediation-diaz-et-al-2021-intermediate-confounding-v1",
    "he_song_xu": "mediation-he-song-xu-2024-single-linear-v1",
    "jones_lnl": "mediation-jones-2024-sparse-mediators-lnl-v1",
    "jones_lnn": "mediation-jones-2024-sparse-mediators-lnn-v1",
    "jones_nnn": "mediation-jones-2024-sparse-mediators-nnn-v1",
    "jones_correlated": "mediation-jones-et-al-2025-correlated-sparse-v1",
    "miles": "mediation-miles-2019-toxicity-path-specific-v1",
    "shan": "mediation-shan-li-ai-2026-missing-confounders-aux-shadow-v1",
    "shan_internal": "mediation-shan-li-ai-2026-missing-confounders-internal-shadow-v1",
    "eric": "nie-eric-original-v1",
    "xia_binary": "mediation-xia-chan-2023-binary-plackett-v1",
    "xia_continuous": "mediation-xia-chan-2023-continuous-copula-v1",
    "zuo": "mediation-zuo-et-al-2025-continuous-mnar-v1",
}
_FAMILIES = {k: ("jones" if k.startswith("jones") else
                 "xia_chan" if k.startswith("xia") else
                 "shan" if k.startswith("shan") else k) for k in _ALIASES}
_TRAIN = ("cheng", "diaz_hejazi", "diaz_intermediate", "he_song_xu",
          "jones_lnl", "jones_lnn", "jones_correlated", "miles", "shan", "zuo")
_SPLITS = {"train": _TRAIN, "dev": _TRAIN, "calibration": _TRAIN,
           "id_test": _TRAIN, "ood_source": ("eric", "xia_binary", "xia_continuous"),
           "ood_composition": ("jones_nnn",),
           "all": tuple(k for k in _ALIASES if k != "shan_internal")}
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(80)
_GH_X = np.sqrt(2.0) * _GH_X
_GH_W = _GH_W / np.sqrt(np.pi)


def expit(x):
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=float)))


def _alias(source_id):
    if source_id in _ALIASES:
        return "shan" if source_id == "shan_internal" else source_id
    for k, v in _ALIASES.items():
        if source_id == v:
            return "shan" if k == "shan_internal" else k
    raise ValueError("Unknown source_id: " + str(source_id))


def available_sources(split="train"):
    """Short source IDs; OOD sources never appear in train/dev/calibration."""
    if split not in _SPLITS:
        raise ValueError("Unknown split: " + split)
    return list(_SPLITS[split])


def registry():
    return [{"source_id": k, "registry_id": v, "source_group": _FAMILIES[k],
             "default_split": "train" if k in _TRAIN else
             "duplicate" if k == "shan_internal" else
             "ood_composition" if k == "jones_nnn" else "ood_source"}
            for k, v in _ALIASES.items()]


@lru_cache(None)
def _record(source):
    return json.loads((_VENDOR / (_ALIASES[source] + ".json")).read_text(encoding="utf-8"))


def _remove_confounders(source, c):
    if source == "cheng":
        c.update(mediator_event_coefficient=0.0, outcome_event_coefficient=0.0)
    elif source == "diaz_intermediate":
        c.update(m_z=0.0, m_az=0.0, y_z=0.0, y_az=0.0)
    elif source == "miles":
        for k in ("mediator_confounder", "mediator_exposure_confounder", "outcome_confounder"):
            c[k] = [0.0, 0.0, 0.0]
    elif source.startswith("xia_"):
        c.update(y_c=0.0, y_ac=0.0)
    return c


def sample_mechanism(source_id, seed=0, randomize=True):
    """Draw ID-prior parameters independently of observation-table seeds.

    The source forms are retained. Coefficients/noise/path strengths vary around
    the source presets; exact zero path nulls are sampled independently.
    This is a new training prior, not a claim to reproduce the original experiments.
    """
    s = _alias(source_id)
    r = np.random.default_rng(int(seed))
    scenarios = _record(s)["scenarios"]
    c = deepcopy(scenarios[int(r.integers(len(scenarios))) if randomize else 0]["parameters"])
    _remove_confounders(s, c)
    if not randomize:
        return c
    # No automatic mutation of probability constraints, observation masks or source IDs.
    keys = {
        "cheng": ["mediator_treatment_coefficient", "outcome_mediator_coefficient", "outcome_noise_sd", "covariate_sd"],
        "diaz_hejazi": ["outcome_treatment_coefficient", "outcome_noise_sd"],
        "diaz_intermediate": ["m_a", "m_aw2", "y_m", "y_a"],
        "he_song_xu": ["alpha_s", "beta_m", "mediator_noise_sd", "outcome_noise_sd", "mediator_x1", "mediator_x2", "outcome_x1", "outcome_x2"],
        "miles": ["mediator_exposure", "outcome_mediator", "outcome_exposure_mediator", "mediator_sd", "outcome_sd"],
        "shan": ["mediator_treatment", "mediator_treatment_x1_squared", "outcome_mediator", "outcome_treatment_mediator", "mediator_noise_sd", "outcome_noise_sd"],
        "zuo": ["alpha_t", "beta_m", "beta_mt", "m_sd", "y_sd", "alpha_x", "beta_x"],
        "eric": ["mediator_treatment_coefficient", "outcome_noise_sd", "x3_sd"],
        "xia_binary": ["m_logit_a", "y_m", "y_am", "y_sd"],
        "xia_continuous": ["m_a", "y_m", "y_am", "m_sd", "y_sd"],
    }
    for k in keys.get(s, []):
        c[k] = float(c[k]) * float(r.uniform(0.5, 1.5))
    if s == "diaz_hejazi":
        c["mediation_path_scale"] = float(r.uniform(0.2, 1.0))
        c["outcome_mediator_coefficients"] = (np.asarray(c["outcome_mediator_coefficients"]) * r.uniform(0.5, 1.5, 3)).tolist()
        c["baseline_probabilities"] = r.uniform(0.25, 0.75, 3).tolist()
    if s.startswith("jones_") and s != "jones_correlated":
        q = int(r.choice([3, 5, 10, 20]))
        k = min(3, q)
        alpha = np.zeros(q); beta = np.zeros(q)
        alpha[:k] = r.uniform(0.3, 1.8, k) * r.choice([-1, 1], k)
        beta[:k] = r.uniform(0.2, 1.0, k) * r.choice([-1, 1], k)
        c.update(mediator_count=q, alpha0=alpha.tolist(), beta0=beta.tolist(),
                 propensity_scale=float(r.uniform(0.4, 1.2)),
                 mediator_noise_sd=float(r.uniform(0.6, 1.5)),
                 outcome_noise_sd=float(r.uniform(0.6, 1.5)))
    if s == "jones_correlated":
        c.update(p=int(r.choice([10, 20, 50, 100])), rho_within=float(r.uniform(0.15, 0.6)),
                 a_to_m_coefficient=float(r.uniform(0.3, 1.5)) * int(r.choice([-1, 1])),
                 m_to_y_coefficient=float(r.uniform(0.3, 1.5)) * int(r.choice([-1, 1])),
                 outcome_error_sd=float(r.uniform(0.6, 1.5)))
        c["rho_bridge"] = float(r.uniform(0.0, c["rho_within"]))
    # Include both composite-null pathways; do not select them from test performance.
    null = float(r.random())
    if null < 0.15:
        _zero_path(s, c, "a_to_m")
    elif null < 0.30:
        _zero_path(s, c, "m_to_y")
    elif r.random() < 0.5:
        _flip_path(s, c)
    return c


def _zero_path(s, c, path):
    ak = {"cheng": ["mediator_treatment_coefficient"], "diaz_hejazi": ["mediation_path_scale"],
          "diaz_intermediate": ["m_a", "m_aw2"], "he_song_xu": ["alpha_s"],
          "miles": ["mediator_exposure"], "shan": ["mediator_treatment", "mediator_treatment_x1_squared"],
          "zuo": ["alpha_t"], "eric": ["mediator_treatment_coefficient"],
          "xia_binary": ["m_logit_a"], "xia_continuous": ["m_a"],
          "jones_correlated": ["a_to_m_coefficient"]}
    bk = {"cheng": ["outcome_mediator_coefficient"], "diaz_hejazi": ["outcome_mediator_coefficients"],
          "diaz_intermediate": ["y_m"], "he_song_xu": ["beta_m"],
          "miles": ["outcome_mediator", "outcome_exposure_mediator"],
          "shan": ["outcome_mediator", "outcome_treatment_mediator"],
          "zuo": ["beta_m", "beta_mt"], "xia_binary": ["y_m", "y_am"],
          "xia_continuous": ["y_m", "y_am"], "jones_correlated": ["m_to_y_coefficient"]}
    if s.startswith("jones_") and s != "jones_correlated":
        key = "alpha0" if path == "a_to_m" else "beta0"
        c[key] = [0.0] * len(c[key]); return
    if s == "eric" and path == "m_to_y":
        c["outcome_coefficients"][5:7] = [0.0, 0.0]; return
    for k in (ak if path == "a_to_m" else bk).get(s, []):
        c[k] = [0.0] * len(c[k]) if isinstance(c[k], list) else 0.0


def _flip_path(s, c):
    # Signed indirect effects prevent memorizing source-specific effect signs.
    key = {"cheng": "outcome_mediator_coefficient", "he_song_xu": "beta_m",
           "diaz_hejazi": "outcome_mediator_coefficients", "diaz_intermediate": "y_m",
           "miles": "outcome_mediator", "shan": "outcome_mediator", "zuo": "beta_m",
           "xia_binary": "y_m", "xia_continuous": "y_m"}.get(s)
    if key:
        c[key] = [-v for v in c[key]] if isinstance(c[key], list) else -c[key]


def _weights_binary(w, probs):
    return np.prod(np.where(w, probs, 1.0 - probs), axis=1)


def _dh_probs(a, w):
    w1, w2, w3 = w.T
    return np.column_stack([1.0-expit((a+w1)/(a+w1+0.5)),
                            expit((a-1.0+w2)/(w3+3.0)),
                            expit((a-2.0+2.0*w1)/(2.0*w1+0.5))])


def _effect(s, c, x):
    n = len(x)
    if s == "cheng":
        b = c["mediator_intercept"] + x @ np.asarray(c["mediator_coefficients"])
        return c["outcome_mediator_coefficient"] * (expit(b+c["mediator_treatment_coefficient"])-expit(b))
    if s == "diaz_hejazi":
        return (_dh_probs(c["mediation_path_scale"], x)-_dh_probs(0.0, x)) @ np.asarray(c["outcome_mediator_coefficients"])
    if s == "diaz_intermediate":
        b = c["m_w1"]*x[:, 0]+c["m_w2"]*x[:, 1]
        pm = expit(b+c["m_a"]+c["m_aw2"]*x[:, 1])-expit(b)
        den = x.sum(1)+c["y_denominator_intercept"]
        return pm*(expit((c["y_a"]+c["y_m"])/den)-expit(c["y_a"]/den))
    if s == "eric":
        f = np.column_stack([np.ones(n), x, x[:, 0]*x[:, 2]])
        b = f @ np.asarray(c["mediator_coefficients"])
        return (c["outcome_coefficients"][5]+c["outcome_coefficients"][6])*(expit(b+c["mediator_treatment_coefficient"])-expit(b))
    if s == "xia_binary":
        b = c["m_logit_intercept"]+c["m_logit_x"]*x[:, 0]
        return (c["y_m"]+c["y_am"])*(expit(b+c["m_logit_a"])-expit(b))
    if s == "shan":
        return (c["outcome_mediator"]+c["outcome_treatment_mediator"])*(c["mediator_treatment"]+c["mediator_treatment_x1_squared"]*x[:, 0]**2)
    value = None
    if s == "he_song_xu": value = c["alpha_s"]*c["beta_m"]
    elif s.startswith("jones_") and s != "jones_correlated": value = float(np.dot(c["alpha0"], c["beta0"]))
    elif s == "jones_correlated": value = len(set(c["a_to_m_indices"]) & set(c["m_to_y_indices"]))*c["a_to_m_coefficient"]*c["m_to_y_coefficient"]
    elif s == "miles": value = c["mediator_exposure"]*(c["outcome_mediator"]+c["outcome_exposure_mediator"])
    elif s == "zuo": value = c["alpha_t"]*(c["beta_m"]+c["beta_mt"])
    elif s == "xia_continuous": value = c["m_a"]*(c["y_m"]+c["y_am"])
    if value is None: raise ValueError(s)
    return np.full(n, value, dtype=float)


def truth(source_id, config=None):
    s = _alias(source_id)
    c = sample_mechanism(s, randomize=False) if config is None else _remove_confounders(s, deepcopy(config))
    return _truth_cached(s, json.dumps(c, sort_keys=True, separators=(",", ":")))


@lru_cache(8192)
def _truth_cached(s, config_json):
    c = json.loads(config_json)
    if s in ("diaz_hejazi", "diaz_intermediate"):
        w = np.array(list(product([0., 1.], repeat=3)))
        if s == "diaz_hejazi":
            weights = _weights_binary(w, np.asarray(c["baseline_probabilities"]))
        else:
            p3 = c["p_w3_intercept"]+c["p_w3_slope"]*(w[:, 0]+w[:, 1])
            probs = np.column_stack([np.full(8, c["p_w1"]), np.full(8, c["p_w2"]), p3])
            weights = _weights_binary(w, probs)
        return float(weights @ _effect(s, c, w))
    if s == "cheng":
        scale = c["covariate_sd"]*np.linalg.norm(c["mediator_coefficients"])
        b = c["mediator_intercept"]+scale*_GH_X
        return float(c["outcome_mediator_coefficient"]*(_GH_W @ (expit(b+c["mediator_treatment_coefficient"])-expit(b))))
    if s == "eric":
        total = 0.0
        for x1, x2 in product([0., 1.], repeat=2):
            p1 = c["x1_probability"]; p2 = c["x2_coefficients"][0]+c["x2_coefficients"][1]*x1
            w = (p1 if x1 else 1-p1)*(p2 if x2 else 1-p2)
            xc = c["x3_coefficients"]
            x3 = xc[0]+xc[1]*x1+xc[2]*x2+c["x3_sd"]*_GH_X
            x = np.column_stack([np.full(len(x3), x1), np.full(len(x3), x2), x3])
            total += w*float(_GH_W @ _effect(s, c, x))
        return float(total)
    if s == "xia_binary":
        return float(_GH_W @ _effect(s, c, (c["x_sd"]*_GH_X)[:, None]))
    if s == "shan":
        return float((c["outcome_mediator"]+c["outcome_treatment_mediator"])*(c["mediator_treatment"]+c["mediator_treatment_x1_squared"]/3.0))
    return float(_effect(s, c, np.zeros((1, 1)))[0])


def _draw(s, c, n, r):
    """Observed x,a,m,y and types; no structural noise or oracle label is exposed."""
    if s == "cheng":
        x = c["covariate_sd"]*r.standard_normal((n, 4)); a = r.binomial(1, expit(x@c["treatment_coefficients"]))
        m = r.binomial(1, expit(c["mediator_intercept"]+c["mediator_treatment_coefficient"]*a+x@c["mediator_coefficients"]))[:, None]
        y = c["outcome_intercept"]+c["outcome_treatment_coefficient"]*a+c["outcome_mediator_coefficient"]*m[:, 0]+x@c["outcome_coefficients"]+c["outcome_noise_sd"]*r.standard_normal(n)
        return x,a,m,y,["binary"],["continuous"]*4,"continuous"
    if s in ("diaz_hejazi", "diaz_intermediate"):
        if s == "diaz_hejazi":
            x = r.binomial(1, c["baseline_probabilities"], (n,3))
            a = r.binomial(1, x.sum(1)/c["treatment_denominator"]+c["treatment_intercept"])
            m = r.binomial(1, _dh_probs(c["mediation_path_scale"]*a, x))
            y = m@c["outcome_mediator_coefficients"]+c["outcome_treatment_coefficient"]*a+c["outcome_baseline_quadratic_coefficient"]*x.sum(1)**2+c["outcome_noise_sd"]*r.standard_normal(n)
            return x,a,m,y,["binary"]*3,["binary"]*3,"continuous"
        x = np.empty((n,3)); x[:,0]=r.binomial(1,c["p_w1"],n);x[:,1]=r.binomial(1,c["p_w2"],n)
        x[:,2]=r.binomial(1,c["p_w3_intercept"]+c["p_w3_slope"]*(x[:,0]+x[:,1]))
        a=r.binomial(1,expit(c["treatment_intercept"]+c["treatment_w_sum"]*x.sum(1)+c["treatment_w1w2"]*x[:,0]*x[:,1]))
        m=r.binomial(1,expit(c["m_w1"]*x[:,0]+c["m_w2"]*x[:,1]+c["m_a"]*a+c["m_aw2"]*a*x[:,1]))[:,None]
        y=r.binomial(1,expit((c["y_a"]*a+c["y_m"]*m[:,0])/(x.sum(1)+c["y_denominator_intercept"])))
        return x,a,m,y,["binary"],["binary"]*3,"binary"
    if s == "he_song_xu":
        x=np.column_stack([c["x1_sd"]*r.standard_normal(n),r.binomial(1,c["x2_probability"],n)])
        a=r.binomial(1,c["treatment_probability"],n)
        m=(c["mediator_intercept"]+c["alpha_s"]*a+c["mediator_x1"]*x[:,0]+c["mediator_x2"]*x[:,1]+c["mediator_noise_sd"]*r.standard_normal(n))[:,None]
        y=c["outcome_intercept"]+c["beta_m"]*m[:,0]+c["outcome_direct"]*a+c["outcome_x1"]*x[:,0]+c["outcome_x2"]*x[:,1]+c["outcome_noise_sd"]*r.standard_normal(n)
        return x,a,m,y,["continuous"],["continuous","binary"],"continuous"
    if s.startswith("jones_") and s != "jones_correlated":
        x=np.sqrt(c["covariate_variance"])*r.standard_normal((n,3));q=c["mediator_count"]
        pa=expit(c["propensity_scale"]*((x[:,0]+x[:,1]) if c["propensity_form"]=="linear" else (x[:,0]*x[:,1]+x[:,1])))
        a=r.binomial(1,pa)
        xm=(x[:,0] if c["mediator_confounding_form"]=="linear" else x[:,0]**2)+x[:,1]-x[:,2]
        xy=2*((x[:,0]-.5) if c["outcome_confounding_form"]=="linear" else (x[:,0]-.5)**2)+x[:,1]+2*x[:,2]
        m=a[:,None]*np.asarray(c["alpha0"])+xm[:,None]+c["mediator_noise_sd"]*r.standard_normal((n,q))
        y=c["gamma0"]*a+m@c["beta0"]+xy+c["outcome_noise_sd"]*r.standard_normal(n)
        return x,a,m,y,["continuous"]*q,["continuous"]*3,"continuous"
    if s == "jones_correlated":
        q=int(c["p"]); within=float(c["rho_within"]);bridge=float(c["rho_bridge"])
        if not 0<=bridge<=within<1: raise ValueError("Need 0 <= bridge <= within < 1")
        a=r.binomial(1,c["p_treatment"],n)
        m=np.sqrt(1-within)*r.standard_normal((n,q))+np.sqrt(bridge)*r.standard_normal((n,1))
        group=r.standard_normal((n,2));m[:,:2]+=np.sqrt(within-bridge)*group[:,0,None];m[:,2:]+=np.sqrt(within-bridge)*group[:,1,None]
        # Registered code's covariance has unit diagonals; it does not use the unused mediator_error_sd config.
        ai=np.asarray(c["a_to_m_indices"])-1;bi=np.asarray(c["m_to_y_indices"])-1
        if np.any(ai<0) or np.any(ai>=q) or np.any(bi<0) or np.any(bi>=q): raise ValueError("Index outside mediator vector")
        m[:,ai]+=c["a_to_m_coefficient"]*a[:,None]
        y=c["direct_effect"]*a+c["m_to_y_coefficient"]*m[:,bi].sum(1)+c["outcome_error_sd"]*r.standard_normal(n)
        return np.empty((n,0)),a,m,y,["continuous"]*q,[],"continuous"
    if s == "miles":
        x=r.uniform(*c["baseline_support"],size=(n,1));a=r.binomial(1,expit(c["exposure_coefficients"][0]+c["exposure_coefficients"][1]*x[:,0]))
        m=(c["mediator_intercept"]+c["mediator_baseline"]*x[:,0]+c["mediator_exposure"]*a+c["mediator_sd"]*r.standard_normal(n))[:,None]
        y=c["outcome_intercept"]+c["outcome_baseline"]*x[:,0]+c["outcome_exposure"]*a+(c["outcome_mediator"]+c["outcome_exposure_mediator"]*a)*m[:,0]+c["outcome_sd"]*r.standard_normal(n)
        return x,a,m,y,["continuous"],["continuous"],"continuous"
    if s == "shan":
        # Phi(eps1),Phi(eps2) are independent Uniform(0,1); unused shadows/masks are removed.
        x=np.column_stack([r.random((n,2)),r.binomial(1,c["x3_probability"],n)])
        a=r.binomial(1,expit(c["propensity_intercept"]+c["propensity_x1_squared"]*x[:,0]**2+c["propensity_x2"]*x[:,1]+c["propensity_x3"]*x[:,2]))
        m=(c["mediator_intercept"]+c["mediator_treatment"]*a+c["mediator_treatment_x1_squared"]*a*x[:,0]**2+c["mediator_x1_squared"]*x[:,0]**2+c["mediator_sin_x2"]*np.sin(x[:,1])+c["mediator_x3"]*x[:,2]+c["mediator_noise_sd"]*r.standard_normal(n))[:,None]
        y=c["outcome_intercept"]+c["outcome_treatment"]*a+(c["outcome_mediator"]+c["outcome_treatment_mediator"]*a)*m[:,0]+c["outcome_treatment_x1"]*a*x[:,0]+c["outcome_sin_x1"]*np.sin(x[:,0])+c["outcome_x2_squared"]*x[:,1]**2+c["outcome_x3"]*x[:,2]+c["outcome_noise_sd"]*r.standard_normal(n)
        return x,a,m,y,["continuous"],["continuous","continuous","binary"],"continuous"
    if s == "eric":
        x=np.empty((n,3));x[:,0]=r.binomial(1,c["x1_probability"],n)
        x[:,1]=r.binomial(1,c["x2_coefficients"][0]+c["x2_coefficients"][1]*x[:,0])
        xc=c["x3_coefficients"];x[:,2]=xc[0]+xc[1]*x[:,0]+xc[2]*x[:,1]+c["x3_sd"]*r.standard_normal(n)
        f=np.column_stack([np.ones(n),x,x[:,0]*x[:,2]]);a=r.binomial(1,expit(f@c["treatment_coefficients"]))
        m=r.binomial(1,expit(f@c["mediator_coefficients"]+c["mediator_treatment_coefficient"]*a))[:,None]
        y=np.column_stack([np.ones(n),x,a,m[:,0],a*m[:,0]])@c["outcome_coefficients"]+c["outcome_noise_sd"]*r.standard_normal(n)
        return x,a,m,y,["binary"],["binary","binary","continuous"],"continuous"
    if s.startswith("xia_"):
        x=c["x_sd"]*r.standard_normal((n,1));a=r.binomial(1,expit(c["a_logit_intercept"]+c["a_logit_x"]*x[:,0]))
        if s=="xia_binary": m=r.binomial(1,expit(c["m_logit_intercept"]+c["m_logit_a"]*a+c["m_logit_x"]*x[:,0]))[:,None]
        else: m=(c["m_intercept"]+c["m_a"]*a+c["m_x"]*x[:,0]+c["m_sd"]*r.standard_normal(n))[:,None]
        y=c["y_intercept"]+c["y_a"]*a+(c["y_m"]+c["y_am"]*a)*m[:,0]+c["y_x"]*x[:,0]+c["y_sd"]*r.standard_normal(n)
        return x,a,m,y,["binary" if s=="xia_binary" else "continuous"],["continuous"],"continuous"
    if s=="zuo":
        x=c["x_mean"]+c["x_sd"]*r.standard_normal((n,1));a=r.binomial(1,c["p_t"],n)
        m=(c["alpha_0"]+c["alpha_t"]*a+c["alpha_x"]*x[:,0]+c["m_sd"]*r.standard_normal(n))[:,None]
        y=c["beta_0"]+c["beta_t"]*a+(c["beta_m"]+c["beta_mt"]*a)*m[:,0]+c["beta_x"]*x[:,0]+c["y_sd"]*r.standard_normal(n)
        return x,a,m,y,["continuous"],["continuous"],"continuous"
    raise ValueError(s)


def sample_task(source_id, n, seed, mechanism_seed=None, config=None, randomize=True, compute_truth=True):
    if not isinstance(n,(int,np.integer)) or isinstance(n,bool) or n<2: raise ValueError("n must be integer >=2")
    s=_alias(source_id)
    ms=int(seed if mechanism_seed is None else mechanism_seed)
    c=sample_mechanism(s,ms,randomize=randomize) if config is None else _remove_confounders(s,deepcopy(config))
    x,a,m,y,mt,xt,yt=_draw(s,c,int(n),np.random.default_rng(int(seed)))
    cj=json.dumps(c,sort_keys=True,separators=(",",":"))
    mid=hashlib.sha256((s+":"+cj).encode()).hexdigest()
    return {"x":np.asarray(x,dtype=np.float32),"a":np.asarray(a,dtype=np.float32),
            "m":np.asarray(m,dtype=np.float32),"y":np.asarray(y,dtype=np.float32),
            "truth":truth(s,c) if compute_truth else None,"mediator_types":mt,"x_types":xt,"outcome_type":yt,
            "mechanism_id":mid,"source_group":_FAMILIES[s],
            "metadata":{"source_id":s,"source_group":_FAMILIES[s],"registry_id":_ALIASES[s],
                        "mechanism_id":mid,"mechanism_seed":ms,"table_seed":int(seed),
                        "source_commit":SOURCE_COMMIT,"config":c,"target":"superpopulation_NIE_1",
                        "mediator_types":mt,"x_types":xt,"outcome_type":yt,
                        "adaptation":"full-data; induced-confounder paths removed where applicable"}}


def sample_random_task(n, seed, split="train", compute_truth=True):
    """Family-balanced (not duplicate-record-balanced) task sampling."""
    ids=available_sources(split); families=sorted(set(_FAMILIES[k] for k in ids))
    r=np.random.default_rng(int(seed));family=families[int(r.integers(len(families)))]
    candidates=[k for k in ids if _FAMILIES[k]==family];s=candidates[int(r.integers(len(candidates)))]
    ms=int(r.integers(0,2**31-1));ts=int(r.integers(0,2**31-1))
    out=sample_task(s,n,ts,mechanism_seed=ms,compute_truth=compute_truth);out["metadata"]["split"]=split
    return out


def conditional_effect_sample(source_id, n, seed, config=None):
    """Independent conditional-effect Monte Carlo diagnostic, not a model input."""
    s=_alias(source_id);c=sample_mechanism(s,randomize=False) if config is None else deepcopy(config)
    x,*_= _draw(s,c,int(n),np.random.default_rng(int(seed)))
    return _effect(s,c,x)
