"""Observed-data reconstruction of Miles et al. (2019) Section 5.

Exports only (C0, A, C11, C12, C13, M, Y).  There are no latent variables, and no
counterfactual version of C1 or M is produced; the nested counterfactual lives only in
the population target.
"""
import numpy as np

CONFOUNDER_DIMENSION = 3


def _validate(config):
    low, high = config["baseline_support"]
    if not float(low) < float(high):
        raise ValueError("baseline_support must be an increasing interval")
    if len(config["exposure_coefficients"]) != 2:
        raise ValueError("exposure_coefficients must hold (intercept, C0 slope)")
    for key in ("confounder_intercept", "confounder_baseline", "confounder_exposure",
                "confounder_interaction", "mediator_confounder",
                "mediator_exposure_confounder", "outcome_confounder"):
        if len(config[key]) != CONFOUNDER_DIMENSION:
            raise ValueError(f"{key} must hold three coefficients")
    for key in ("confounder_sd", "mediator_sd", "outcome_sd"):
        value = float(config[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive standard deviation")
    if {int(config["reference_level"]), int(config["comparison_level"])} - {0, 1}:
        raise ValueError("reference_level and comparison_level must be 0 or 1")


def expit(values):
    return 1.0 / (1.0 + np.exp(-values))


def confounder_mean(config, baseline, exposure):
    return (np.asarray(config["confounder_intercept"], dtype=float)
            + np.outer(baseline, np.asarray(config["confounder_baseline"], dtype=float))
            + np.outer(exposure, np.asarray(config["confounder_exposure"], dtype=float))
            + np.outer(baseline * exposure, np.asarray(config["confounder_interaction"], dtype=float)))


def mediator_mean(config, baseline, exposure, confounder):
    weights = (np.asarray(config["mediator_confounder"], dtype=float)
               + exposure[:, None] * np.asarray(config["mediator_exposure_confounder"], dtype=float))
    return (float(config["mediator_intercept"])
            + float(config["mediator_baseline"]) * baseline
            + float(config["mediator_exposure"]) * exposure
            + np.sum(weights * confounder, axis=1))


def outcome_mean(config, baseline, exposure, confounder, mediator):
    return (float(config["outcome_intercept"])
            + float(config["outcome_baseline"]) * baseline
            + float(config["outcome_exposure"]) * exposure
            + confounder @ np.asarray(config["outcome_confounder"], dtype=float)
            + float(config["outcome_mediator"]) * mediator
            + float(config["outcome_exposure_mediator"]) * exposure * mediator)


def generate(config, n, seed):
    _validate(config)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    low, high = (float(v) for v in config["baseline_support"])
    c0 = rng.uniform(low, high, n)

    alpha = config["exposure_coefficients"]
    a = (rng.random(n) < expit(alpha[0] + alpha[1] * c0)).astype(float)

    c1 = confounder_mean(config, c0, a) + float(config["confounder_sd"]) * rng.standard_normal((n, CONFOUNDER_DIMENSION))
    m = mediator_mean(config, c0, a, c1) + float(config["mediator_sd"]) * rng.standard_normal(n)
    y = outcome_mean(config, c0, a, c1, m) + float(config["outcome_sd"]) * rng.standard_normal(n)

    return {"C0": c0.tolist(), "A": [int(v) for v in a],
            "C11": c1[:, 0].tolist(), "C12": c1[:, 1].tolist(), "C13": c1[:, 2].tolist(),
            "M": m.tolist(), "Y": y.tolist()}
