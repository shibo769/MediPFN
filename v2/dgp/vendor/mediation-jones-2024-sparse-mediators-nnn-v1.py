"""Observed-data reconstruction of Jones, Ertefaie & Strawderman (2024) Section 5 / Appendix A.

Exports only (X1, X2, X3, D, M1..M10, Y).  There are no latent variables; the
counterfactual mediator vectors M(1) and M(0) exist only inside the population target.
"""
import numpy as np

COVARIATES = 3


def _validate(config):
    if int(config["covariate_count"]) != COVARIATES:
        raise ValueError("This mechanism is defined for exactly three baseline confounders")
    width = int(config["mediator_count"])
    if width < 1:
        raise ValueError("mediator_count must be positive")
    for key in ("alpha0", "beta0"):
        if len(config[key]) != width:
            raise ValueError(f"{key} must have one entry per candidate mediator")
    if config["propensity_form"] not in ("linear", "nonlinear"):
        raise ValueError("propensity_form must be 'linear' or 'nonlinear'")
    for key in ("mediator_confounding_form", "outcome_confounding_form"):
        if config[key] not in ("linear", "nonlinear"):
            raise ValueError(f"{key} must be 'linear' or 'nonlinear'")
    if float(config["covariate_variance"]) <= 0:
        raise ValueError("covariate_variance must be positive")
    for key in ("mediator_noise_sd", "outcome_noise_sd"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")


def expit(values):
    return 1.0 / (1.0 + np.exp(-values))


def propensity(config, x):
    scale = float(config["propensity_scale"])
    if config["propensity_form"] == "linear":
        return expit(scale * (x[:, 0] + x[:, 1]))
    return expit(scale * (x[:, 0] * x[:, 1] + x[:, 1]))


def mediator_confounding(config, x):
    if config["mediator_confounding_form"] == "linear":
        return x[:, 0] + x[:, 1] - x[:, 2]
    return x[:, 0] ** 2 + x[:, 1] - x[:, 2]


def outcome_confounding(config, x):
    if config["outcome_confounding_form"] == "linear":
        return 2.0 * (x[:, 0] - 0.5) + x[:, 1] + 2.0 * x[:, 2]
    return 2.0 * (x[:, 0] - 0.5) ** 2 + x[:, 1] + 2.0 * x[:, 2]


def generate(config, n, seed):
    _validate(config)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    width = int(config["mediator_count"])
    scale = np.sqrt(float(config["covariate_variance"]))
    x = scale * rng.standard_normal((n, COVARIATES))

    d = (rng.random(n) < propensity(config, x)).astype(float)

    alpha0 = np.asarray(config["alpha0"], dtype=float)
    beta0 = np.asarray(config["beta0"], dtype=float)
    # psi_M is scalar valued and added to every mediator coordinate (the source's 1_p).
    m = (np.outer(d, alpha0)
         + mediator_confounding(config, x)[:, None]
         + float(config["mediator_noise_sd"]) * rng.standard_normal((n, width)))

    y = (float(config["gamma0"]) * d + m @ beta0 + outcome_confounding(config, x)
         + float(config["outcome_noise_sd"]) * rng.standard_normal(n))

    columns = {"X1": x[:, 0].tolist(), "X2": x[:, 1].tolist(), "X3": x[:, 2].tolist(),
               "D": [int(v) for v in d]}
    for index in range(width):
        columns[f"M{index + 1}"] = m[:, index].tolist()
    columns["Y"] = y.tolist()
    return columns
