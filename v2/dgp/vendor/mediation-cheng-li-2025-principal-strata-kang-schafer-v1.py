"""Observed-data reconstruction of Cheng & Li (2025) Section 5.

Exports only (X1..X4, Z, D, M, Y).  There are no latent variables, and no
counterfactual version of D or M is produced: the printed mechanism does not specify
the joint counterfactual law of (D(1), D(0)), so none is invented here.
"""
import numpy as np

DIMENSION = 4


def _validate(config):
    if int(config["covariate_dimension"]) != DIMENSION:
        raise ValueError("This mechanism is defined for exactly four baseline covariates")
    for key in ("treatment_coefficients", "event_coefficients",
                "mediator_coefficients", "outcome_coefficients"):
        values = config[key]
        if len(values) != DIMENSION or not all(np.isfinite(float(v)) for v in values):
            raise ValueError(f"{key} must contain four finite covariate coefficients")
    for key in ("event_intercept", "event_treatment_coefficient", "mediator_intercept",
                "mediator_treatment_coefficient", "mediator_event_coefficient",
                "outcome_intercept", "outcome_treatment_coefficient",
                "outcome_event_coefficient", "outcome_mediator_coefficient"):
        if not np.isfinite(float(config[key])):
            raise ValueError(f"{key} must be finite")
    for key in ("covariate_sd", "outcome_noise_sd"):
        value = float(config[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive standard deviation")


def expit(values):
    return 1.0 / (1.0 + np.exp(-values))


def generate(config, n, seed):
    _validate(config)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    x = float(config["covariate_sd"]) * rng.standard_normal((n, DIMENSION))

    z = (rng.random(n) < expit(x @ np.asarray(config["treatment_coefficients"], dtype=float))).astype(float)

    event_linear = (float(config["event_intercept"])
                    + float(config["event_treatment_coefficient"]) * z
                    + x @ np.asarray(config["event_coefficients"], dtype=float))
    d = (rng.random(n) < expit(event_linear)).astype(float)

    mediator_linear = (float(config["mediator_intercept"])
                       + float(config["mediator_treatment_coefficient"]) * z
                       + float(config["mediator_event_coefficient"]) * d
                       + x @ np.asarray(config["mediator_coefficients"], dtype=float))
    m = (rng.random(n) < expit(mediator_linear)).astype(float)

    mean_y = (float(config["outcome_intercept"])
              + float(config["outcome_treatment_coefficient"]) * z
              + float(config["outcome_event_coefficient"]) * d
              + float(config["outcome_mediator_coefficient"]) * m
              + x @ np.asarray(config["outcome_coefficients"], dtype=float))
    y = mean_y + float(config["outcome_noise_sd"]) * rng.standard_normal(n)

    return {"X1": x[:, 0].tolist(), "X2": x[:, 1].tolist(),
            "X3": x[:, 2].tolist(), "X4": x[:, 3].tolist(),
            "Z": [int(v) for v in z], "D": [int(v) for v in d],
            "M": [int(v) for v in m], "Y": y.tolist()}
