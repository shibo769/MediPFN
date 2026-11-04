"""Observed-data reconstruction of Shan, Li and Ai (2026) Section 5.

Only observed columns are exported.  The exogenous normals (eps1, eps1', eps2, eps2',
eps3, eps4) and the complete covariates X1, X2 stay internal: a partially observed
covariate leaves this module only as the pair (R_j, Xj_obs0) with
Xj_obs0 = R_j * Xj, so a unit whose Xj is missing carries the declared fill value and
never its latent value.  No counterfactual, oracle or potential-outcome column is
returned.
"""
import math

import numpy as np

_ERF = np.vectorize(math.erf, otypes=[float])

MISSINGNESS_FORMS = ("dgp1_residual_dependent", "dgp2_covariate_treatment_dependent")
SHADOW_KINDS = ("auxiliary", "internal_mediator_outcome")


def _normal_cdf(values):
    """Phi, the standard normal distribution function, via the error function."""
    return 0.5 * (1.0 + _ERF(np.asarray(values, dtype=float) / math.sqrt(2.0)))


def _finite(config, key):
    value = float(config[key])
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _validate(config):
    form = config["missingness_form"]
    if form not in MISSINGNESS_FORMS:
        raise ValueError(f"missingness_form must be one of {MISSINGNESS_FORMS}")
    kind = config["shadow_variables"]
    if kind not in SHADOW_KINDS:
        raise ValueError(f"shadow_variables must be one of {SHADOW_KINDS}")
    if (form == "dgp1_residual_dependent") != (kind == "auxiliary"):
        raise ValueError(
            "the source pairs the residual-dependent missingness of DGP1 with external "
            "auxiliary shadow variables, and the covariate/treatment-dependent "
            "missingness of DGP2 with (M, Y) as shadow variables"
        )
    probability = _finite(config, "x3_probability")
    if not 0.0 < probability < 1.0:
        raise ValueError("x3_probability must lie strictly inside (0, 1)")
    for key in ("mediator_noise_sd", "outcome_noise_sd"):
        if _finite(config, key) <= 0.0:
            raise ValueError(f"{key} must be a positive standard deviation")
    count = config["incomplete_pattern_count"]
    if count != 3:
        raise ValueError(
            "the source's missingness pattern set is {(1,0), (0,1), (0,0)}, so "
            "incomplete_pattern_count is structurally 3 and is not a free parameter"
        )
    if _finite(config, "missing_fill_value") != 0.0:
        raise ValueError("missing_fill_value is 0.0 in this registration")
    keys = ["propensity_intercept", "propensity_x1_squared", "propensity_x2",
            "propensity_x3", "mediator_intercept", "mediator_treatment",
            "mediator_treatment_x1_squared", "mediator_x1_squared", "mediator_sin_x2",
            "mediator_x3", "outcome_intercept", "outcome_treatment",
            "outcome_treatment_mediator", "outcome_mediator", "outcome_treatment_x1",
            "outcome_sin_x1", "outcome_x2_squared", "outcome_x3",
            "missingness_x1", "missingness_x3"]
    if form == "dgp1_residual_dependent":
        keys += ["missingness_x2", "missingness_treatment_times_mediator_noise",
                 "missingness_outcome_noise", "shadow_correlation"]
    else:
        keys += ["missingness_intercept", "missingness_x2_squared",
                 "missingness_treatment", "missingness_treatment_x1_squared"]
    for key in keys:
        _finite(config, key)
    if form == "dgp1_residual_dependent":
        alpha = _finite(config, "shadow_correlation")
        if not -1.0 < alpha < 1.0:
            raise ValueError("shadow_correlation must lie strictly inside (-1, 1)")


def _missingness_index(config, x1, x2, x3, treatment, eps3, eps4):
    """U, the index that drives the multinomial missingness pattern probabilities."""
    if config["missingness_form"] == "dgp1_residual_dependent":
        return (config["missingness_x1"] * x1
                + config["missingness_x2"] * x2
                + config["missingness_x3"] * x3
                + config["missingness_treatment_times_mediator_noise"] * treatment * eps3
                + config["missingness_outcome_noise"] * eps4)
    return (config["missingness_intercept"]
            + config["missingness_x1"] * x1
            + config["missingness_x2_squared"] * x2 ** 2
            + config["missingness_x3"] * x3
            + config["missingness_treatment"] * treatment
            + config["missingness_treatment_x1_squared"] * treatment * x1 ** 2)


def _pattern_probabilities(config, index):
    """(p11, p10, p01, p00).  p11 / p_kl = exp(U) and p11 + 3 p_kl = 1 identically."""
    count = float(config["incomplete_pattern_count"])
    complete = 1.0 / (1.0 + count * np.exp(-index))
    incomplete = 1.0 / (np.exp(index) + count)
    return complete, incomplete


def generate(config, n, seed):
    _validate(config)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.Generator(np.random.PCG64(seed))

    # Exogenous normals, in the order the source lists them.
    eps1 = rng.standard_normal(n)
    eps1_prime = rng.standard_normal(n)
    eps2 = rng.standard_normal(n)
    eps2_prime = rng.standard_normal(n)
    eps3 = rng.standard_normal(n)
    eps4 = rng.standard_normal(n)

    x1 = _normal_cdf(eps1)                                  # Uniform(0, 1) marginally
    x2 = _normal_cdf(eps2)
    x3 = (rng.random(n) < float(config["x3_probability"])).astype(float)

    logit = (config["propensity_intercept"]
             + config["propensity_x1_squared"] * x1 ** 2
             + config["propensity_x2"] * x2
             + config["propensity_x3"] * x3)
    treatment = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)

    mediator = (config["mediator_intercept"]
                + config["mediator_treatment"] * treatment
                + config["mediator_treatment_x1_squared"] * treatment * x1 ** 2
                + config["mediator_x1_squared"] * x1 ** 2
                + config["mediator_sin_x2"] * np.sin(x2)
                + config["mediator_x3"] * x3
                + float(config["mediator_noise_sd"]) * eps3)

    outcome = (config["outcome_intercept"]
               + config["outcome_treatment"] * treatment
               + config["outcome_treatment_mediator"] * treatment * mediator
               + config["outcome_mediator"] * mediator
               + config["outcome_treatment_x1"] * treatment * x1
               + config["outcome_sin_x1"] * np.sin(x1)
               + config["outcome_x2_squared"] * x2 ** 2
               + config["outcome_x3"] * x3
               + float(config["outcome_noise_sd"]) * eps4)

    index = _missingness_index(config, x1, x2, x3, treatment, eps3, eps4)
    complete, incomplete = _pattern_probabilities(config, index)
    # Inverse-CDF draw over the patterns (R1,R2) = (1,1), (1,0), (0,1), (0,0).
    draw = rng.random(n)
    edges = np.column_stack([complete,
                             complete + incomplete,
                             complete + 2.0 * incomplete])
    pattern = np.sum(draw[:, None] >= edges, axis=1)        # 0, 1, 2 or 3
    r1 = np.isin(pattern, (0, 1)).astype(float)
    r2 = np.isin(pattern, (0, 2)).astype(float)

    fill = float(config["missing_fill_value"])
    x1_observed = np.where(r1 > 0.5, x1, fill)
    x2_observed = np.where(r2 > 0.5, x2, fill)

    columns = {
        "R1": [int(v) for v in r1],
        "R2": [int(v) for v in r2],
        "X1_obs0": x1_observed.tolist(),
        "X2_obs0": x2_observed.tolist(),
        "X3": [int(v) for v in x3],
    }
    if config["shadow_variables"] == "auxiliary":
        alpha = float(config["shadow_correlation"])
        beta = math.sqrt(1.0 - alpha * alpha)
        columns["Z1"] = _normal_cdf(alpha * eps1 + beta * eps1_prime).tolist()
        columns["Z2"] = _normal_cdf(alpha * eps2 + beta * eps2_prime).tolist()
    columns["T"] = [int(v) for v in treatment]
    columns["M"] = mediator.tolist()
    columns["Y"] = outcome.tolist()
    return columns
