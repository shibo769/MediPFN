"""Independent Python implementation of the displayed JRSS-B §5 DGP."""
import math
import random


def _expit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _mediator_probabilities(a, w):
    w1, w2, w3 = w
    p1 = 1.0 - _expit((a + w1) / (a + w1 + 0.5))
    p2 = _expit((a - 1.0 + w2) / (w3 + 3.0))
    p3 = _expit((a - 1.0 + 2.0 * w1 - 1.0) / (2.0 * w1 + 0.5))
    return p1, p2, p3


def _validate(config):
    p = config["baseline_probabilities"]
    if len(p) != 3 or any(not 0.0 < float(v) < 1.0 for v in p):
        raise ValueError("Three baseline probabilities must lie in (0,1)")
    b = config["outcome_mediator_coefficients"]
    if len(b) != 3 or any(not math.isfinite(float(v)) for v in b):
        raise ValueError("Three finite outcome mediator coefficients required")
    for key in ("treatment_denominator", "treatment_intercept",
                "outcome_treatment_coefficient", "outcome_baseline_quadratic_coefficient",
                "outcome_noise_sd", "mediation_path_scale"):
        if not math.isfinite(float(config[key])):
            raise ValueError(f"{key} must be finite")
    if config["treatment_denominator"] <= 0 or config["outcome_noise_sd"] <= 0:
        raise ValueError("Positive treatment denominator and outcome noise SD required")
    if not 0 <= config["mediation_path_scale"] <= 1:
        raise ValueError("mediation_path_scale must lie in [0,1]")
    for total in range(4):
        pa = total / config["treatment_denominator"] + config["treatment_intercept"]
        if not 0 < pa < 1:
            raise ValueError("Treatment propensity outside (0,1)")


def generate(config, n, seed):
    _validate(config)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = random.Random(seed)
    out = {name: [] for name in ("W1", "W2", "W3", "A", "Z1", "Z2", "Z3", "Y")}
    for _ in range(n):
        w = tuple(int(rng.random() < p) for p in config["baseline_probabilities"])
        a_prob = sum(w) / config["treatment_denominator"] + config["treatment_intercept"]
        a = int(rng.random() < a_prob)
        probs = _mediator_probabilities(config["mediation_path_scale"] * a, w)
        z = tuple(int(rng.random() < p) for p in probs)
        mean_y = (sum(c * v for c, v in zip(config["outcome_mediator_coefficients"], z))
                  + config["outcome_treatment_coefficient"] * a
                  + config["outcome_baseline_quadratic_coefficient"] * sum(w) ** 2)
        y = rng.gauss(mean_y, config["outcome_noise_sd"])
        for name, value in zip(out, (*w, a, *z, y)):
            out[name].append(value)
    return out
