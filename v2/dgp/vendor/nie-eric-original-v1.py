"""Observed-data reconstruction of Section 3. No PFN, oracle labels, or training."""
import math
import random


def expit(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    v = math.exp(value)
    return v / (1.0 + v)


def validate(config):
    sizes = {"x2_coefficients": 2, "x3_coefficients": 3,
             "treatment_coefficients": 5, "mediator_coefficients": 5,
             "outcome_coefficients": 7}
    for key, length in sizes.items():
        arr = config[key]
        if len(arr) != length or not all(math.isfinite(float(v)) for v in arr):
            raise ValueError(f"{key} must contain {length} finite coefficients")
    for x1 in (0, 1):
        p2 = config["x2_coefficients"][0] + config["x2_coefficients"][1] * x1
        if not 0 <= p2 <= 1:
            raise ValueError("P(X2=1|X1) outside [0,1]")
    if not 0 <= config["x1_probability"] <= 1:
        raise ValueError("P(X1=1) outside [0,1]")
    if not all(math.isfinite(float(config[k])) and config[k] > 0
               for k in ("x3_sd", "outcome_noise_sd")):
        raise ValueError("Noise standard deviations must be finite and positive")
    if not math.isfinite(config["mediator_treatment_coefficient"]):
        raise ValueError("Mediator treatment coefficient must be finite")


def generate(config, n, seed):
    validate(config)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = random.Random(seed)
    out = {name: [] for name in ("X1", "X2", "X3", "A", "M", "Y")}
    ac, mc, yc = (config[k] for k in
                  ("treatment_coefficients", "mediator_coefficients", "outcome_coefficients"))
    x2c, x3c = config["x2_coefficients"], config["x3_coefficients"]
    for _ in range(n):
        x1 = int(rng.random() < config["x1_probability"])
        x2 = int(rng.random() < x2c[0] + x2c[1] * x1)
        x3 = rng.gauss(x3c[0] + x3c[1] * x1 + x3c[2] * x2, config["x3_sd"])
        features = (1.0, x1, x2, x3, x1 * x3)
        a = int(rng.random() < expit(sum(c * f for c, f in zip(ac, features))))
        m = int(rng.random() < expit(sum(c * f for c, f in zip(mc, features))
                                    + config["mediator_treatment_coefficient"] * a))
        ymean = sum(c * f for c, f in zip(yc, (1.0, x1, x2, x3, a, m, a * m)))
        y = rng.gauss(ymean, config["outcome_noise_sd"])
        for name, value in zip(out, (x1, x2, x3, a, m, y)):
            out[name].append(value)
    return out
