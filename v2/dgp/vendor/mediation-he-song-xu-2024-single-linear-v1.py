"""Observed-data reconstruction of He–Song–Xu equation (11)."""
import math
import random


def generate(config, n, seed):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be positive integer")
    for key in ("x1_sd", "mediator_noise_sd", "outcome_noise_sd"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive finite")
    for key in ("x2_probability", "treatment_probability"):
        if not 0 < config[key] < 1:
            raise ValueError(f"{key} must lie in (0,1)")
    rng = random.Random(seed)
    out = {name: [] for name in ("X1", "X2", "A", "M", "Y")}
    for _ in range(n):
        x1 = rng.gauss(0.0, config["x1_sd"])
        x2 = int(rng.random() < config["x2_probability"])
        a = int(rng.random() < config["treatment_probability"])
        m = (config["alpha_s"] * a + config["mediator_intercept"]
             + config["mediator_x1"] * x1 + config["mediator_x2"] * x2
             + rng.gauss(0.0, config["mediator_noise_sd"]))
        y = (config["beta_m"] * m + config["outcome_intercept"]
             + config["outcome_x1"] * x1 + config["outcome_x2"] * x2
             + config["outcome_direct"] * a + rng.gauss(0.0, config["outcome_noise_sd"]))
        for name, value in zip(out, (x1, x2, a, m, y)):
            out[name].append(value)
    return out
