# Intention-to-treat natural direct effect with a post-treatment event in a Kang-Schafer style design

> **Unverified extraction — scientific review required.**

**Primary area:** Mediation — Direct and Indirect Effects

**Suggested cross-tags (unverified):** none

## Reference paper

**Citation:** Cheng, C. and Li, F. (2025). Identification and multiply robust estimation in causal mediation analysis across principal strata. Journal of the Royal Statistical Society Series B: Statistical Methodology 88(1), 1-22. DOI: 10.1093/jrsssb/qkaf037.

**DOI:** 10.1093/jrsssb/qkaf037 · **Source locator:** Section 5 'A simulation study', arXiv:2304.10025v4 PDF page 23: the displayed data-generating process for (X, Z, D, M, Y). ITT-NIE and ITT-NDE are defined on PDF page 23 just above that display (ratio scale shown there; the paper states it primarily works on the mean-difference scale).

## DGP / generation mechanism

**Sampling design:** Unit: individual; Dependence: iid individuals; X, then Z, then D, then M, then Y are generated in temporal order within an individual

**Generator source:** [generator.py](../generators/mediation-cheng-li-2025-principal-strata-kang-schafer-v1.py)

**Keywords (source-described):** Nonlinearity

**Observed variables**

X1, X2, X3, X4, Z, D, M, Y

**Latent variables**

Not explicitly identified (review required)

**Latent-variable identification:** explicit structured metadata

**Equations / distributions (LaTeX rendering)**

> Automatically typeset for readability; English display transcriptions are not changes to the submitted scientific data.

$$
X = (X_{1}, X_{2}, X_{3}, X_{4}) \sim \mathcal{N}(0_\{4x_{1}\}, I_\{4x_{4}\})
$$

$$
Z \mid  X \sim \mathrm{Bernoulli}(\mathrm{expit}([-1, 0.5, -0.25, -0.1]^T X))
$$

$$
D \mid  Z, X \sim \mathrm{Bernoulli}(\mathrm{expit}(-1 + 2 Z + [1, -0.8, 0.6, -1]^T X))
$$

$$
M \mid  D, Z, X \sim \mathrm{Bernoulli}(\mathrm{expit}(-1.8 + 2 Z + 1.5 D + [1, -0.5, 0.9, -1]^T X))
$$

$$
Y \mid  M, D, Z, X \sim \mathcal{N}(210 + 1.5 Z - D + M + [27.4, 13.7, 13.7, 13.7]^T X, 1)
$$

<details>
<summary>Original source equations (verbatim)</summary>

<pre>
X = (X1, X2, X3, X4) ~ N(0_{4x1}, I_{4x4})
Z | X ~ Bernoulli(expit([-1, 0.5, -0.25, -0.1]^T X))
D | Z, X ~ Bernoulli(expit(-1 + 2 Z + [1, -0.8, 0.6, -1]^T X))
M | D, Z, X ~ Bernoulli(expit(-1.8 + 2 Z + 1.5 D + [1, -0.5, 0.9, -1]^T X))
Y | M, D, Z, X ~ N(210 + 1.5 Z - D + M + [27.4, 13.7, 13.7, 13.7]^T X, 1)
</pre>

</details>


## Parameter scenarios, estimands and truth

| Scenario | Estimand | Definition | True value | Evidence status |
|---|---|---|---|---|
| original | ITT-NDE = E[Y_{1 M_0}] - E[Y_{0 M_0}] | Intention-to-treat natural direct effect of the randomized treatment Z with mediator M, in a design that also contains a post-treatment event D (for example noncompliance). Setting Z = z lets the post-treatment event take D(z), while the mediator is held at M(0). This is NOT the principal-stratum quantity theta^(10)_10 that the source plots, NOT an indirect effect, and NOT a complier average causal effect. | 1.180857104770165 | reported_numeric |

## Identifiability assumptions

- Consistency and well-defined interventions on Z and M
- Independent individuals; no interference
- Randomized Z given X, as encoded by the printed propensity model
- Mediator-outcome exchangeability given (Z, D, X); the post-treatment event D must be adjusted for
- The cross-world condition required for the natural effect contrast
- Positivity: the source deliberately builds a Kang-Schafer style design in which positivity is PRACTICALLY violated under model misspecification, so finite-sample overlap is poor by design even though every probability is strictly inside (0,1) for finite X
- The registered target depends only on the marginal laws of D(1), D(0), M(1) and M(0); it does not require the joint counterfactual law of (D(1), D(0)), which this mechanism does not specify

## Provenance and review

**Submitted by:** github_file_upload

**Mode:** structured_zip

**Original versus adapted:** adapted_from_source

**Status:** needs_scientific_review

**Warnings:**

- Nothing automatically flagged; this does NOT equal scientific acceptance.

