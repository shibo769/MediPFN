# Sparse natural indirect effect with ten candidate mediators, scenario LNL (linear propensity, nonlinear mediator confounding, linear outcome confounding)

> **Unverified extraction — scientific review required.**

**Primary area:** Mediation — Direct and Indirect Effects

**Suggested cross-tags (unverified):** none

## Reference paper

**Citation:** Jones, J., Ertefaie, A. and Strawderman, R. L. (2025). Causal mediation analysis: selection with asymptotically valid inference. Journal of the Royal Statistical Society Series B: Statistical Methodology 87(3), 678-700. DOI: 10.1093/jrsssb/qkae109.

**DOI:** 10.1093/jrsssb/qkae109 · **Source locator:** Section 5 'Simulation Studies', arXiv:2110.06127v2 PDF page 15 (the two forms of mu_D0, psi_M and psi_Y, and the three-letter scenario naming; this contribution registers LNL), and Appendix A 'Simulation Results', PDF page 26 (all variances, dimensions and the structural parameters alpha_0, beta_0, gamma_0, and the statement NIE = 2.4). Structural model (1) is on PDF page 4; Proposition 2.1 giving NDE = gamma_0 and NIE = alpha_0 . beta_0 is on PDF page 6.

## DGP / generation mechanism

**Sampling design:** Unit: individual; Dependence: iid individuals; X, then D, then the ten-dimensional mediator vector, then Y are generated in temporal order within an individual

**Generator source:** [generator.py](../generators/mediation-jones-2024-sparse-mediators-lnl-v1.py)

**Keywords (source-described):** Nonlinearity

**Observed variables**

X1, X2, X3, D, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, Y

**Latent variables**

Not explicitly identified (review required)

**Latent-variable identification:** explicit structured metadata

**Equations / distributions (LaTeX rendering)**

> Automatically typeset for readability; English display transcriptions are not changes to the submitted scientific data.

$$
X_l \sim \mathcal{N}(0, 1/4) independently, l = 1, 2, 3
$$

mu_D0(X) = expit{0.8 (X1 + X2)}

$$
psi_M(X) = X_{1}^{2} + X_{2} - X_{3}
$$

$$
psi_Y(X) = 2 (X_{1} - 0.5) + X_{2} + 2 X_{3}
$$

D | X ~ Bernoulli(mu_D0(X))

$$
M = \alpha_0 D + psi_M(X) 1_p + eta, eta_j \sim \mathcal{N}(0, 1) independently, p = 10, \alpha_0 = (1, 2, 2, 0, \ldots , 0)
$$

$$
Y = \gamma_0 D + \beta_0' M + psi_Y(X) + \varepsilon, \varepsilon \sim \mathcal{N}(0, 1)
$$

$$
\gamma_0 = 2, \beta_0 = (0.8, 0.4, 0.4, 0, \ldots , 0)
$$

$$
NDE = \gamma_0 = 2
$$

$$
NIE = \alpha_0 . \beta_0 = 2.4
$$

<details>
<summary>Original source equations (verbatim)</summary>

<pre>
X_l ~ N(0, 1/4) independently, l = 1, 2, 3
mu_D0(X) = expit{0.8 (X1 + X2)}
psi_M(X) = X1^2 + X2 - X3
psi_Y(X) = 2 (X1 - 0.5) + X2 + 2 X3
D | X ~ Bernoulli(mu_D0(X))
M = alpha_0 D + psi_M(X) 1_p + eta, eta_j ~ N(0, 1) independently, p = 10, alpha_0 = (1, 2, 2, 0, ..., 0)
Y = gamma_0 D + beta_0&#x27; M + psi_Y(X) + epsilon, epsilon ~ N(0, 1), gamma_0 = 2, beta_0 = (0.8, 0.4, 0.4, 0, ..., 0)
NDE = gamma_0 = 2; NIE = alpha_0 . beta_0 = 2.4
</pre>

</details>


## Parameter scenarios, estimands and truth

| Scenario | Estimand | Definition | True value | Evidence status |
|---|---|---|---|---|
| LNL | NIE = alpha_0 . beta_0 | Natural indirect effect through the sparse mediator set: NIE = E[Y(1, M(1))] - E[Y(1, M(0))]. Under the source's additive structural model (its equation (1)) and its Proposition 2.1 this equals the inner product alpha_0 . beta_0 of the treatment-to-mediator and mediator-to-outcome coefficient vectors. The companion natural direct effect equals gamma_0. This is NOT a total effect and NOT a single-mediator indirect effect; ten candidate mediators are generated and only the first three are true mediators. | 2.4000000000000004 | reported_numeric |

## Identifiability assumptions

- Consistency and well-defined joint interventions on the treatment and the mediator vector
- Independent individuals; no interference
- Treatment exchangeability given X and mediator-outcome exchangeability given (X, D), as encoded by the source's additive model and its Assumptions 2.1 and 2.2
- The source's sparsity assumptions: the first three mediators form the good mediator set S*, and no mediator outside S* is related to any variable in S*
- Mean-zero errors conditional on the preceding variables: E(eta\|X,D) = 0 and E(epsilon\|X,D,M) = 0
- Treatment positivity: the source states it deliberately chose the small covariate variance 1/4 to minimise the chance of inadvertent positivity violations; expit of a bounded linear or product term stays strictly inside (0,1) for every finite X, but no uniform overlap bound is claimed on unbounded Gaussian support
- The source notes that under slightly more general assumptions its estimands also admit an interventional-effect interpretation; the registered definition follows the natural-effect reading the source adopts

## Provenance and review

**Submitted by:** github_file_upload

**Mode:** structured_zip

**Original versus adapted:** adapted_from_source

**Status:** needs_scientific_review

**Warnings:**

- Nothing automatically flagged; this does NOT equal scientific acceptance.

