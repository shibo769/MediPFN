# Mediation functional under nonignorable missing confounders with covariate- and treatment-dependent missingness and internal shadow variables

> **Unverified extraction — scientific review required.**

**Primary area:** Mediation — Direct and Indirect Effects

**Suggested cross-tags (unverified):** none

## Reference paper

**Citation:** Shan, J., Li, W. and Ai, C. (2026). Efficient Nonparametric Inference for Mediation Analysis with Nonignorable Missing Confounders. Journal of the American Statistical Association, published online 25 June 2026, pages 1-12. DOI: 10.1080/01621459.2026.2654218.

**DOI:** 10.1080/01621459.2026.2654218 · **Source locator:** Section 5 'Numerical studies', the displayed data-generating process and the DGP2 branch; arXiv:2402.05384 PDF page 14. Estimand definitions: Section 2.1, PDF page 4 (NIE xi(t), NDE zeta(t), ATE tau) and the mediation functional theta = E[Y(1, M(0))] with its identification formula (1) on PDF page 5. Figure 2 and its printed axis ranges: PDF page 15. Reported missingness split of about 58/14/14/14 per cent: PDF page 14. Page numbers are preprint PDF pages; the article is online first and carries no journal page numbers beyond 1-12.

## DGP / generation mechanism

**Sampling design:** Unit: independent individual; Dependence: i.i.d. across units; no interference, no clustering and no panel structure. Missingness patterns are drawn independently across units given each unit's own covariates, treatment and noises.

**Generator source:** [generator.py](../generators/mediation-shan-li-ai-2026-missing-confounders-internal-shadow-v1.py)

**Keywords (source-described):** Interactions · Nonlinearity

**Observed variables**

R1, R2, X1_obs0, X2_obs0, X3, T, M, Y

**Latent variables**

Not explicitly identified (review required)

**Latent-variable identification:** not explicitly specified

**Equations / distributions (LaTeX rendering)**

> Automatically typeset for readability; English display transcriptions are not changes to the submitted scientific data.

$$
X_{1} = Phi(eps1)
$$

$$
X_{2} = Phi(eps2) (both \mathrm{Uniform}(0,1))
$$

$$
X_{3} \sim \mathrm{Bernoulli}(0.5)
$$

$$
T \sim \mathrm{Bernoulli}(\mathrm{expit}(0.1 + X_{1}^{2} - X_{2} + 0.2 X_{3}))
$$

$$
M = -1 + T - T X_{1}^{2} + 4 X_{1}^{2} - \sin(X_{2}) + X_{3} + eps3
$$

$$
Y = -1 + T + 3 T M - 1.5 M + T X_{1} + 5 \sin(X_{1}) - 2 X_{2}^{2} + X_{3} + eps4
$$

$$
(R_{1},R_{2}) \sim Multinomial(p_{11}, p_{10}, p_{01}, p_{00})
$$

$$
p_{11} = 1/(1 + 3 \exp(-U))
$$

$$
p_{10} = p_{01} = p_{00} = 1/(\exp(U) + 3)
$$

$$
\theta = \mathbb{E}[Y(1, M(0))] = 3.192275262794844
$$

<details>
<summary>Original source equations (verbatim)</summary>

<pre>
X1 = Phi(eps1), X2 = Phi(eps2) (both Uniform(0,1)), X3 ~ Bernoulli(0.5)
T ~ Bernoulli(expit(0.1 + X1^2 - X2 + 0.2 X3))
M = -1 + T - T X1^2 + 4 X1^2 - sin(X2) + X3 + eps3
Y = -1 + T + 3 T M - 1.5 M + T X1 + 5 sin(X1) - 2 X2^2 + X3 + eps4
(R1,R2) ~ Multinomial(p11, p10, p01, p00), p11 = 1/(1 + 3 exp(-U)), p10 = p01 = p00 = 1/(exp(U) + 3)
theta = E[Y(1, M(0))] = 3.192275262794844
</pre>

</details>


## Parameter scenarios, estimands and truth

| Scenario | Estimand | Definition | True value | Evidence status |
|---|---|---|---|---|
| dgp2_internal_shadow | MF theta = E[Y(1, M(0))] | Mediation functional: the mean potential outcome when treatment is set to 1 while the mediator is set to the value it would have taken under treatment 0, theta = E[Y(1, M(0))]. The source calls this the fundamental quantity and derives the natural indirect effect xi(1) = E[Y(1,M(1))] - theta, the natural direct effect zeta(0) = theta - E[Y(0,M(0))] and the total effect tau = xi(1) + zeta(0) from it. theta is a cross-world mean and is NOT an indirect effect, NOT a direct effect and NOT a total effect. | 3.192275262794844 | reported_numeric |

## Identifiability assumptions

- Consistency: Y = Y(T, M(T)) and M = M(T), as the source states before its Assumption 1.
- Sequential ignorability (source Assumption 1): {Y(t,m), M(t')} independent of T given X, and Y(t',m) independent of M(t) given T = t and X. This is a cross-world condition and is not testable from the data.
- Positivity (source Assumption 2): f(t\|x) &gt; 0 and f(m\|t,x) &gt; 0. The logistic propensity is bounded inside (0,1) on the compact covariate support, but no uniform overlap bound is claimed for the complete-case missingness pattern.
- Independent individuals; no interference, no grouping and no panel structure.
- The registered target is a FULL-DATA functional. Whether it is identifiable from the observed data requires the source's shadow-variable condition, which this contribution does not assert and does not verify.

## Provenance and review

**Submitted by:** github_file_upload

**Mode:** structured_zip

**Original versus adapted:** adapted_from_source

**Status:** needs_scientific_review

**Warnings:**

- No explicit latent-variable metadata or reliably identifiable latent declarations; absence from this list does not prove that no latent variables exist.

