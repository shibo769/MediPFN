# Path-specific effect along A to M to Y with a treatment-induced mediator-outcome confounder vector

> **Unverified extraction — scientific review required.**

**Primary area:** Mediation — Direct and Indirect Effects

**Suggested cross-tags (unverified):** none

## Reference paper

**Citation:** Miles, C. H., Shpitser, I., Kanki, P., Meloni, S. and Tchetgen Tchetgen, E. J. (2019). On semiparametric estimation of a path-specific effect in the presence of mediator-outcome confounding. Biometrika 107(1), 159-172. DOI: 10.1093/biomet/asz063.

**DOI:** 10.1093/biomet/asz063 · **Source locator:** Section 5 'Simulation study', arXiv:1710.02011v1 PDF pages 12-13: the displayed data-generating mechanism for (C0, A, C1, M, Y). Estimand beta_0 and the path-specific effect P_AMY are defined in Section 2, PDF pages 4-5; the identification formula is equation (1), PDF page 5.

## DGP / generation mechanism

**Sampling design:** Unit: individual; Dependence: iid individuals; C0, then A, then the three-vector C1, then M, then Y are generated in temporal order within an individual

**Generator source:** [generator.py](../generators/mediation-miles-2019-toxicity-path-specific-v1.py)

**Keywords (source-described):** Interactions · Nonlinearity

**Observed variables**

C0, A, C11, C12, C13, M, Y

**Latent variables**

Not explicitly identified (review required)

**Latent-variable identification:** explicit structured metadata

**Equations / distributions (LaTeX rendering)**

> Automatically typeset for readability; English display transcriptions are not changes to the submitted scientific data.

$$
C_{0} \sim \mathrm{Uniform}(0, 2)
$$

$$
A \mid  C_{0} \sim \mathrm{Bernoulli}(1 - \{1 + \exp(0.9 + 0.3 C_{0})\}^\{-1\}) = \mathrm{Bernoulli}(\mathrm{expit}(0.9 + 0.3 C_{0}))
$$

$$
C_{1} = (0.8, 0.6, -0.3)^T + (1, 0.1, 0.2)^T C_{0} + (0.5, -0.4, 0.5)^T A + (-0.1, 0.8, -0.2)^T C_{0} A + \mathcal{N}(0, I_3)
$$

$$
M = -0.5 - 0.2 C_{0} + 0.3 A + [-0.2, 0.1, 0.5] C_{1} + [0.4, 0, 0] A C_{1} + \mathcal{N}(0, 1)
$$

$$
Y = 0.2 + 0.2 C_{0} + 0.6 A + [1, 0.7, 0.3] C_{1} - 0.9 M - 0.8 A M + \mathcal{N}(0, 1)
$$

$$
\beta_0 = int int int \mathbb{E}(Y|m, c_{1}, a', c_{0}) dF(m|c_{1}, a, c_{0}) dF(c_{1}|a', c_{0}) dF(c_{0})
$$

<details>
<summary>Original source equations (verbatim)</summary>

<pre>
C0 ~ Uniform(0, 2)
A | C0 ~ Bernoulli(1 - {1 + exp(0.9 + 0.3 C0)}^{-1}) = Bernoulli(expit(0.9 + 0.3 C0))
C1 = (0.8, 0.6, -0.3)^T + (1, 0.1, 0.2)^T C0 + (0.5, -0.4, 0.5)^T A + (-0.1, 0.8, -0.2)^T C0 A + N(0, I_3)
M = -0.5 - 0.2 C0 + 0.3 A + [-0.2, 0.1, 0.5] C1 + [0.4, 0, 0] A C1 + N(0, 1)
Y = 0.2 + 0.2 C0 + 0.6 A + [1, 0.7, 0.3] C1 - 0.9 M - 0.8 A M + N(0, 1)
beta_0 = int int int E(Y|m, c1, a&#x27;, c0) dF(m|c1, a, c0) dF(c1|a&#x27;, c0) dF(c0)
</pre>

</details>


## Parameter scenarios, estimands and truth

| Scenario | Estimand | Definition | True value | Evidence status |
|---|---|---|---|---|
| original | P_AMY = E(Y[M{a, C1(a')}, C1(a'), a']) - E[Y(a')] | The path-specific effect of A on Y along the path A -&gt; M -&gt; Y, as defined in Section 2 of the source. It is the CONTRAST of two nested counterfactual means: in both arms the treatment-induced confounder vector C1 and the outcome are held at the reference level a' = 0, and only the exposure level fed to the mediator equation moves, from a' = 0 to a = 1. Writing beta_0 = E(Y[M{a, C1(a')}, C1(a'), a']) for the moved arm and E[Y(a')] for the unmoved arm, P_AMY = beta_0 - E[Y(a')]. beta_0 on its own is a counterfactual MEAN, not an effect, and is registered only as a component; the source's Figure 2 plots beta_0, which is why that figure is used as coding evidence rather than as the target. P_AMY is NOT a natural indirect effect, NOT a natural direct effect and NOT a total effect; keeping C1 at a' inside the mediator counterfactual is what distinguishes it from the standard natural effects. | -0.9180000000000001 | reported_numeric |

## Identifiability assumptions

- Consistency, and nested counterfactuals are well defined, as assumed by the source
- Independent and identically distributed individuals; no interference
- No unmeasured confounding of the exposure-outcome and mediator-outcome relationships, i.e. the relaxed Markovian model of the source's Figure 1(b)
- Positivity as stated by the source: the conditional densities of M given (C1, A, C0) and of C1 given (A, C0) are bounded away from zero almost everywhere, and 0 &lt; f(a\|C0) &lt; 1 almost everywhere
- C1 is a treatment-induced mediator-outcome confounder, so it must be adjusted for in the outcome regression and must be kept at its a'-counterfactual value inside the nested expectation
- Positivity for the exposure holds pointwise: expit(0.9 + 0.3 C0) lies in (0.71, 0.82) over the support C0 in [0, 2]

## Provenance and review

**Submitted by:** github_file_upload

**Mode:** structured_zip

**Original versus adapted:** adapted_from_source

**Status:** needs_scientific_review

**Warnings:**

- Nothing automatically flagged; this does NOT equal scientific acceptance.

