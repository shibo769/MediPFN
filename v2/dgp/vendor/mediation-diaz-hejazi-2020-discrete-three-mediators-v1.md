# Natural indirect effect in the Díaz–Hejazi three-binary-mediator simulation

> **Unverified extraction — scientific review required.**

**Primary area:** Mediation — Direct and Indirect Effects

**Suggested cross-tags (unverified):** none

## Reference paper

**Citation:** Díaz, I. and Hejazi, N. S. (2020). Causal Mediation Analysis for Stochastic Interventions. Journal of the Royal Statistical Society: Series B 82(3), 661–683. DOI: 10.1111/rssb.12362.

**DOI:** 10.1111/rssb.12362 · **Source locator:** Journal §5 Simulation study, unnumbered displayed DGP; compared with arXiv:1901.02776v2 (2019-06-24), PDF p.19 (zero-index page 18), lines 745–785.

## DGP / generation mechanism

**Sampling design:** Unit: individual; Dependence: iid individuals; Z1,Z2,Z3 sampled independently conditional on A,W

**Generator source:** [generator.py](../generators/mediation-diaz-hejazi-2020-discrete-three-mediators-v1.py)

**Keywords (source-described):** Nonlinearity

**Observed variables**

W1, W2, W3, A, Z1, Z2, Z3, Y

**Latent variables**

Not explicitly identified (review required)

**Latent-variable identification:** not explicitly specified

**Equations / distributions (LaTeX rendering)**

> Automatically typeset for readability; English display transcriptions are not changes to the submitted scientific data.

$$
W_{1}\sim Bern(0.50)
$$

$$
W_{2}\sim Bern(0.65)
$$

$$
W_{3}\sim Bern(0.35)
$$

mutually independent

$$
A\mid W\sim Bern((W_{1}+W_{2}+W_{3})/4 + 0.1)
$$

$$
Z_{1}\mid A,W\sim Bern(1-\mathrm{expit}((A+W_{1})/(A+W_{1}+0.5)))
$$

$$
Z_{2}\mid A,W\sim Bern(\mathrm{expit}((A-1+W_{2})/(W_{3}+3)))
$$

$$
Z_{3}\mid A,W\sim Bern(\mathrm{expit}((A-1+2W_{1}-1)/(2W_{1}+0.5)))
$$

$$
Y=Z_{1}+Z_{2}-Z_{3}+A-0.1(W_{1}+W_{2}+W_{3})^{2}+\varepsilon, \varepsilon\sim \mathcal{N}(0,0.25)
$$

<details>
<summary>Original source equations (verbatim)</summary>

<pre>
W1~Bern(0.50), W2~Bern(0.65), W3~Bern(0.35), mutually independent
A|W~Bern((W1+W2+W3)/4 + 0.1)
Z1|A,W~Bern(1-expit((A+W1)/(A+W1+0.5)))
Z2|A,W~Bern(expit((A-1+W2)/(W3+3)))
Z3|A,W~Bern(expit((A-1+2W1-1)/(2W1+0.5)))
Y=Z1+Z2-Z3+A-0.1(W1+W2+W3)^2+epsilon, epsilon~N(0,0.25)
</pre>

</details>


## Parameter scenarios, estimands and truth

| Scenario | Estimand | Definition | True value | Evidence status |
|---|---|---|---|---|
| journal_section_5 | NIE_1 | E[Y(1,Z(1)) - Y(1,Z(0))] for the original §5 superpopulation, jointly intervening on all three mediators Z=(Z1,Z2,Z3). Treatment is fixed at 1 in both outcome terms. | -0.11950701713951144 | reported_numeric |

## Identifiability assumptions

- Consistent well-defined treatment and joint-mediator interventions
- IID individuals and no interference
- Pretreatment W suffices for treatment and mediator-outcome exchangeability under the displayed NPSEM
- Independent mediator disturbance draws conditional on A,W, as in the sequential Bernoulli generator
- Cross-world independence from independent structural disturbances for this source-derived natural effect
- Treatment and joint-mediator positivity at all eight W states

## Provenance and review

**Submitted by:** github_file_upload

**Mode:** structured_zip

**Original versus adapted:** adapted_from_source

**Status:** needs_scientific_review

**Warnings:**

- No explicit latent-variable metadata or reliably identifiable latent declarations; absence from this list does not prove that no latent variables exist.

