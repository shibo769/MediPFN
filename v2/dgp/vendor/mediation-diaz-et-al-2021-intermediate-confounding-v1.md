# Interventional indirect effect with a treatment-induced binary confounder

> **Unverified extraction — scientific review required.**

**Primary area:** Mediation — Direct and Indirect Effects

**Suggested cross-tags (unverified):** none

## Reference paper

**Citation:** Díaz, I., Hejazi, N. S., Rudolph, K. E. and van der Laan, M. J. (2021). Non-parametric efficient causal mediation with intermediate confounders. Biometrika 108(3), 627–641. DOI: 10.1093/biomet/asaa085.

**DOI:** 10.1093/biomet/asaa085 · **Source locator:** Section 5 Numerical studies, displayed seven-equation DGP; checked against arXiv:1912.09936v2 PDF p.12 (zero-index p.11), lines 693–706. Interventional effect definition Eq. (2), PDF p.5 (zero-index p.4), and identification Eq. (3), PDF p.6.

## DGP / generation mechanism

**Sampling design:** Unit: individual; Dependence: IID units; W3 depends on W1,W2; A-&gt;Z-&gt;M-&gt;Y within each unit

**Generator source:** [generator.py](../generators/mediation-diaz-et-al-2021-intermediate-confounding-v1.py)

**Keywords (source-described):** Interactions · Nonlinearity

**Observed variables**

W1, W2, W3, A, Z, M, Y

**Latent variables**

Not explicitly identified (review required)

**Latent-variable identification:** not explicitly specified

**Equations / distributions (LaTeX rendering)**

> Automatically typeset for readability; English display transcriptions are not changes to the submitted scientific data.

$$
W_{1}\sim Bern(0.6), W_{2}|W_{1}\sim Bern(0.3), W_{3}|W_{1},W_{2}\sim Bern(0.2+(W_{1}+W_{2})/3)
$$

$$
A\mid W\sim Bern(\mathrm{expit}(0.25 sum(W)+3W_{1}W_{2}-2))
$$

$$
Z\mid A,W\sim Bern(\mathrm{expit}(sum(W)/3-A-AW3-0.25))
$$

$$
M\mid Z,A,W\sim Bern(\mathrm{expit}(W_{1}+W_{2}+A-Z+AZ-0.3AW2))
$$

$$
Y\mid M,Z,A,W\sim Bern(\mathrm{expit}((A-Z+M-AZ)/(sum(W)+1)))
$$

<details>
<summary>Original source equations (verbatim)</summary>

<pre>
W1~Bern(0.6), W2|W1~Bern(0.3), W3|W1,W2~Bern(0.2+(W1+W2)/3)
A|W~Bern(expit(0.25 sum(W)+3W1W2-2))
Z|A,W~Bern(expit(sum(W)/3-A-AW3-0.25))
M|Z,A,W~Bern(expit(W1+W2+A-Z+AZ-0.3AW2))
Y|M,Z,A,W~Bern(expit((A-Z+M-AZ)/(sum(W)+1)))
</pre>

</details>


## Parameter scenarios, estimands and truth

| Scenario | Estimand | Definition | True value | Evidence status |
|---|---|---|---|---|
| section_5 | IIE_1 | E[Y(1,G1)-Y(1,G0)], where G_a is an independent draw from M(a)\|W, marginalizing the post-treatment Z under treatment a. | 0.03026875404999696 | reported_numeric |

## Identifiability assumptions

- IID units and no interference
- Sequential Bernoulli structural sampling as displayed in Section 5
- Well-defined stochastic mediator draws conditional on W
- Treatment and mediator positivity
- Interventional rather than natural indirect effect because Z is treatment-induced mediator-outcome confounder

## Provenance and review

**Submitted by:** github_file_upload

**Mode:** structured_zip

**Original versus adapted:** adapted_from_source

**Status:** needs_scientific_review

**Warnings:**

- No explicit latent-variable metadata or reliably identifiable latent declarations; absence from this list does not prove that no latent variables exist.

