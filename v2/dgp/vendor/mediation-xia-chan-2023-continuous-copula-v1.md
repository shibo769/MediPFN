# Continuous Gaussian-copula mediator and treatment-induced confounder

> **Unverified extraction — scientific review required.**

**Primary area:** Mediation — Direct and Indirect Effects

**Suggested cross-tags (unverified):** none

## Reference paper

**Citation:** Xia, F. and Chan, K. C. G. (2023). Identification, Semiparametric Efficiency, and Quadruply Robust Estimation in Mediation Analysis with Treatment-Induced Confounding. Journal of the American Statistical Association 118(542), 1272–1281. DOI: 10.1080/01621459.2021.1990765.

**DOI:** 10.1080/01621459.2021.1990765 · **Source locator:** Accepted manuscript §4.1, continuous C and M paragraph, PDF p.17; https://par.nsf.gov/servlets/purl/10300663.

## DGP / generation mechanism

**Sampling design:** Unit: individual; Dependence: IID rows; within each row C and M residuals have Gaussian correlation 0.2

**Generator source:** [generator.py](../generators/mediation-xia-chan-2023-continuous-copula-v1.py)

**Keywords (source-described):** Interactions · Nonlinearity

**Observed variables**

X, A, C, M, Y

**Latent variables**

E_M, E_C

**Latent-variable identification:** candidates from equations need review

**Equations / distributions (LaTeX rendering)**

> Automatically typeset for readability; English display transcriptions are not changes to the submitted scientific data.

$$
X\sim \mathcal{N}(0,1)
$$

$$
A\mid X\sim Bern(\mathrm{expit}(-0.4+0.6X))
$$

$$
(E_M,E_C)\sim N_2(0,\begin{bmatrix}
1 & 0.2 \\
0.2 & 1
\end{bmatrix})
$$

$$
M=3+2A+4X+5E_M
$$

$$
C=1+2A+2X+4E_C
$$

$$
Y=1+2A+2M+3C+5X+4AC+2AM+\varepsilon_Y
$$

$$
\varepsilon_Y\sim \mathcal{N}(0,16)
$$

$$
NDE=2+3(2)+4E[C(1)]+2E[M(0)]=26
$$

<details>
<summary>Original source equations (verbatim)</summary>

<pre>
X~N(0,1); A|X~Bern(expit(-0.4+0.6X))
(E_M,E_C)~N_2(0,[[1,0.2],[0.2,1]])
M=3+2A+4X+5E_M
C=1+2A+2X+4E_C
Y=1+2A+2M+3C+5X+4AC+2AM+epsilon_Y; epsilon_Y~N(0,16)
NDE=2+3(2)+4E[C(1)]+2E[M(0)]=26
</pre>

</details>


## Parameter scenarios, estimands and truth

| Scenario | Estimand | Definition | True value | Evidence status |
|---|---|---|---|---|
| section_4_1_continuous | NDE | E[Y(1,M(0))-Y(0,M(0))], with the treatment-induced C under its own treatment in each outcome potential. | 26.0 | reported_numeric |

## Identifiability assumptions

- IID units, consistency, positivity
- Gaussian-copula joint potential construction with shared standardized errors
- Conditional treatment propensity as source
- Cross-world structure for natural direct effect follows the coupled structural equations; such effects require stronger assumptions than interventional effects

## Provenance and review

**Submitted by:** github_file_upload

**Mode:** structured_zip

**Original versus adapted:** adapted_from_source

**Status:** needs_scientific_review

**Warnings:**

- Latent-variable candidates inferred from source equations; confirm their roles and completeness against the original paper or generator.

