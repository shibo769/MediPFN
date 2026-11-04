# Binary mediator and induced confounder with Plackett odds-ratio copula

> **Unverified extraction — scientific review required.**

**Primary area:** Mediation — Direct and Indirect Effects

**Suggested cross-tags (unverified):** none

## Reference paper

**Citation:** Xia, F. and Chan, K. C. G. (2023). Identification, Semiparametric Efficiency, and Quadruply Robust Estimation in Mediation Analysis with Treatment-Induced Confounding. Journal of the American Statistical Association 118(542), 1272–1281. DOI: 10.1080/01621459.2021.1990765.

**DOI:** 10.1080/01621459.2021.1990765 · **Source locator:** Author accepted manuscript §4.1, “Binary C and M”, PDF p.17; https://par.nsf.gov/servlets/purl/10300663.

## DGP / generation mechanism

**Sampling design:** Unit: individual; Dependence: IID rows; within-row binary M and C dependent via Plackett copula

**Generator source:** [generator.py](../generators/mediation-xia-chan-2023-binary-plackett-v1.py)

**Keywords (source-described):** Interactions · Nonlinearity

**Observed variables**

X, A, C, M, Y

**Latent variables**

Not explicitly identified (review required)

**Latent-variable identification:** not explicitly specified

**Equations / distributions (LaTeX rendering)**

> Automatically typeset for readability; English display transcriptions are not changes to the submitted scientific data.

$$
X\sim \mathcal{N}(0,1)
$$

$$
A\mid X\sim Bern(\mathrm{expit}(-0.2+0.3X))
$$

$$
M\mid A,X\sim Bern(\mathrm{expit}(-0.3-0.2A+0.5X))
$$

$$
C\mid A,X\sim Bern(\mathrm{expit}(-0.2-0.1A+0.3X))
$$

$$
OR(M,C\mid A,X)=\exp(1-2A+3X)
$$

$$
joint p_{11} solves p_{11}(1-pM-pC+p_{11})/[(pM-p_{11})(pC-p_{11})]=OR
$$

$$
Y=1+3A+6M+3C+6X+4AC+2AM+\varepsilon_Y
$$

$$
\varepsilon_Y\sim \mathcal{N}(0,16)
$$

$$
NDE=E_X[3+3\{pC(1,X)-pC(0,X)\}+4pC(1,X)+2pM(0,X)]
$$

<details>
<summary>Original source equations (verbatim)</summary>

<pre>
X~N(0,1); A|X~Bern(expit(-0.2+0.3X))
M|A,X~Bern(expit(-0.3-0.2A+0.5X))
C|A,X~Bern(expit(-0.2-0.1A+0.3X))
OR(M,C|A,X)=exp(1-2A+3X); joint p11 solves p11(1-pM-pC+p11)/[(pM-p11)(pC-p11)]=OR
Y=1+3A+6M+3C+6X+4AC+2AM+epsilon_Y; epsilon_Y~N(0,16)
NDE=E_X[3+3{pC(1,X)-pC(0,X)}+4pC(1,X)+2pM(0,X)]
</pre>

</details>


## Parameter scenarios, estimands and truth

| Scenario | Estimand | Definition | True value | Evidence status |
|---|---|---|---|---|
| section_4_1_binary | NDE | E[Y(1,M(0))-Y(0,M(0))] under the specified binary copula structural model, with C following the treatment in each outcome potential. | 5.4954757582156875 | reported_numeric |

## Identifiability assumptions

- IID units and positivity
- Conditional Bernoulli margins and Plackett odds ratio fix joint distribution of M and C given A,X
- Natural effect interpreted under a structural coupling of potential variables; numeric NDE here depends only on marginal mediator and confounder probabilities because outcome is additive in C,M apart from A interactions

## Provenance and review

**Submitted by:** github_file_upload

**Mode:** structured_zip

**Original versus adapted:** adapted_from_source

**Status:** needs_scientific_review

**Warnings:**

- No explicit latent-variable metadata or reliably identifiable latent declarations; absence from this list does not prove that no latent variables exist.

