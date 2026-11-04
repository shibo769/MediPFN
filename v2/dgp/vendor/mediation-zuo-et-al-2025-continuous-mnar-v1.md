# Continuous mediation with MNAR mediator and outcome

> **Unverified extraction — scientific review required.**

**Primary area:** Mediation — Direct and Indirect Effects

**Suggested cross-tags (unverified):** none

## Reference paper

**Citation:** Zuo, S., Ghosh, D., Ding, P. and Yang, F. (2025). Mediation Analysis with the Mediator and Outcome Missing Not at Random. Journal of the American Statistical Association 120(550), 794–804. DOI: 10.1080/01621459.2024.2359132.

**DOI:** 10.1080/01621459.2024.2359132 · **Source locator:** Published paper §5.1, Setting C and Assumption III; numeric coefficients in author arXiv:2212.05577v3 supplemental Table S1, PDF p.72 (zero-index 71), lines 3984–4006.

## DGP / generation mechanism

**Sampling design:** Unit: individual; Dependence: IID X,T,error terms and missingness uniforms; within unit M-&gt;Y and M-&gt;R_M, Y-&gt;R_Y

**Generator source:** [generator.py](../generators/mediation-zuo-et-al-2025-continuous-mnar-v1.py)

**Keywords (source-described):** Interactions · Nonlinearity

**Observed variables**

X, T, R_M, M_obs0, R_Y, Y_obs0

**Latent variables**

Not explicitly identified (review required)

**Latent-variable identification:** not explicitly specified

**Equations / distributions (LaTeX rendering)**

> Automatically typeset for readability; English display transcriptions are not changes to the submitted scientific data.

$$
X\sim \mathcal{N}(0,1)
$$

$$
T\sim Bern(0.5), independent
$$

$$
M\mid T,X\sim \mathcal{N}(T+X,1)
$$

$$
Y\mid M,T,X\sim \mathcal{N}(M+T+MT+X,1)
$$

$$
R_M\mid M,T,X\sim Bern(\mathrm{expit}(1.4+M+T+X))
$$

$$
R_Y\mid Y,T,X\sim Bern(\mathrm{expit}(1.8+Y+T+X))
$$

$$
M_obs0=R_M\cdot M
$$

$$
Y_obs0=R_Y\cdot Y
$$

include R_M,R_Y to distinguish missing from genuine zero

<details>
<summary>Original source equations (verbatim)</summary>

<pre>
X~N(0,1), T~Bern(0.5), independent
M|T,X~N(T+X,1)
Y|M,T,X~N(M+T+MT+X,1)
R_M|M,T,X~Bern(expit(1.4+M+T+X))
R_Y|Y,T,X~Bern(expit(1.8+Y+T+X))
M_obs0=R_M*M; Y_obs0=R_Y*Y; include R_M,R_Y to distinguish missing from genuine zero
</pre>

</details>


## Parameter scenarios, estimands and truth

| Scenario | Estimand | Definition | True value | Evidence status |
|---|---|---|---|---|
| setting_C_III | NIE_1 | E[Y(1,M(1))-Y(1,M(0))] in the full-data law before MNAR masking. | 2.0 | reported_numeric |

## Identifiability assumptions

- Randomized treatment and independent structural errors
- Consistency and no interference
- Sequential ignorability for full-data potential outcomes conditional on X
- Missingness depends on unobserved M and Y as specified; observed-only identification depends on the paper's extra conditions

## Provenance and review

**Submitted by:** github_file_upload

**Mode:** structured_zip

**Original versus adapted:** adapted_from_source

**Status:** needs_scientific_review

**Warnings:**

- No explicit latent-variable metadata or reliably identifiable latent declarations; absence from this list does not prove that no latent variables exist.

