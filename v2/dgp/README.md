# MediPFN v2 rich mediation generator bank

Frozen upstream: `Statsinference/stats-inference-arena` commit
`8182caf7185845a31a2122e032811bb2254d6342`. The 15 requested registry records are
vendored with per-file Git blob hashes. Zhou's ordered and Luo's longitudinal
generators are excluded. The two Shan observation schemes share one complete-data
SEM, so the API exposes 14 unique presets rather than double-counting them.

## API

```python
from dgp import available_sources, sample_task, sample_random_task, truth
t = sample_random_task(n=200, seed=1234, split='train', compute_truth=False)
t = sample_task('jones_lnl', n=200, seed=5001, mechanism_seed=1101)
```

Each task has `x[n,p]`, `a[n]`, `m[n,q]`, `y[n]`, role/type labels and provenance
metadata. `p=0` is valid; `q` ranges from 1 to 100. All observed arrays are float32.
The mechanism seed controls structural parameters; the table seed controls new
observations under those same parameters. Source names and true parameters must
never be supplied to the prediction network. `truth` is optional during factual
pretraining and is reserved for the synthetic error-target/validation stages.

The target is always the **superpopulation NIE at treatment 1**,
`E[Y(1,M(1))-Y(1,M(0))]`. This does not reuse the original registry's principal-
stratum, path-specific, interventional or missing-data truth labels. It also does
not switch to an empirical-covariate finite-population target for a test table.

## Source split frozen before fitting

| Partition | Presets | Provenance families |
|---|---|---|
| Train / independent ID dev / independent ID calibration / ID test | Cheng; Diaz-Hejazi; Diaz intermediate; He-Song-Xu; Jones LNL/LNN/correlated; Miles; Shan complete; Zuo complete | 8 |
| Source OOD | Eric; Xia-Chan discrete and continuous | 2 held-out families |
| Composition OOD | Jones NNN | Jones family is seen, this composition is held out |

The four Jones presets are treated as one provenance group because the registry
associates them with the same DOI. Source-OOD means the entire Eric and Xia-Chan
families are absent from pretraining, error training, dev and calibration.
OOD observations are for the frozen-model evaluation stage only. DEV/test/table
seed namespaces must additionally be kept separate by the run configuration.

Random training tasks first sample uniformly among the 8 families and then among
that family's retained presets. Jones and Shan are therefore not overweighted by
the number of registry records.

## Structural adaptations and truth

| Preset | Adaptation | Superpopulation truth computation |
|---|---|---|
| Cheng | Set event D coefficients in M and Y to zero; omit D; A is source Z | 1D Gaussian quadrature of logistic-normal mediator contrast times Y-M coefficient |
| Diaz-Hejazi | Preserve three joint mediators and baseline law | Exact enumeration of 8 baseline states; additive Y gives probability contrast |
| Diaz intermediate | Remove Z and all M-Z, A-Z-M, Y-Z, A-Z-Y coefficients | Exact enumeration of 8 baseline states |
| He-Song-Xu | Preserve complete scalar Gaussian mediation | alpha_s * beta_m |
| Jones LNL/LNN/NNN | Preserve source forms; optionally vary candidate count | alpha dot beta |
| Jones correlated | Preserve two-block Gaussian covariance, randomized A | shared active-path count times two path coefficients |
| Miles | Remove three induced C1 covariates and their interactions in M/Y; retain A-M-Y interaction | mediator_exposure * (outcome_mediator + outcome_exposure_mediator) |
| Shan | Export complete X1/X2/X3 before observation masks; omit shadows/masks; merge duplicates | (outcome_mediator + outcome_treatment_mediator) * (mediator_treatment + mediator_treatment_x1_squared / 3) |
| Eric | Preserve source structural forms | 4 discrete baseline states and 80-node Gaussian quadrature |
| Xia-Chan discrete | Remove C effects in Y; sample original M marginal; retain A-M interaction | 80-node Gaussian quadrature |
| Xia-Chan continuous | Remove C effects in Y; preserve M marginal and A-M interaction | m_a * (y_m + y_am) |
| Zuo | Export complete M/Y before masks; omit missingness indicators | alpha_t * (beta_m + beta_mt) |

The confounders are removed structurally, rather than merely hidden from the
observed table. Adaptation is enforced again for explicit config overrides.
The correlated Jones source's `mediator_error_sd` is unused in the source code;
the adaptation retains that source's unit-diagonal covariance and records the
choice. The factorized sampler is algebraically equivalent to its block covariance
for `0 <= rho_bridge <= rho_within < 1` and avoids repeated Cholesky factorizations.

## Initial training parameter domains

These domains form a new pretraining prior around the registered presets; they are
not claims about the source paper's own train/test design. The JSON configuration
of every sampled mechanism is recorded in metadata.

* Selected path, confounding, noise and scale coefficients are multiplied by
  independent Uniform(0.5, 1.5) factors. Exact selected keys are explicit in
  `sample_mechanism`; constrained probabilities and arbitrary metadata are not
  blindly mutated.
* Diaz-Hejazi mediator path scale is Uniform(0.2,1); baseline probabilities are
  Uniform(0.25,0.75); the three Y-M coefficients vary independently by the above
  factors.
* Jones LNL/LNN/NNN mediator count is one of 3,5,10,20; up to 3 active treatment
  and outcome paths have independent signs and magnitudes in [0.3,1.8] and
  [0.2,1.0]. Propensity scale is [0.4,1.2], noise SDs [0.6,1.5].
* Correlated Jones mediator count is 10,20,50,100; within-block correlation is
  [0.15,0.6], bridge correlation [0,within]; path coefficients have independent
  signs and magnitudes [0.3,1.5], Y noise SD [0.6,1.5].
* After these draws, probability .15 zeros every A-M path and the next .15 zeros
  every M-Y path. Remaining mechanisms can have a signed outcome path. He-Song-Xu
  additionally samples among the source's four registered alternatives/nulls.
  Null mechanisms retain independent variation in nuisance/noise parameters.

No coefficient range, source allocation or parameter choice is selected using
held-out model results. Parameter-OOD/stress banks will require separately frozen
domains; the current implementation does not silently create them.

## Verification and acceleration

Run `python -m dgp.test_generators`. The suite covers exact vendored hashes,
90 native smoke tables, seeded reproducibility, independent observation draws for
the same mechanism, full-data duplicate merging, source separation, individual/
joint pathway nulls, canceling vector paths, published compatible analytic truths,
independent covariate Monte Carlo integration checks, observed-distribution
agreement with the frozen native generator, and correlated Gaussian covariances.

NumPy vectorization replaces source row loops; direct independent Uniform draws
replace Phi(Normal) for complete-data Shan covariates; a two-block Gaussian factor
sampler replaces per-table dense factorization. No SciPy dependency is required.
Vectorized equivalence is distributional, so identical seeds do not imply identical
individual rows relative to a native Python `random` implementation.
