# Arena model card

**Estimator** · MediPFN V2 with a frozen TabPFN v3 backbone, an Arena-specific mediation adapter, a supervised fine-tuned error-quantile head and a calibration fitted independently of evaluation.

**Target and data** · Superpopulation natural indirect effect under treatment 1, with joint intervention on the mediator vector. One complete, finite, numeric observed table has 500 rows and distinct treatment and control groups. Supply pretreatment baseline covariates as `x` and post-treatment mediators as `m`. Interpret the estimated contrast causally under consistency, positivity, sufficient adjustment for treatment and mediator confounding, and the cross-world condition used to identify the natural effect. No post-treatment mediator–outcome confounder is permitted by this target's identification contract.

**Fit and evidence** · The meeting run's 20,000 distinct fitting tables comprise 8,000 adapter, 10,000 error-head and 2,000 new SFT tables. SFT also replayed prior fitting examples. The 1,024 calibration tables were independent ID draws outside the fitting budget. Eric and Xia–Chan were excluded from mediation-specific fitting, development and calibration. The 2,000-table independent audit verified empirical 95% coverage of 95.39% in ID and 97.90% in source OOD, with mean lengths 0.552 and 2.020, respectively. These are two named benchmark distributions.

**Assets** · The package preserves the original selected checkpoint bytes. Its adapter metadata retains historical configuration paths and hashed development mechanism identities for provenance. The portable runner binds the official backbone to the files provided by the user. The method archive contains no held-out table or oracle. Keep the maintainer archive separate.

**License** · See `NOTICE` and `TABPFN-3-LICENSE`. Obtain the external official weights separately from Prior Labs.
