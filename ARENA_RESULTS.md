# Arena meeting-run evaluation

The `arena_n500` checkpoint was evaluated on 2,000 independently generated
500-row tables. The ID split and source-OOD split each contain 500 mechanisms
and two independent tables per mechanism. There were no recorded prediction
failures. Results are descriptive for the specified simulation mixture.

## 95% intervals

| Split | Tables | Bias | Sampling SE | RMSE | Mean length | Coverage | Mean time |
|---|---:|---:|---:|---:|---:|---:|---:|
| ID | 1,000 | 0.0002 | 0.1992 | 0.2101 | 0.5519 | 95.39% | 0.636 s |
| Source OOD | 1,000 | 0.0311 | 0.7231 | 0.7325 | 2.0203 | 97.90% | 0.646 s |

Within each split, families are equally weighted, then presets within a family,
then mechanisms and tables. Sampling SE is the square root of the weighted
mean within-mechanism sampling variance, using the two independent tables per
mechanism. Coverage SE clustered by mechanism is 0.66 and 0.45 percentage
points for ID and source OOD, respectively. Mean inference time includes the
table estimator, error head and interval calculation on A40/L40S GPUs after
model loading and warm-up; it excludes data generation. Different settings can
have different outcome and effect scales, so interval widths across settings
should be interpreted in their respective units.

## Results by configuration

| Split | Configuration | Tables | Bias | Sampling SE | RMSE | Mean length | Coverage | Mean time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ID | Cheng | 124 | -0.0110 | 0.1280 | 0.1410 | 0.4411 | 93.55% | 0.633 s |
| ID | Diaz-Hejazi | 126 | 0.0033 | 0.0849 | 0.0927 | 0.2789 | 96.03% | 0.644 s |
| ID | Diaz intermediate | 124 | -0.0126 | 0.0501 | 0.0520 | 0.1491 | 94.35% | 0.654 s |
| ID | He-Song-Xu | 126 | 0.0007 | 0.0347 | 0.0432 | 0.0964 | 97.62% | 0.640 s |
| ID | Jones LNL | 42 | -0.0528 | 0.2752 | 0.2757 | 0.9098 | 97.62% | 0.633 s |
| ID | Jones LNN | 42 | -0.1124 | 0.4106 | 0.3962 | 1.1186 | 90.48% | 0.663 s |
| ID | Jones correlated | 42 | 0.0828 | 0.2877 | 0.3404 | 1.2160 | 92.86% | 0.466 s |
| ID | Miles | 124 | 0.0304 | 0.1553 | 0.1464 | 0.4749 | 94.35% | 0.636 s |
| ID | Shan | 126 | -0.0123 | 0.3081 | 0.3324 | 1.1020 | 96.83% | 0.639 s |
| ID | Zuo | 124 | 0.0304 | 0.2490 | 0.2684 | 0.7910 | 96.77% | 0.651 s |
| Source OOD | Eric | 500 | 0.0137 | 0.0815 | 0.0851 | 0.4200 | 99.00% | 0.642 s |
| Source OOD | Xia-Chan binary | 250 | 0.0509 | 0.3174 | 0.3299 | 1.2849 | 96.40% | 0.650 s |
| Source OOD | Xia-Chan continuous | 250 | 0.0462 | 1.4063 | 1.4222 | 5.9562 | 97.20% | 0.648 s |

The 10 ID configurations represent eight source families. Eric and Xia-Chan
are held out from mediation-specific fitting, development and calibration.
TabPFN's upstream pretraining exposure is not controlled. Jones NNN is a
separate composition-shift setting and is not in this evaluation. The
generators adapt source mechanisms to a complete-data natural indirect effect;
some treatment-induced confounding and missingness from the source papers were
removed. The held-out bank and evaluator are maintained separately.

The model was trained on 8,000 adapter, 10,000 error-head and 2,000 fresh SFT
tables. SFT replay does not add distinct tables. A further 1,024 independent ID
tables were used for calibration. The checkpoint was selected on development
data, without using the evaluation outcomes. The selected adapter is from step
1,750 of the 2,000-step training run.

The archived method passed an independent SCC CPU restoration on a synthetic
500-row table. A subsequent Windows CPU run with the same table, checkpoints
and folds produced finite intervals and a point estimate differing by about
0.0010 from the SCC result; a repeated Windows run was identical. The two
environments use different operating systems and PyTorch builds. The cause of
the cross-environment numerical difference has not been established.

Same-table linear and random-forest comparators for this meeting run have not
yet been evaluated.
