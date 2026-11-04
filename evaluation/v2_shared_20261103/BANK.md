# Shared scientific test bank

Use **`frozen_bank_final/`** for every route. The earlier `frozen_bank/` and its ZIP
are retired pre-evaluation protocol drafts; no model read their scientific tables.
The only change at finalization clarified the limits of the OOD claim. All 1,152
table NPZ bytes and fold assignments remain identical to the first generation.

* Final manifest SHA256: `532d590e08e55be17a6603cfa3ed11e243476a0bd52a5fe01dfbaaa17038974c`
* Transfer archive: `frozen_bank_final.zip`, 18,498,398 bytes
* Archive SHA256: `fddfa11d3b5ee365dca48fb06040d2c8e50afcb8a8d90871aaff2713d5e3d158`
* ID-only execution fixture archive: `id_validation_fixtures.zip`, 488,116 bytes
* Fixture archive SHA256: `dcbae8c5caef642b093ce3d372d1cc8d3769cb5e0632c5732f5fe1a416f88f1b`
* Fixture manifest SHA256: `f11592228c44050fd2dcea6a0f0fda787b2f7e3fe3667eda724779293c457383`

The final design has 768 ID tables, 288 source-OOD tables and 96 composition-OOD
tables, with sample sizes 200/600/1,000. Each of 576 sampled configurations has
two independently generated observation tables. The native configuration hash and
an independently defined active-SEM population hash both give 576 distinct
populations: 384 ID, 144 source-OOD and 48 composition-OOD. These are parameter
configurations from the stated source families, not 576 different literature DGPs.

Each table stores one common outer-fold seed and the exact corresponding stratified
two-fold assignment. Analysis weights give equal family, preset and sample-size
weight as specified in `bank_protocol.json`. Jones' three ID presets rotate the
6/5/5 allocation across sample sizes, and receive equal preset weights in analysis.
Eric and Xia-Chan source OOD refer to our mediation-specific fitting/adaptation/
DEV/calibration; upstream TabPFN-v3 pretraining is not controlled by this experiment.

`bank_admission.json` records successful all-table byte/semantic/fold/schema checks,
exact regeneration of all 1,152 observed tables, exact hashes for 15 upstream
generator files, an independent structural-truth calculation (largest numerical
difference 9.19e-8), and actual two-table replication checks. Test seeds have 319
bits and are in a new purpose namespace, disjoint from prior <=164-bit training/
DEV/calibration seeds. None of 13,680 available archived native configuration
fingerprints overlaps this bank. A matching fixed ID population with new rows
would not itself constitute data leakage; the audit distinguishes configurations,
seed reuse and observation-table reuse.

`load_observed_case(bank_root, case_doc)` returns only x/a/m/y and role-type arrays.
`load_case` additionally returns offline truth, metadata and the common fold array
for the evaluator. True targets, source identities and structural parameters must
not enter any inference wrapper. Six local bank-loader/tamper contracts pass.

The four additional ID-only fixture tables are separate from the scientific bank,
including zero baseline covariates and 100 mediators at n=200 and n=1,000. Their
execution establishes interface/memory behavior, not scientific performance.
No trained model was called during bank generation or this model-free audit.
