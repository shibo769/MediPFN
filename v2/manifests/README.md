# Source identities

Each manifest records a distinct source closure. Checkpoint manifests retain
the release that produced those weights, even as later stages are added.

| Manifest | Meaning |
|---|---|
| [factual_65fff8193cab7488.json](factual_65fff8193cab7488.json) | Original 59-file factual/nested release; produced the archived nuisance weights |
| [error_475a2abf480e9e36.json](error_475a2abf480e9e36.json) | 68-file release that produced the error/post-trained weights; original 59 files plus nine later-stage files |
| [error_c1bd3404749513a9.json](error_c1bd3404749513a9.json) | Current 68-file primary pipeline; only the workflow calibration-output filename differs from the preceding release |
| [optional_67409e5e616eea45_full.json](optional_67409e5e616eea45_full.json) | Original optional release file identities, including operational files omitted from this repository |
| [optional_67409e5e616eea45_repository.json](optional_67409e5e616eea45_repository.json) | Exact distributed subset, omitted paths/reasons and the separate bootstrap utility hash |

The top-level [`../release_manifest.json`](../release_manifest.json) matches
the current primary error-pipeline manifest. It does not silently include the
optional experiments, checkpoint binaries or editorial documentation.

Frozen source guides and protocol status fields describe their freeze points.
Execution progress and verified artifacts are recorded in
[`../../STATUS.md`](../../STATUS.md), the model cards and checkpoint manifests.
