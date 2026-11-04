#!/bin/bash -l
set -euo pipefail
[[ $# == 5 && "$2" =~ ^[0-9a-f]{64}$ && "$3" =~ ^[0-9a-f]{64}$ && "$4" =~ ^[0-9a-f]{64}$ && "$5" =~ ^[0-9a-f]{64}$ ]] || exit 40
R=/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9/capacity_4m_v10_20261028/isolated_cpu_supplement_20261029
[[ "$(sha256sum -- "$0" | awk '{print $1}')" == "$2" ]] || exit 41
[[ "$(sha256sum -- "$R/sources/cpu_stage_launch_v5.py" | awk '{print $1}')" == "$3" ]] || exit 42
[[ "$(sha256sum -- "$R/sources/cpu_simple_workflow_v5.py" | awk '{print $1}')" == "$4" ]] || exit 43
printf 'CPU_STAGE_WRAPPER_SPOOL_SHA %s %s\n' "$JOB_ID" "$2"
module load python3/3.10.12
B=/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9
exec "$B/env/bin/python" -B "$R/sources/cpu_stage_launch_v5.py" "$1" "$4" "$3" "$5"
