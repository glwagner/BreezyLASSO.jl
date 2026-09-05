#!/bin/bash
# The staged Covert-public-bin development benchmark runs (1M-control → P3-N75 → P3-aer2),
# full 256×256×192 grid at 35 m, 6 h (06-12 UTC 18 July 2017), one A100 each.
set -euo pipefail
cd "$(dirname "$0")/.."
# MODE=fixed (default): the preset's fixed dt = 0.5 s (SAM fidelity, ~2-3x more steps).
# MODE=adaptive: labelled override, CFL wizard up to --max_dt (recorded in provenance).
# FLOAT (default Float64): the P3 runs showed Float32-specific surface rain undershoot/repair
# artefacts in the 45-minute probes, so production uses Float64 unless overridden.
# PARTITION (default gpua100largex4spot): one GPU per run.
MODE=${MODE:-fixed}
FLOAT=${FLOAT:-Float64}
PARTITION=${PARTITION:-gpua100largex4spot}
SB="sbatch --partition=$PARTITION --gres=gpu:1 --cpus-per-task=6 --mem=150G"
common="--data data/covert2022_bin --preset covert_public_bin --Nx 256 --Ny 256 --float $FLOAT"
if [ "$MODE" = "adaptive" ]; then
    common="$common --max_dt ${MAX_DT:-1.5}"
    suffix="_adaptive"
else
    suffix=""
fi
$SB --job-name=lasso-1m   scripts/submit_gpu.sbatch $common --microphysics one_moment --output output/covert_public_bin_one_moment$suffix
$SB --job-name=lasso-n75  scripts/submit_gpu.sbatch $common --microphysics p3_n75     --output output/covert_public_bin_p3_n75$suffix
$SB --job-name=lasso-aer2 scripts/submit_gpu.sbatch $common --microphysics p3_aer2 --aerosol_replenishment diagnostic_ccn --output output/covert_public_bin_p3_aer2$suffix
