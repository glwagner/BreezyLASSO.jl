#!/bin/bash
# The staged Covert-public-bin development benchmark runs (1M-control → P3-N75 → P3-aer2),
# full 256×256×192 grid at 35 m, 6 h (06-12 UTC 18 July 2017), one A100 each.
set -euo pipefail
cd "$(dirname "$0")/.."
# MODE=fixed (default): the preset's fixed dt = 0.5 s (SAM fidelity, ~2-3x more steps).
# MODE=adaptive: labelled override, CFL wizard up to --max_dt (recorded in provenance).
MODE=${MODE:-fixed}
common="--data data/covert2022_bin --preset covert_public_bin --Nx 256 --Ny 256 --float Float32"
if [ "$MODE" = "adaptive" ]; then
    common="$common --max_dt ${MAX_DT:-1.5}"
    suffix="_adaptive"
else
    suffix=""
fi
sbatch --job-name=lasso-1m   scripts/submit_gpu.sbatch $common --microphysics one_moment --output output/covert_public_bin_one_moment$suffix
sbatch --job-name=lasso-n75  scripts/submit_gpu.sbatch $common --microphysics p3_n75     --output output/covert_public_bin_p3_n75$suffix
sbatch --job-name=lasso-aer2 scripts/submit_gpu.sbatch $common --microphysics p3_aer2 --aerosol_replenishment diagnostic_ccn --output output/covert_public_bin_p3_aer2$suffix
