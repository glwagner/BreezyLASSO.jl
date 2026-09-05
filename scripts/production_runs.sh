#!/bin/bash
# The staged Covert-public-bin development benchmark runs (1M-control → P3-N75 → P3-aer2),
# full 256×256×192 grid at 35 m, 6 h (06-12 UTC 18 July 2017), one A100 each.
set -euo pipefail
cd "$(dirname "$0")/.."
# MODE=fixed (default): the preset's fixed dt = 0.5 s (SAM fidelity, ~2-3x more steps).
# MODE=adaptive: labelled override, CFL wizard up to --max_dt (recorded in provenance).
# FLOAT (default Float32, the preset precision). A single Float32/Float64 pair of chaotic LES
# probes differed late in the run; that is not evidence of a precision artefact, so both
# precisions are run as a controlled pair (FLOAT=Float64 for the second member).
# PARTITION (default gpua100largex4; PARTITION_1M / PARTITION_N75 / PARTITION_AER2 override
# one run), one GPU per run; TIME (default 72:00:00): the 32²×260 Float32 smokes step at
# ~0.05 s, so the 43 200 fixed steps of the 12.6-million-cell grid need a day or more.
# RUNS (default "one_moment p3_n75 p3_aer2") selects the members; CPUS (4) and MEM (100G) per run.
MODE=${MODE:-fixed}
FLOAT=${FLOAT:-Float32}
PARTITION=${PARTITION:-gpua100largex4}
TIME=${TIME:-72:00:00}
RUNS=${RUNS:-"one_moment p3_n75 p3_aer2"}
common="--data data/covert2022_bin --preset covert_public_bin --Nx 256 --Ny 256 --float $FLOAT"
if [ "$MODE" = "adaptive" ]; then
    common="$common --max_dt ${MAX_DT:-1.5}"
    suffix="_adaptive"
else
    suffix=""
fi
[ "$FLOAT" = "Float64" ] && suffix="${suffix}_f64"
submit() {  # submit <partition> <job name> <microphysics> [extra run_case options]
    local partition=$1 name=$2 microphysics=$3; shift 3
    sbatch --partition="$partition" --time="$TIME" --gres=gpu:1 --cpus-per-task="${CPUS:-4}" --mem="${MEM:-100G}" --job-name="$name" \
        scripts/submit_gpu.sbatch $common --microphysics "$microphysics" "$@" --output "output/covert_public_bin_$microphysics$suffix"
}
for run in $RUNS; do
    case $run in
        one_moment) submit "${PARTITION_1M:-$PARTITION}"   lasso-1m   one_moment ;;
        p3_n75)     submit "${PARTITION_N75:-$PARTITION}"  lasso-n75  p3_n75 ;;
        p3_aer2)    submit "${PARTITION_AER2:-$PARTITION}" lasso-aer2 p3_aer2 --aerosol_replenishment diagnostic_ccn ;;
        *) echo "unknown run $run" >&2; exit 1 ;;
    esac
done
