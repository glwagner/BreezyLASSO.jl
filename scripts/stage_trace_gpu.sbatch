#!/bin/bash
#SBATCH --job-name=stagetrace
#SBATCH --partition=gpua100largex4spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm-stagetrace-%j.out
set -euo pipefail
export JULIA_DEPOT_PATH="$HOME/.julia-compute:$HOME/.julia"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p output
echo "checkout: $(pwd) $(git rev-parse --short HEAD 2>/dev/null) | args: $* | PROBE_GRID=${PROBE_GRID:-covert} PROBE_MOMENTS=${PROBE_MOMENTS:-positive} PROBE_NX=${PROBE_NX:-256}"
julia --startup-file=no --project=. scripts/p3_stage_trace_gpu.jl "$@"
