#!/bin/bash
#SBATCH --job-name=rainprobe
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm-rainprobe-%j.out
set -euo pipefail
export JULIA_DEPOT_PATH="$HOME/.julia-compute:$HOME/.julia"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p output
echo "checkout: $(pwd) $(git rev-parse --short HEAD 2>/dev/null) | args: $* | PROBE_GRID=${PROBE_GRID:-covert}"
julia --startup-file=no --project=. scripts/p3_rain_number_probe.jl gpu "$@"
