#!/bin/bash
#SBATCH --job-name=breezylasso
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=output/slurm-%j.out
# Usage: sbatch scripts/submit_gpu.sbatch <run_case.jl arguments...>
set -euo pipefail
# Separate compiled-code cache for the compute nodes (different CPU target than the login node);
# packages and artifacts are still read from the shared depot.
export JULIA_DEPOT_PATH="$HOME/.julia-compute:$HOME/.julia"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p output
export JULIA_CUDA_USE_COMPAT=false
julia --startup-file=no --project=. scripts/run_case.jl --arch gpu "$@"
