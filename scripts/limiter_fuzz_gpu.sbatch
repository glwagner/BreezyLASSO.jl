#!/bin/bash
#SBATCH --job-name=limiterfuzz
#SBATCH --partition=gpua100largex4spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=output/slurm-limiterfuzz-%j.out
set -euo pipefail
export JULIA_DEPOT_PATH="$HOME/.julia-compute:$HOME/.julia"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p output
julia --startup-file=no --project=. scripts/limiter_fuzz_gpu.jl
