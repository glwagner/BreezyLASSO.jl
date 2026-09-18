#!/bin/bash
#SBATCH --job-name=ena-les-weno
#SBATCH --partition=gpua100largex4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=output/weno_only_logs/%x-%A_%a.out
set -euo pipefail
# Submit from the repository root; Slurm copies this script to its spool directory.
cd "${SLURM_SUBMIT_DIR:?Submit this script from the BreezyLASSO repository root}"
test -f scripts/weno_only/run_member.jl
# This depot is for compute nodes only. Do not populate its caches on the login node.
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${HOME}/.julia-compute:${HOME}/.julia}"
export JULIA_CUDA_USE_COMPAT=false
julia --startup-file=no --project=. scripts/weno_only/run_member.jl \
  "${1:-${SLURM_ARRAY_TASK_ID:?Pass smoke or submit a member array}}"
