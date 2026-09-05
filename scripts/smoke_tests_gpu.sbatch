#!/bin/bash
#SBATCH --job-name=breezylasso-smoke
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=output/slurm-smoke-%j.out
# GPU smoke tests before any long run: forcing-only, dry dynamics, cloudy P3 (30-60 min).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p output
julia --startup-file=no --project=. scripts/smoke_tests.jl gpu
