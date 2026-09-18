# WENO-only LES comparison — 11 September 2026

Six matched cases: 1M, P3-N75, and P3-aer2 on each of the existing Covert and LASSO vertical grids. All remain the **Covert-public-input development benchmark**, 8.96 km horizontally at 35 m resolution, 256×256, six hours from 06 UTC 18 July 2017.

Requested change: Smagorinsky–Lilly → `closure = nothing`. WENO(5) supplies numerical dissipation; no explicit SGS turbulence closure is added. Vapor and all condensate mass bounds remain [0,1], number/volume moments remain positivity-limited, and the thermal variable remains unbounded. The existing prescribed surface fluxes, radiation, forcing, sponge and upper relaxation remain in place. “WENO-only” means no explicit turbulence closure, not no other physics or damping.

The launcher reuses each control's recorded command verbatim except for `--closure none` and a new `_weno_only` output suffix. Existing results are never overwritten. The full production configurations retain Float32, theta thermodynamics, seed1234, Δt=0.5 s, 60 s slices and the aer2 diagnostic-CCN projection. Dependencies stay pinned at Breeze a7fa3c8 and Oceananigans c78eeaa, matching the controls.

A separate GPU smoke tests all six reduced 32×32 cases for 60 simulated seconds, asserting absent closure, retained bounds, fixed timestep and finite prognostics. Production is submitted with an `afterok` dependency on that smoke and an array concurrency limit of two. A smoke success is not evidence of six-hour stability.

Array indices 0/1/2: Covert grid, 1M/N75/aer2. Indices 3/4/5: LASSO vertical grid, same member order.

Submitted 11 September 2026: GPU smoke **1040**, production array **1041**, dependency `afterok:1040`, concurrency `0-5%2`, partition `gpua100largex4` (80 GB A100). At submission check the smoke was running and the production array was waiting on it. Driver closure-option unit checks and launcher syntax validation passed locally. These are submission statuses, not completed simulation results.

The production wrapper compares the final written configuration against the corresponding control, excluding only closure, output directory and descriptive label. Outputs go to the existing BreezyLASSO `output/` directory with `_weno_only` appended. Future logs go to `output/weno_only_logs/`. Original September 11 logs remain in `/home/greg_aeolus_earth/Breeze-static-energy/les_weno_only/logs/` and are not committed.

After completion, compare matching 06–12 and 09–12 UTC windows: LWP/rain accumulation, cloud base/top with common thresholds, cloud fraction, resolved velocity variance and spatial spectra/structure. Retain the Smagorinsky controls and do not describe a visually sharper field as more accurate without observation/scale-aware diagnostics. If the no-closure run violates transport stability at the fixed control timestep, document that failure before any separate timestep sensitivity.

## Recorded completion

Slurm accounting checked 18 September 2026 reports smoke 1040 and all six array members **COMPLETED**. The smoke log records 98/98 passing assertions. This establishes run completion, not a scientific assessment of the resulting clouds.

| Job | Grid | Microphysics | Wall time |
|---|---|---|---|
| 1041_0 | Covert | 1M | 01:08:02 |
| 1041_1 | Covert | P3-N75 | 03:48:41 |
| 1041_2 | Covert | P3-aer2 | 05:49:34 |
| 1041_3 | LASSO | 1M | 01:27:54 |
| 1041_4 | LASSO | P3-N75 | 05:00:14 |
| 1041_5 | LASSO | P3-aer2 | 07:40:28 |

These jobs used the original launcher outside this repository. The committed launcher preserves the experiment but resolves the repository from its own location; the batch script now uses the submission directory and writes logs inside the repository's ignored output tree.

## Reproduction

Use the repository's pinned Julia environment and fetch the Covert inputs using the existing input-download instructions. The six tracked control provenance files under `results/covert_grid_theta/` and `results/lasso_grid_theta/` are required by the launcher. No input archives, credentials, logs or simulation output are included here.

From the repository root, first run the lightweight checks:

```sh
julia --startup-file=no test/weno_only_cli.jl
bash -n scripts/weno_only/submit.sbatch
```

On this cluster, submit a smoke followed by the six-member array:

```sh
mkdir -p output/weno_only_logs
smoke_job=$(sbatch --parsable --job-name=ena-les-weno-smoke --time=02:00:00 scripts/weno_only/submit.sbatch smoke)
sbatch --array=0-5%2 --dependency=afterok:"$smoke_job" scripts/weno_only/submit.sbatch
```

The wrapper refuses to overwrite existing `_weno_only` output directories. Preserve or relocate a previous output explicitly before a rerun. The batch script targets the non-spot 80 GB A100 partition; change resource settings for another cluster. The compute-node depot default may be overridden with `JULIA_DEPOT_PATH`, but do not precompile into `~/.julia-compute` on a login node.

For a direct run on an allocated GPU:

```sh
julia --project=. scripts/weno_only/run_member.jl smoke
julia --project=. scripts/weno_only/run_member.jl 0
```

The general driver also accepts `--closure none` or `--closure smagorinsky_lilly`. Omitting the option preserves the existing preset default.
