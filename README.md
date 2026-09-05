# BreezyLASSO.jl

Reproducing the LASSO-ENA large-eddy-simulation protocol with [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)
for the closed-cell stratocumulus case of **18 July 2017** at the ARM Eastern North Atlantic site.

> **Status (5 September 2026).** The official LASSO-ENA `samin` archive (restricted; free ARM
> account) is not yet in the workspace. Every run reported here is driven by the *public*
> Covert, Mechem & Zhang (2022) SAM input files and is labelled a **Covert-public-bin
> development benchmark**, not an official LASSO-ENA reproduction. The official preset
> (`:lasso_ena_official`) is implemented and refuses to run without the archive.

## What is implemented

The LASSO SAM protocol was taken from the official custom source
(`https://code.arm.gov/lasso/lasso-ena-codes/lasso_sam_sbm.git`, branch `lasso_ena_noice`,
commit `12d02446a2147388dc89d828e6e0553106abea0f`, 2025-10-24), file by file:

| SAM component (file) | BreezyLASSO implementation |
|---|---|
| `snd`/`lsf`/`sfc`/`prm`/`grd` readers (`setforcing.f90`, `setdata.f90`) | `read_sam_*` in `src/sam_input_files.jl`: pressure- or height-coordinate records, 7- or 9-column `lsf` (optional `ug`/`vg`), Fortran namelist parser, SAM hydrostatic heights |
| Interpolation to the model levels (`forcing.f90`) | `sam_interpolate_column`: linear in pressure for pressure-coordinate records against the reference pressure, linear in height otherwise; above the record top `tls = qls = wls = 0`, winds hold |
| Geostrophic wind `ug0/vg0` in `coriolis.f90` | `TimeVaryingGeostrophicForcing` (a `FieldTimeSeries`, distinct from the nudging target) |
| Domain-mean wind nudging to `ul0/vl0` (`nudging.f90`, n0, `tauls`) | `MeanProfileNudging` (horizontal mean, not pointwise `Relaxation`) |
| Full-field upwind `-wls ∂zϕ` on u, v, t, vapor and every microphysical field (`subsidence.f90`) | `LargeScaleVerticalAdvection` (same object under every specific key; bottom/top cells skipped as in SAM) |
| `tls` added to `t = T + gz/cp - Lv qˡ/cp`, `qls` to the vapor mixing ratio (`forcing.f90`, constant `cp = 1004`) | `large_scale_thermodynamic_forcings`: `dT/dt = tls` and `drᵛ/dt = qls` in Breeze's `s = cᵖᵐ T + gz - ℒqˡ`, with `dqᵛ/dt = qᵈ²/(1 - qᶜ) qls` (condensate mass fractions held fixed — an approximation to SAM's fixed dry-basis ratios) and `ds/dt = cᵖᵐ tls + (cᵖᵛ - cᵖᵈ) T dqᵛ/dt` (physical-temperature invariant; step- and kernel-tested) |
| Upper sponge on `u - ū`, `v - v̄`, `w` (`damping.f90`) | `SAMSponge`: top 30 %, `τ` geometric from 1800 s to 60 s |
| Top-two-level relaxation of `t`, vapor toward the sounding (`upperbound.f90`) | `upper_boundary_relaxation_forcings`, `τ = 3600 s` |
| Prescribed H/LE/τ, stress along the domain-mean lowest-level wind (`surface.f90`, `SFC_FLX_FXD`, `SFC_TAU_FXD`) | `prescribed_surface_flux_boundary_conditions` + `PrescribedStressUpdater` (surface wind `ū + (ug, vg)`); energy flux `H + (cᵖᵛ - cᵖᵈ) SST E` keeps evaporation temperature-neutral |
| Translating frame: namelist `ug`, `vg` subtracted from initial winds, nudging targets and geostrophic profiles (`setdata.f90`, `forcing.f90`) | `translation_velocity` (preset default from the namelist for prescribed fluxes; refused with bulk fluxes, which need the absolute wind) |
| Bulk fluxes from SST (`oceflx.f90`, LASSO `flxsst`) | `bulk_surface_flux_boundary_conditions`: an **approximation** — only the neutral drag polynomial coincides; SAM's separate Stanton/Dalton numbers, two MOST iterations and 1 m s⁻¹ wind floor are not reproduced (Breeze uses Li et al. 2010 stability and gustiness). Shared SST field also feeds radiation |
| Fixed `dt` from the namelist | presets set `Δt = max_Δt = dt` (fixed step); adaptive stepping is an explicit, labelled override |
| `rad_simple.f90` (F₀ = 113, F₁ = 22 W m⁻², κ = 85, no free-troposphere term) | `SimpleLongwaveRadiation` radiation model (legacy Covert-era control; `:dycoms` variant restores Stevens et al. 2005) |
| RRTMG LW+SW (official protocol) | Breeze RRTMGP all-sky (`AllSkyOptics`) at the ENA position and date, prescribed effective radii (10 μm liquid, 30 μm ice) |
| HUJI-SBM two-mode aerosol (`aer1/2/3`) | P3 `AerosolActivation` with the LASSO radii/widths; cm⁻³ → kg⁻¹ using the surface density and a constant mixing ratio with height (`FCCN0 = FCCNR_mp ρ(z)/ρ(0)`); `DiagnosticCCNProjection` reproduces the `diagCCN` reservoir rule |
| `setperturb.f90` case 5 (±0.1 K, ±0.025 g kg⁻¹ below 600 m, one draw per cell) | `perturbation_array`: one deterministic host array reused for T and vapor |

Microphysics is staged as **1M-control → P3-N75 → P3-aer2** (`microphysics = :one_moment`,
`:p3_n75`, `:p3_aer2`), all using the complete P3 implementation on Breeze `origin/main`.

### Sedimentation must carry its energy (P3-N75 surface runaway)

On the pinned Breeze `main`, sedimentation moves condensate mass but not its static-energy
content (Breeze PR 959 describes the same defect); rain piling into the surface cell warmed it
by ℒ Δqʳ/cᵖ and the P3-N75 GPU smoke test ran away to T > 320 K within minutes of drizzle
onset (isolated with `scripts/p3_runaway_probe.jl`: hot cell at k = 1, Δs ≈ 0 while qʳ jumped).
`SedimentationEnthalpyForcing` adds, for every sedimenting prognostic condensate, the
divergence of `h · [Φ(w + w_fall, q) − Φ(w, q)]` with the tracer's own advection scheme
(mirroring the bounds-preserving limiter when that scheme is used), each flux carrying the
content `h = ∂s/∂q|_T = (cˣ − cᵖᵈ) T − ℒˣ` of its donor cell; a rain shaft crossing an isothermal saturated
column now leaves T unchanged to 5 mK (test), versus ±1.7 K without it. It is on by default
(`sedimentation_enthalpy = true`) and recorded in provenance; PR 959 will supersede it.

### Bounds-preserving WENO is not conservative where its limiter fires

A second P3 runaway (rain piling into the surface cell, reaching 10 g kg⁻¹ with vapor
converted to rain by the negative-moisture repair) traced to Oceananigans'
`WENO(bounds=(0, 1))` as released in 0.111: it limits only the upwind reconstruction of the
evaluating cell, so the two cells sharing a face apply different fluxes whenever the limiter
is active, and P3's fast sedimentation across the sharp rain gradient above the surface kept
it active. Evidence retained: in otherwise identical 45-minute P3-N75 GPU probes the surface
cell reached qʳ = 9.5 g kg⁻¹ with the bounded scheme versus < 0.03 g kg⁻¹ with plain WENO,
and the forcing-free rain-shaft budget `scripts/mass_budget_probe.jl` (8×8×60 column, 2 min)
gives a residual of −0.134 % of the initial rain for bounded WENO versus −0.048 % for plain
WENO (the plain residual is the explicit-Euler outflow estimate's own error).

The production configuration nevertheless follows Breeze's `examples/rico.jl`: every
microphysical water-mass tracer (vapor and all condensate masses, P3 included) is advected
with bounds-preserving WENO, the number/volume moments with its positivity-only form, and the static energy with plain WENO
(`scalar_advection_schemes`; `bounded_condensate_advection = false` is a diagnostic
sensitivity only, recorded in provenance). The limiter path itself is the thing to fix:
Oceananigans `main` (post-0.111, commit `67a2204`) stores the limiter factor per cell and
rescales every face reconstruction with its own cell's factor, which makes the flux
single-valued at shared faces. It is exercised in an isolated environment (branch `ocmain`:
`[sources]` overrides to Oceananigans `main` and the Breeze branch `glw/lasso-ena-ocmain`,
which follows Oceananigans' renamed `update_advection!` contract and refreshes each scalar's
limiter on the specific field Breeze advects). Result of the same rain-shaft budget there:
bounded WENO −0.046 % versus plain −0.048 %, i.e. the bounded residual collapses onto the
outflow-estimate error, and the full test suite passes. The 45-minute fixed-Δt (0.5 s) GPU
probes of P3-N75 and P3-aer2 (Float64, 32²×260, Slurm jobs 853 and 854) then ran finite in
that environment with the bounded scheme on every water mass, surface `qʳ` staying at
0.002 g kg⁻¹ (N75) and 0.00003 g kg⁻¹ (aer2) where the 0.111 limiter had piled 9.5 g kg⁻¹.
On that evidence the production pin was moved to these revisions (see *Dependency notes*).

### Float32 WENO-Z weights overflow on number concentrations (Oceananigans fix)

Every Float32 P3-aer2 run went non-finite at iteration 1 (GPU jobs 836/837/847/848 and the
8×8 CPU reproduction in `scripts/p3_float32_first_step_probe.jl`), while Float64 ran for
tens of minutes. `scripts/p3_float32_stage_probe.jl` located it: with every forcing off, the
stage-2 tendencies of `ρnᵃ` and `ρnᶜˡ` are NaN in the cloud layer, the P3 bundle at those
cells is finite in both precisions, and the **advection** of `nᵃ` is NaN wherever activation
at stage 1 has cut the aerosol number from 4.6×10⁸ to 2.8×10⁸ kg⁻¹ within the stencil.
Oceananigans' WENO-Z weights are `C★ (1 + (τ / (β + ϵ))²)` with `ϵ = 1e-8`: a smooth
sub-stencil has `β ≈ 0` while the global indicator is `τ ~ Δψ² ≈ 3×10¹⁶`, so the squared
ratio overflows Float32 (any `Δψ ≳ 4×10⁵` does), one `α` becomes `Inf` and the normalized
weights become NaN. The fix caps the ratio at `sqrt(floatmax(FT)) / 8` (finite ratios are
unchanged) and lives on the Oceananigans branch `glw/weno-z-float32-overflow` (commit
`877618e`, with a regression test on the failing stencil), which this package pins.

### Rain-number runaway in the 10-m surface layer of the Covert grid (P3)

On the 192-level Covert grid (Δz = 10 m below 1.5 km) both P3 production members went
non-finite after 13–21 minutes, while the one-moment control completed its 6 hours and
the same P3 configuration survives on the 260-level LASSO grid (Δz = 25 m). The 32²
probes reproduce it on the Covert grid at Δt = 0.5 **and** 0.25 s: the rain number in the
lowest cell grows by a factor 100 every 30 s of simulated time (a dt-independent rate,
so a rate term, not a time-stepping instability) while its mass stays ~5×10⁻⁷ kg kg⁻¹.
`scripts/p3_rain_number_probe.jl` shows the chain: cells just above the surface carry rain
mass but no rain number, so P3's slope clamp diagnoses 2-mm drops there and lets the mass
fall at 9.3 m s⁻¹ and the number at 5.4 m s⁻¹ — a sedimentation Courant number of 0.47 at
10 m, above the 5/18 up to which the bounds-preserving limiter guarantees its bounds. In the
surface cell the number spike makes P3 diagnose micron drops whose number fall speed
(0.006 m s⁻¹) no longer removes it, the plain (unlimited) WENO number advection plus the
negative-number clamp then grow the spike by ~8 % per step, and P3's DSD-consistency
relaxation (−n/10 s) cannot hold it. The fix is a positivity-only limiter for the number and
volume moments (`moment_advection = :positive`, i.e. `WENO(bounds = (0, ∞))`, now the
default and recorded in provenance): with it the same 32² Covert-grid probe ran 45 minutes
with the rain number never above 1.7×10⁵ m⁻³ (job 868), where the unlimited scheme reached
10¹⁸ and diverged. Plain WENO for the moments is kept as the labelled sensitivity
`moment_advection = :plain`. The P3 production members are run on both grids with the
limiter; on the 260-level LASSO grid the sedimentation Courant number also stays inside the
limiter's guarantee (P3-N75 at full 256² scale ran its first hour there cleanly before the
fix, job 865).

### The limiter's 0/0 corner at the lower bound (second Oceananigans fix)

With the positivity limiter, the full 256² Covert-grid P3-N75 run still went non-finite at
iteration 105 (52 s), in two cells at cloud top, while the same configuration at 32² ran
45 minutes. The per-stage trace (`scripts/p3_stage_trace_gpu.jl`, restoring the exact
pre-stage state and recomputing every term) showed the rain-number advection tendency NaN
with finite P3 rates, and the limiter factor θ NaN in one cell whose rain number is exactly
zero below a decaying tail of ε₂-multiples (2×10⁻¹⁹, 4.7×10⁻²⁹, 0, …): its minimum
reconstruction is exactly −ε₂ = 0.3 × (−1/6) × 2×10⁻¹⁹, so `θᵐⁱⁿ = |(0 − 0)/(m − cᵢ + ε₂)|`
is 0/0 — reproduced bit for bit on the CPU. Because m − cᵢ ≤ 0 by construction, the
regularization belongs on the other side of that denominator (`m − cᵢ − ε₂`, always
negative); the max side already adds ε₂ to a non-negative difference. The fix and a
regression test on the offending stencil (for bounds (0, 1) and (0, ∞)) are the second
commit of the Oceananigans branch `glw/weno-z-float32-overflow` (`c78eeaa`), which this
package pins. The corner is not specific to the infinite bound: the mass tracers'
`(0, 1)` limiter has the same 0/0 for a zero cell with an undershoot of exactly ε₂.

### Breeze fix required for P3-aer2

The prognostic-aerosol P3 configuration produced `NaN` in `ρnᶜˡ` after 8–9 steps in every
configuration (with or without forcing, either initialization, any Δt): advection and
sedimentation leave positive but subnormal cloud mass in cloud-free cells while the DSD
diagnosis floors the droplet number above zero, so `Nᶜˡ / (ρ qᶜˡ)` overflows to `Inf` and
`Inf × 0 = NaN` enters the number tendency. The generic fix (threshold the quotient at
`minimum_mass_mixing_ratio`, plus a Float32/Float64 regression test) lives on the Breeze
branch `glw/p3-subnormal-cloud-mass` (commit `0f4ffac`), which this package pins.

### Dependency notes

- The model advects every scalar with explicit WENO (bounds-preserving for mass fractions);
  the adaptive-implicit vertical advection (`AdaptiveImplicitVerticalAdvection`) whose
  sedimentation coupling is corrected in Breeze PR 964 is **not** exercised here, so that fix
  is not required for these runs. Switching to AIVA would require rebasing onto PR 964 first.
- Oceananigans is pinned by `[sources]` to the branch `glw/weno-z-float32-overflow` (commit
  `c78eeaa`, still versioned 0.111.0): `main` at `67a2204` (per-cell bounds-preserving
  limiter, renamed `update_advection!` contract) plus the Float32 WENO-Z weight cap and the limiter 0/0 fix described
  above. Breeze is pinned to the branch `glw/lasso-ena-ocmain`
  (commit `5fc404c`), which stacks that contract (a per-scalar limiter refresh on the specific
  fields Breeze advects, the acoustic stepper's split time step as an
  `adaptive_advection_timestep` method) on top of the P3 subnormal-cloud fix of
  `glw/p3-subnormal-cloud-mass`. Both revisions are recorded in every provenance file.

## Layout

```
src/            package (readers, grids, forcings, radiation, surface, initial state, case driver, diagnostics)
scripts/        fetch_inputs.jl, run_case.jl, smoke_tests.jl, Slurm submit scripts
test/           unit + regression tests (synthetic fixtures; Covert files used when present)
analysis/       plot_results.jl (separate environment)
data/           inputs (not versioned except README and MANIFEST)
results/        figures and lightweight summaries committed from runs
```

## Running

```julia
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project scripts/fetch_inputs.jl                # public Covert files + checksums
julia --project -e 'using Pkg; Pkg.test()'
julia --project scripts/smoke_tests.jl cpu             # staged smoke tests
sbatch scripts/smoke_tests_gpu.sbatch                  # same on one GPU
sbatch scripts/submit_gpu.sbatch --preset covert_public_bin --microphysics p3_n75 --Nx 256 --Ny 256
```

Every run writes `provenance.toml` (TOML; input checksums including `grd` when present, Breeze
pinned revision and checkout state, Oceananigans/Julia versions, Manifest checksum, LASSO SAM
reference commit, and the full configuration record, including the preset label, time-stepping
mode, translation frame, aerosol chemistry, and any overrides).

## Presets

- `:covert_public_bin` — exactly the runnable configuration of the public bin-paper
  repository namelist: `256x256x192` at 35 m (8.96 km), `day0 = 199.25`, `nstop × dt = 6 h`,
  prescribed fluxes, `rad_simple` longwave only, no wind nudging, `doupperbound`, `dodamping`.
- All water-mass tracers (vapor and every condensate mass, P3 included) use bounds-preserving WENO with bounds (0, 1); number and volume moments use the same limiter with bounds (0, ∞) (positivity only); static energy uses plain WENO (see the limiter notes above).
  different, not directly runnable target. The 192-level vertical grid is a labelled
  reconstruction (the repository does not ship its `grd`).
- `:lasso_ena_official` — needs `snd`, `lsf`, `sfc`, `prm`, `grd` from the `samin` bundle;
  bulk SST fluxes, RRTMGP, `tauls` nudging, `nrad` radiation cadence, P3-aer2 with the
  `diagCCN` projection, 24 h. Missing files or namelist switches raise an error.

## Known model-form differences (documented, not hidden)

- SGS: Smagorinsky–Lilly instead of SAM's 1.5-order TKE; advection: WENO instead of MPDATA.
- P3 versus HUJI-SBM/Morrison: no bin spectra; the SBM initializes condensate-free with
  `qᵗ = q0` (the default `p3_initialization = :condensate_free` mirrors that; `:equilibrium` is
  a labelled alternative that starts from the 1M control's saturation partition).
- P3 condensate masses use plain WENO (conservative; P3 repairs undershoots); one-moment rain keeps the bounded scheme.
- RRTMGP columns end at the LES top (no radiative-only layers up to TOA as in SAM's RRTMG), so
  the `:rrtmgp` option is a Breeze *analog* of the official radiation until padded-column
  fluxes are validated; effective radii are prescribed, not diagnosed from P3.
- P3 aerosol chemistry is set explicitly to the HUJI-SBM values (density 1790 kg m⁻³,
  molecular weight 0.115 kg mol⁻¹, van 't Hoff factor 3). The SBM `diagCCN` reservoir rule
  is applied once per time step (`DiagnosticCCNProjection`), not at every microphysics call.
- Breeze uses `s = cᵖᵐ(q) T + gz − ℒqˡ` with variable heat capacity where SAM uses `cp = 1004`;
  the forcing and surface-flux adapters keep the physical temperature response identical.
- Below-surface sounding levels are interpolated through (as SAM does; `exclude_subsurface_levels`
  is an explicit sensitivity).

## Results

See `results/README.md` (updated with each committed run).

## References

- LASSO-ENA documentation: <https://lasso-ena.svcs.arm.gov/latest/>
- Covert, Mechem & Zhang (2022), ACP 22, 1159, doi:10.5194/acp-22-1159-2022; inputs at
  <https://github.com/dmechem/ENA_variability_LES_bulk_paper> and `..._bin_paper`
- Stevens et al. (2005), MWR 133, 1443 (DYCOMS-II RF01 simple radiation)
