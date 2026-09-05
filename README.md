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
`WENO(bounds=(0, 1))`: it limits only the upwind reconstruction of the evaluating cell, so
the two cells sharing a face apply different fluxes whenever the limiter is active. P3's fast
sedimentation across the sharp rain gradient above the surface kept it active. Condensate
masses therefore use plain (conservative) WENO with P3's `SpeciesBorrowing` repairing
undershoots; vapor keeps the bounded scheme (`bounded_condensate_advection = false`,
recorded).

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
- Oceananigans is held at 0.111. Version 0.112 renames `update_advection_timestep!` and
  rebuilds bounded WENO; migrating needs explicit compatibility work and scalar-bound
  regression tests rather than a casual bump.

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
  The published Covert et al. (2022) run (864² × 192 at 35 m, 30.24 km, 06–15 UTC) is a
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
