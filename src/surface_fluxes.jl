#####
##### Surface boundary conditions.
#####
##### Two modes mirror the SAM namelist switches:
#####
#####   * `PrescribedSurfaceFluxes` — `SFC_FLX_FXD = .true.`, `SFC_TAU_FXD = .true.` (the
#####     public Covert et al. 2022 configuration): H(t), LE(t) and the kinematic stress
#####     τ(t) from the `sfc` file; the stress is aligned with the local near-surface wind
#####     (`surface.f90`: `taux = -(u/|U|) τ₀ ρ`, `|U| = max(1, |U|)`).
#####   * `BulkSurfaceFluxes` — `SFC_FLX_FXD = .false.` over `OCEAN` (LASSO `flxsst`): bulk
#####     fluxes from the prescribed SST(t) with wind- and stability-dependent coefficients.
#####     SAM's `oceflx.f90` neutral drag law cdn = 0.0027/U + 0.000142 + 0.0000764 U is the
#####     Large & Yeager polynomial that Breeze's `PolynomialCoefficient` uses by default.
#####
##### Static-energy convention. Breeze's prognostic is s = cᵖᵐ(q) T + gz - ℒqˡ. Adding vapor
##### at the surface at fixed s lowers T because cᵖᵐ grows; SAM's t = T + gz/cp (constant
##### cp) does not have this coupling. To keep the surface latent heat flux temperature-
##### neutral (the same physical-temperature invariant used for tls/qls) the prescribed
##### energy flux is H + (cᵖᵛ - cᵖᵈ) T₀ E, where E = LE/ℒ is the vapor mass flux.
#####

using Oceananigans: FieldTimeSeries, Field, Center, Face, set!
using Oceananigans.Fields: interior
using Statistics: mean
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.Units: Time
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux, PolynomialCoefficient

#####
##### Prescribed fluxes (Covert et al. 2022 / SFC_FLX_FXD)
#####

"""
    PrescribedStressUpdater

Callback that realizes SAM's `SFC_TAU_FXD` stress for the LES/`UNIFORM_SFC_FLX` branch of
`surface.f90`: one horizontally uniform stress of kinematic magnitude τ₀(t), aligned with
the *domain-mean* lowest-level wind,

    τˣ = -ρ₀ τ₀ ū₁ / max(1, |Ū₁|),   τʸ = -ρ₀ τ₀ v̄₁ / max(1, |Ū₁|)

written into the 2D flux fields `τˣ`, `τʸ` that the `ρu`/`ρv` bottom boundary conditions
read. Runs every iteration (the stress lags the wind by one step, as in SAM where
`surface` is called at the start of the step).
"""
struct PrescribedStressUpdater{X, Y, U, V, T, FT}
    τˣ :: X
    τʸ :: Y
    u :: U
    v :: V
    times :: T
    kinematic_stress :: Vector{FT}
    surface_density :: FT
    frame_velocity :: Tuple{FT, FT}   # SAM namelist (ug, vg): the surface wind is ū + ug
end

function (updater::PrescribedStressUpdater)(simulation)
    t = simulation.model.clock.time
    τ₀ = interpolate_time_series(updater.times, updater.kinematic_stress, t)
    ū = lowest_level_mean(updater.u) + updater.frame_velocity[1]
    v̄ = lowest_level_mean(updater.v) + updater.frame_velocity[2]
    U = max(1, sqrt(ū^2 + v̄^2))
    ρ₀ = updater.surface_density
    set!(updater.τˣ, - ρ₀ * τ₀ * ū / U)
    set!(updater.τʸ, - ρ₀ * τ₀ * v̄ / U)
    return nothing
end

"""
    prescribed_surface_flux_boundary_conditions(grid, sfc, day0; thermodynamic_constants,
                                                surface_density, moisture_name,
                                                temperature_neutral_evaporation=true)

Bottom boundary conditions for `ρu`, `ρv`, `ρs` and the moisture density from the SAM
`sfc` time series: energy flux H(t) [+ (cᵖᵛ - cᵖᵈ) SST(t) E(t)], vapor flux E = LE/ℒ, and a
horizontally uniform stress of kinematic magnitude τ(t) aligned with the domain-mean
lowest-level wind (updated by [`PrescribedStressUpdater`](@ref)). Returns
`(boundary_conditions, stress_record)`.
"""
function prescribed_surface_flux_boundary_conditions(grid, sfc::SAMSurfaceForcing, day0;
                                                     thermodynamic_constants,
                                                     surface_density,
                                                     moisture_name,
                                                     temperature_neutral_evaporation = true,
                                                     frame_velocity = (0, 0))
    FT = eltype(grid)
    constants = thermodynamic_constants
    ℒ = constants.liquid.reference_latent_heat
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    times = FT[day_to_seconds(d, day0) for d in sfc.day]

    E = sfc.latent_heat_flux ./ ℒ
    energy = temperature_neutral_evaporation ?
             sfc.sensible_heat_flux .+ (cᵖᵛ - cᵖᵈ) .* sfc.sst .* E :
             copy(sfc.sensible_heat_flux)

    make(values) = begin
        fts = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        for n in eachindex(times)
            set!(fts[n], FT(values[n]))
        end
        fts
    end

    energy_flux = make(energy)
    vapor_flux = make(E)
    ρ₀ = FT(surface_density)

    τˣ = Field{Face, Center, Nothing}(grid)
    τʸ = Field{Center, Face, Nothing}(grid)
    ρu_bc = FluxBoundaryCondition(τˣ)
    ρv_bc = FluxBoundaryCondition(τʸ)
    ρs_bc = FluxBoundaryCondition(energy_flux)
    ρq_bc = FluxBoundaryCondition(vapor_flux)

    moisture_density_name = Symbol("ρ", moisture_name)
    bcs = (; ρu = FieldBoundaryConditions(bottom=ρu_bc),
             ρv = FieldBoundaryConditions(bottom=ρv_bc),
             ρs = FieldBoundaryConditions(bottom=ρs_bc))
    bcs = merge(bcs, NamedTuple{(moisture_density_name,)}((FieldBoundaryConditions(bottom=ρq_bc),)))
    stress = (; τˣ, τʸ, times, kinematic_stress = FT.(sfc.kinematic_stress), surface_density = ρ₀,
                frame_velocity = (FT(frame_velocity[1]), FT(frame_velocity[2])))
    return bcs, stress
end

# Horizontal mean of the lowest level as a device-side reduction (`sum` over a GPU array
# view dispatches to a kernel; `Statistics.mean` on the view would fall back to scalar indexing).
function lowest_level_mean(field)
    level = interior(field, :, :, 1)
    return sum(level) / length(level)
end

"""
    prescribed_stress_updater(stress, velocities)

Build the [`PrescribedStressUpdater`](@ref) for the `stress` record returned by
`prescribed_surface_flux_boundary_conditions` and the model `velocities`.
"""
prescribed_stress_updater(stress, velocities) =
    PrescribedStressUpdater(stress.τˣ, stress.τʸ, velocities.u, velocities.v, stress.times,
                            stress.kinematic_stress, stress.surface_density, stress.frame_velocity)

#####
##### Bulk fluxes from SST (LASSO flxsst / SFC_FLX_FXD = .false.)
#####

"""
    bulk_surface_flux_boundary_conditions(grid, surface_temperature; moisture_name,
                                          roughness_length=1.5e-4, gustiness=0.1)

Bottom boundary conditions computed online from the shared `surface_temperature` field
with Breeze's wind- and stability-dependent `PolynomialCoefficient`. **Approximation of
SAM's `oceflx.f90`, not the same law:** only the neutral drag polynomial coincides
(Large & Yeager / Large & Pond `cdn = 0.0027/U + 0.000142 + 0.0000764 U`); SAM uses
separate neutral Stanton/Dalton numbers (`0.0327√cdn` unstable, `0.0180√cdn` stable,
`0.0346√cdn`), two Monin-Obukhov iterations, and a 1 m s⁻¹ minimum wind, whereas Breeze
uses its own scalar polynomials, the Li et al. (2010) non-iterative stability mapping,
and `gustiness`. Exact fidelity needs a SAM-oceflx boundary implementation.
"""
function bulk_surface_flux_boundary_conditions(grid, surface_temperature; moisture_name,
                                               roughness_length = 1.5e-4, gustiness = 0.1)
    coefficient = PolynomialCoefficient(; roughness_length)
    ρu_bc = BulkDrag(; coefficient, gustiness, surface_temperature)
    ρv_bc = BulkDrag(; coefficient, gustiness, surface_temperature)
    ρs_bc = BulkSensibleHeatFlux(; coefficient, gustiness, surface_temperature)
    ρq_bc = BulkVaporFlux(; coefficient, gustiness, surface_temperature)
    moisture_density_name = Symbol("ρ", moisture_name)
    bcs = (; ρu = FieldBoundaryConditions(bottom=ρu_bc),
             ρv = FieldBoundaryConditions(bottom=ρv_bc),
             ρs = FieldBoundaryConditions(bottom=ρs_bc))
    return merge(bcs, NamedTuple{(moisture_density_name,)}((FieldBoundaryConditions(bottom=ρq_bc),)))
end

"""
    SeaSurfaceTemperatureUpdater(surface_temperature, times, values)

Callback that sets the shared 2D `surface_temperature` field from the `sfc` SST series at
the current simulation time (linear interpolation, as SAM's `forcing.f90`). The same field
feeds the bulk flux boundary conditions and the radiation surface temperature.
"""
struct SeaSurfaceTemperatureUpdater{F, T, V}
    surface_temperature :: F
    times :: T
    values :: V
end

function (updater::SeaSurfaceTemperatureUpdater)(simulation)
    t = simulation.model.clock.time
    sst = interpolate_time_series(updater.times, updater.values, t)
    set!(updater.surface_temperature, sst)
    return nothing
end
