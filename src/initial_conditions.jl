#####
##### Sounding → reference state and initial condition.
#####
##### The SAM `snd` file gives potential temperature (interpreted, as in SAM, as the
##### liquid-water potential temperature of the initial state, which has no condensate
##### until the first saturation adjustment), total water, and winds. Following
##### `setdata.f90`, levels are converted to heights hydrostatically and interpolated
##### linearly to the grid; the ENA cases have no levels above the sounding top so no
##### standard-atmosphere extension is needed.
#####

using Random: AbstractRNG, MersenneTwister
using Oceananigans.Grids: znodes, Center
using Breeze: ThermodynamicConstants, SaturationAdjustment, WarmPhaseEquilibrium
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, MoistureMassFractions, temperature
using Breeze.Microphysics: adjust_thermodynamic_state

"""
    SoundingProfiles

Piecewise-linear interpolants (in height) of a `snd` record: `θ`, `qᵗ`, `u`, `v`, plus the
surface pressure. Levels below the surface are kept in the record but excluded from the
interpolation only when `exclude_subsurface_levels=true`. The fidelity default is `false`:
SAM's `setdata.f90` interpolates through the below-surface levels, so they bracket the
lowest LES cells.
"""
struct SoundingProfiles{FT}
    z :: Vector{FT}
    θ :: Vector{FT}
    qᵗ :: Vector{FT}
    u :: Vector{FT}
    v :: Vector{FT}
    surface_pressure :: FT
end

function SoundingProfiles(sounding::SAMSounding; exclude_subsurface_levels=false)
    z = record_heights(sounding)
    keep = exclude_subsurface_levels ? findall(≥(0), z) : eachindex(z)
    return SoundingProfiles(z[keep], sounding.θ[keep], sounding.q[keep],
                            sounding.u[keep], sounding.v[keep], sounding.surface_pressure)
end

(p::SoundingProfiles)(name::Symbol, z) = interpolate_profile(p.z, getproperty(p, name), z)

#####
##### Saturation partition of (θˡ, qᵗ, p) into (T, qᵛ, qᶜˡ) with Breeze's own thermodynamics
#####

"""
    saturation_partition(θˡ, qᵗ, p; constants=ThermodynamicConstants(Float64), standard_pressure=1e5)

Given the liquid-water potential temperature `θˡ`, total water `qᵗ` and pressure `p`,
return `(T, qᵛ, qᶜˡ)` in warm-phase saturation equilibrium, computed with Breeze's
`SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())` secant solver acting on a
`LiquidIcePotentialTemperatureState` — the same thermodynamics (Clausius-Clapeyron,
mixture heat capacity, Exner function) that the one-moment control run uses internally,
so the P3 and one-moment initial states are partitioned identically. This mirrors what
SAM's `micro_init`/`satadj_liquid` does before the first step; the two codes' saturation
formulae differ at the sub-percent level.
"""
function saturation_partition(θˡ, qᵗ, p; constants=ThermodynamicConstants(Float64), standard_pressure=1e5)
    FT = Float64
    q₀ = MoistureMassFractions(FT(qᵗ))
    𝒰₀ = LiquidIcePotentialTemperatureState{FT}(FT(θˡ), q₀, FT(standard_pressure), FT(p))
    adjustment = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    𝒰₁ = adjust_thermodynamic_state(𝒰₀, adjustment, constants)
    T = temperature(𝒰₁, constants)
    q₁ = 𝒰₁.moisture_mass_fractions
    return (T, q₁.vapor, q₁.liquid)
end

"""
    InitialPerturbation(; amplitude_T=0.1, amplitude_q=0.025e-3, depth=600, seed=1234)

Uniform random perturbations of the SAM LASSO-ENA `setperturb.f90` case 5: ±0.1 K in
temperature and ±0.025 g/kg in vapor below 600 m.
"""
Base.@kwdef struct InitialPerturbation
    amplitude_T :: Float64 = 0.1
    amplitude_q :: Float64 = 0.025e-3
    depth :: Float64 = 600
    seed :: Int = 1234
end

"""
    initial_state_columns(profiles::SoundingProfiles, z_centers, reference_pressure)

Compute the column profiles (T, qᵛ, qᶜˡ, θˡ, qᵗ, u, v) at the cell centers `z_centers`
given the reference pressure at those heights.
"""
function initial_state_columns(profiles::SoundingProfiles, z_centers, reference_pressure;
                               constants=ThermodynamicConstants(Float64), standard_pressure=1e5)
    n = length(z_centers)
    T = zeros(n); qᵛ = zeros(n); qᶜˡ = zeros(n); θˡ = zeros(n); qᵗ = zeros(n); u = zeros(n); v = zeros(n)
    Tᵈ = zeros(n) # condensate-free temperature (all water as vapor), as SAM's HUJI-SBM micro_init
    for (k, z) in enumerate(z_centers)
        θˡ[k] = profiles(:θ, z)
        qᵗ[k] = profiles(:qᵗ, z)
        u[k] = profiles(:u, z)
        v[k] = profiles(:v, z)
        T[k], qᵛ[k], qᶜˡ[k] = saturation_partition(θˡ[k], qᵗ[k], reference_pressure[k]; constants)
        𝒰 = LiquidIcePotentialTemperatureState{Float64}(θˡ[k], MoistureMassFractions(qᵗ[k]), standard_pressure, reference_pressure[k])
        Tᵈ[k] = temperature(𝒰, constants)
    end
    return (; z=collect(z_centers), T, qᵛ, qᶜˡ, θˡ, qᵗ, u, v, T_condensate_free=Tᵈ)
end

"""
    perturbation_array(Nx, Ny, z_centers, perturbation::InitialPerturbation)

One deterministic array of uniform random numbers in [-1, 1] for every cell below
`perturbation.depth` (zero above), drawn from `MersenneTwister(perturbation.seed)` in a
fixed (i, j, k) order on the host. The *same* array multiplies both the temperature and
the vapor perturbation, as in `setperturb.f90` (one `rrr` per cell), and it is copied to
the device by `set!`, so CPU and GPU runs start from identical states.
"""
function perturbation_array(Nx, Ny, z_centers, perturbation::InitialPerturbation)
    rng = MersenneTwister(perturbation.seed)
    ϵ = zeros(Nx, Ny, length(z_centers))
    for k in eachindex(z_centers), j in 1:Ny, i in 1:Nx
        r = 2 * rand(rng) - 1
        ϵ[i, j, k] = z_centers[k] ≤ perturbation.depth ? r : 0.0
    end
    return ϵ
end
