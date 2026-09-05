#####
##### Lightweight diagnostics: liquid water path, cloud fraction, cloud boundaries,
##### surface rain flux, and the progress message.
#####

using KernelAbstractions: @kernel, @index
using Oceananigans: Field, Average, Integral, Center, Face, KernelFunctionOperation, compute!
using Oceananigans.Grids: znodes
using Oceananigans.Fields: interior
using Statistics: mean
using Printf: @sprintf

"""
    cloud_liquid(model)

The cloud-liquid mass fraction field `qᶜˡ` (one-moment and P3 schemes both provide it).
"""
cloud_liquid(model) = model.microphysical_fields.qᶜˡ

"""
    rain_mass_fraction(model)

The rain mass fraction field `qʳ`.
"""
rain_mass_fraction(model) = model.microphysical_fields.qʳ

reference_density(model) = model.dynamics.reference_state.density

"""
    liquid_water_path(model; species=:cloud)

2D field of the vertically integrated liquid water, `∫ ρ q dz` [kg m⁻²], for
`species = :cloud` (`qᶜˡ`), `:rain` (`qʳ`) or `:total` (both).
"""
function liquid_water_path(model; species=:cloud)
    ρ = reference_density(model)
    q = species === :cloud ? cloud_liquid(model) :
        species === :rain ? rain_mass_fraction(model) :
        cloud_liquid(model) + rain_mass_fraction(model)
    return Field(Integral(ρ * q, dims=3))
end

@inline column_indicator(i, j, k, grid, lwp, threshold) = ifelse(@inbounds(lwp[i, j, 1]) > threshold, 1, 0)
@inline cell_indicator(i, j, k, grid, q, threshold) = ifelse(@inbounds(q[i, j, k]) > threshold, 1, 0)

"""
    cloud_fraction(model; lwp_threshold=5e-3)

0-D field: fraction of columns whose cloud liquid water path exceeds `lwp_threshold`
[kg m⁻²] (5 g m⁻² by default, the usual satellite/LES comparison threshold).
"""
function cloud_fraction(model; lwp_threshold=5e-3)
    grid = model.grid
    lwp = liquid_water_path(model; species=:cloud)
    indicator = KernelFunctionOperation{Center, Center, Nothing}(column_indicator, grid, lwp, lwp_threshold)
    return Field(Average(indicator, dims=(1, 2)))
end

"""
    cloud_fraction_profile(model; threshold=1e-5)

Horizontal fraction of cloudy cells (`qᶜˡ > threshold`) at every level.
"""
function cloud_fraction_profile(model; threshold=1e-5)
    grid = model.grid
    indicator = KernelFunctionOperation{Center, Center, Center}(cell_indicator, grid, cloud_liquid(model), threshold)
    return Field(Average(indicator, dims=(1, 2)))
end

"""
    surface_rain_flux(model)

2D field of the downward rain mass flux through the surface [kg m⁻² s⁻¹] (positive
downward). P3: `-ρqʳ wʳ` with the mass-weighted rain fall speed at the bottom face;
one-moment schemes: the bottom level of Breeze's `precipitation_rate`.
"""
function surface_rain_flux(model)
    μ = model.microphysical_fields
    grid = model.grid
    if haskey(μ, :wʳ)
        op = KernelFunctionOperation{Center, Center, Nothing}(p3_surface_rain_flux, grid, μ.ρqʳ, μ.wʳ)
        return Field(op)
    else
        P = Breeze.precipitation_rate(model, :liquid)
        op = KernelFunctionOperation{Center, Center, Nothing}(bottom_level, grid, P)
        return Field(op)
    end
end

@inline p3_surface_rain_flux(i, j, k, grid, ρqʳ, wʳ) = @inbounds -ρqʳ[i, j, 1] * wʳ[i, j, 1]
@inline bottom_level(i, j, k, grid, P) = @inbounds P[i, j, 1]

"""
    cloud_boundaries(qᶜˡ_profile, z; threshold=1e-5)

Cloud base and top heights [m] from a horizontally averaged cloud-liquid profile, or
`(NaN, NaN)` when no level exceeds `threshold`.
"""
function cloud_boundaries(qᶜˡ_profile, z; threshold=1e-5)
    cloudy = findall(>(threshold), qᶜˡ_profile)
    isempty(cloudy) && return (NaN, NaN)
    return (z[first(cloudy)], z[last(cloudy)])
end

# Value of a 0-D reduced field, without scalar indexing on the device.
scalar_value(field) = sum(interior(field))

"""
    ProgressMessenger(model; wall_clock=Ref(time_ns()))

Callback printing iteration, time, Δt, wall time, max |w|, mean LWP, cloud fraction, max
cloud and rain mass fractions, and the domain-mean surface rain rate.
"""
struct ProgressMessenger{W, L, C, R, QC, QR, T}
    wall_clock :: W
    mean_lwp :: L
    cloud_fraction :: C
    mean_rain_flux :: R
    qᶜˡ :: QC
    qʳ :: QR
    temperature :: T
end

function ProgressMessenger(model)
    mean_lwp = Field(Average(liquid_water_path(model; species=:cloud), dims=(1, 2)))
    cf = cloud_fraction(model)
    rain = Field(Average(surface_rain_flux(model), dims=(1, 2)))
    return ProgressMessenger(Ref(time_ns()), mean_lwp, cf, rain, cloud_liquid(model), rain_mass_fraction(model), model.temperature)
end

function (p::ProgressMessenger)(simulation)
    model = simulation.model
    compute!(p.mean_lwp); compute!(p.cloud_fraction); compute!(p.mean_rain_flux)
    lwp = scalar_value(p.mean_lwp)
    cf = scalar_value(p.cloud_fraction)
    rain = scalar_value(p.mean_rain_flux) * 86400 # mm/day
    wmax = maximum(abs, model.velocities.w)
    qᶜmax = maximum(p.qᶜˡ)
    qʳmax = maximum(p.qʳ)
    Tmin, Tmax = extrema(p.temperature)
    elapsed = 1e-9 * (time_ns() - p.wall_clock[])
    msg = @sprintf("iter %6d, t = %s, Δt = %s, wall = %s | max|w| %.2f m/s, LWP %.1f g/m², CF %.2f, max qᶜˡ %.2e, max qʳ %.2e, rain %.3f mm/d, T ∈ [%.1f, %.1f]",
                   Oceananigans.iteration(simulation), Oceananigans.prettytime(simulation),
                   Oceananigans.prettytime(simulation.Δt), Oceananigans.prettytime(elapsed),
                   wmax, 1e3 * lwp, cf, qᶜmax, qʳmax, rain, Tmin, Tmax)
    @info msg
    return nothing
end
