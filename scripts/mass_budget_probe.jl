# Forcing-free sedimentation mass budget (evidence for the bounds-preserving WENO conservation
# issue): a rain shaft in a quiescent P3 column, no radiation/surface/large-scale forcing. Total
# vapor + cloud + rain mass may change only by the rain leaving through the bottom face; the
# residual (with an explicit-Euler estimate of the outflow) is printed for bounded and plain WENO.
#   julia --project scripts/mass_budget_probe.jl
using BreezyLASSO, Breeze, Oceananigans, Oceananigans.Units, Statistics, Printf
using Breeze.Thermodynamics: saturation_specific_humidity, PlanarLiquidSurface
using Breeze.Microphysics.PredictedParticleProperties: CloudDroplets
grid = RectilinearGrid(CPU(), Float64; size=(8, 8, 60), x=(0, 280), y=(0, 280), z=(0, 600), halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded))
constants = ThermodynamicConstants(Float64)
rs = ReferenceState(grid, constants; surface_pressure=101300, potential_temperature=290)
dynamics = AnelasticDynamics(rs)
zc = Array(znodes(grid, Center())); ρᵣ = Array(interior(rs.density, 1, 1, :)); Δz = 10.0
p3 = P3Microphysics(Float64; cloud=CloudDroplets(Float64; number_concentration=75e6))
T₀ = 285.0
qsat = [saturation_specific_humidity(T₀, ρᵣ[k], constants, PlanarLiquidSurface()) for k in 1:60]
col(v) = repeat(reshape(v, 1, 1, 60), 8, 8, 1)
qʳ₀(x, y, z) = (300 ≤ z ≤ 400) ? (2e-3 * (1 + 0.5 * sin(2π * x / 280) * sin(2π * y / 280))) : 0.0
nʳ₀(x, y, z) = qʳ₀(x, y, z) / (4/3 * π * 1000 * (0.5e-3)^3)
for bounded in (true, false)
    sa = BreezyLASSO.scalar_advection_schemes(5, p3, :qᵛ; bounded_condensates=bounded)
    forcing = (; ρs = sedimentation_enthalpy_forcings(p3, sa; thermodynamic_constants=constants))
    model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics=p3, thermodynamic_constants=constants, momentum_advection=WENO(order=5), scalar_advection=sa, forcing)
    set!(model; T=T₀, qᵛ=col(qsat), qʳ=qʳ₀, nʳ=nʳ₀)
    μ = model.microphysical_fields
    ρΔV = reshape(ρᵣ .* Δz .* (35.0^2), 1, 1, :)
    water(m) = sum(interior(μ.qᵛ) .* ρΔV) + sum(interior(μ.qᶜˡ) .* ρΔV) + sum(interior(μ.qʳ) .* ρΔV)
    rain0 = sum(interior(μ.qʳ) .* ρΔV)
    rain_flux = surface_rain_flux(model)
    W₀ = water(model); out = 0.0; Δt = 1.0
    for n in 1:120
        compute!(rain_flux)
        out += sum(interior(rain_flux)) * 35.0^2 * Δt
        time_step!(model, Δt)
    end
    residual = water(model) - W₀ + out
    @printf("bounded=%s: initial rain %.4f kg, rain out %.4f kg, residual (created) %+.5f kg = %+.3f%% of initial rain | min ρqʳ %.2e | max qʳ %.2e | T range (%.3f, %.3f)\n",
            bounded, rain0, out, residual, 100 * residual / rain0, minimum(μ.ρqʳ), maximum(μ.qʳ), extrema(model.temperature)...)
end
