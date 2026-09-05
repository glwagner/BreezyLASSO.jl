# Probe the P3-N75 GPU/Float32 blow-up seen in the smoke tests: run the 32²×260 cloudy stage with
# per-minute diagnostics and report the first non-finite prognostic.
#   julia --project scripts/p3_stability_probe.jl <cpu|gpu> <Float32|Float64> <max_dt> [minutes] [microphysics] [off-switches]
# off-switches: comma-separated subset of vadv,thermo,nudging,sponge,geo,upper,radiation,surface,closure
using BreezyLASSO, Breeze, Oceananigans, Oceananigans.Units, CUDA, Printf, Statistics
using Oceananigans.Grids: zspacings

arch = lowercase(ARGS[1]) == "gpu" ? GPU() : CPU()
FT = ARGS[2] == "Float64" ? Float64 : Float32
max_Δt = parse(Float64, ARGS[3])
minutes = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 45.0
scheme_arg = length(ARGS) ≥ 5 ? ARGS[5] : "p3_n75"
# "p3_aer2" runs with the SBM diagCCN projection; "p3_aer2_noproj" without it
scheme = Symbol(replace(scheme_arg, "_noproj" => ""))
off = length(ARGS) ≥ 6 ? split(ARGS[6], ",") : String[]
data = joinpath(@__DIR__, "..", "data", "covert2022_bin")

switches = Dict{Symbol, Any}()
"vadv" ∈ off && (switches[:vertical_advection] = nothing)
"thermo" ∈ off && (switches[:thermodynamic_tendencies] = false)
"nudging" ∈ off && (switches[:wind_nudging_timescale] = nothing)
"sponge" ∈ off && (switches[:sponge] = nothing)
"geo" ∈ off && (switches[:geostrophic] = false)
"upper" ∈ off && (switches[:upper_boundary_relaxation] = false)
"radiation" ∈ off && (switches[:radiation] = nothing)
"surface" ∈ off && (switches[:surface] = nothing)
"closure" ∈ off && (switches[:closure] = nothing)
"enthalpy" ∈ off && (switches[:sedimentation_enthalpy] = false)
scheme === :p3_aer2 && !endswith(scheme_arg, "_noproj") && (switches[:aerosol_replenishment] = :diagnostic_ccn)
println("off switches: ", off, "  scheme: ", scheme_arg)

case = lasso_ena_simulation(data; preset=:covert_public_bin, arch, FT, Nx=32, Ny=32, Lx=1120, Ly=1120,
                            z_faces=lasso_ena_vertical_faces(), microphysics=scheme, stop_time=minutes*60,
                            Δt=min(1.0, max_Δt), max_Δt, write_output=false, progress_interval=1minute, switches...)
model = case.model
μ = model.microphysical_fields
function first_bad(model)
    for (name, f) in pairs(Oceananigans.prognostic_fields(model))
        all(isfinite, Array(interior(f))) || return name
    end
    return nothing
end
# Water budget: domain-integrated water mass versus the surface vapor flux in and rain flux out
grid_ = model.grid
Δz_ = diff(Array(znodes(grid_, Face()))); Δx_ = grid_.Lx / grid_.Nx; Δy_ = grid_.Ly / grid_.Ny
ρ_ = Array(interior(model.dynamics.reference_state.density, 1, 1, :))
ρΔV_ = reshape(ρ_ .* Δz_ .* (Δx_ * Δy_), 1, 1, :)
water_mass(m) = sum(Array(interior(m.microphysical_fields.qᵛ)) .* ρΔV_) + sum(Array(interior(m.microphysical_fields.qᶜˡ)) .* ρΔV_) + sum(Array(interior(m.microphysical_fields.qʳ)) .* ρΔV_)
rain_flux_field = surface_rain_flux(model)
last_time = Ref(0.0); accumulated_rain = Ref(0.0); accumulated_vapor = Ref(0.0); water₀ = Ref(water_mass(model))
ℒ_ = ThermodynamicConstants(Float64).liquid.reference_latent_heat
sfc_times = [day_to_seconds(d, case.config.day0) for d in case.sfc.day]
sfc_E = case.sfc.latent_heat_flux ./ ℒ_
function diagnostics(sim)
    m = sim.model
    Δt_diag = m.clock.time - last_time[]; last_time[] = m.clock.time
    compute!(rain_flux_field)
    accumulated_rain[] += sum(Array(interior(rain_flux_field))) * Δx_ * Δy_ * Δt_diag
    E = case.config.surface == "prescribed_fluxes" ? interpolate_profile(sfc_times, sfc_E, m.clock.time) : 0.0
    accumulated_vapor[] += E * grid_.Lx * grid_.Ly * Δt_diag
    budget = water_mass(m) - water₀[] - accumulated_vapor[] + accumulated_rain[]
    qʳ_bottom = maximum(Array(interior(μ.qʳ, :, :, 1)))
    aerosol = haskey(μ, :ρnᵃ) ? @sprintf(" | ρnᶜˡ [%.2e, %.2e] ρnᵃ [%.2e, %.2e]", minimum(μ.ρnᶜˡ), maximum(μ.ρnᶜˡ), minimum(μ.ρnᵃ), maximum(μ.ρnᵃ)) : ""
    @printf("t=%6.1f min Δt=%.2f | qʳ max %.2e (k=1 max %.2e, min ρqʳ %.2e) | ρnʳ [%.2e, %.2e] | qᶜˡ max %.2e | max|w| %.2f | T [%.1f, %.1f] | forced water budget (excludes qls/subsidence/upper relaxation) %.3e kg (of %.3e)%s\n",
            m.clock.time / 60, sim.Δt, maximum(μ.qʳ), qʳ_bottom, minimum(μ.ρqʳ), minimum(μ.ρnʳ), maximum(μ.ρnʳ), maximum(μ.qᶜˡ),
            maximum(abs, m.velocities.w), minimum(m.temperature), maximum(m.temperature), budget, water₀[], aerosol)
    bad = first_bad(m)
    if !isnothing(bad)
        println("FIRST NONFINITE PROGNOSTIC: ", bad, " at t = ", m.clock.time, " iteration ", m.clock.iteration)
        error("non-finite state")
    end
end
# Every iteration: catch the first non-finite prognostic before the NaN checker aborts, with
# the extrema of the moment fields at that moment.
function iteration_guard(sim)
    m = sim.model
    bad = first_bad(m)
    isnothing(bad) && return nothing
    println("FIRST NONFINITE PROGNOSTIC: ", bad, " at iteration ", m.clock.iteration, " t = ", m.clock.time, " Δt = ", sim.Δt)
    for name in (:ρqᶜˡ, :ρnᶜˡ, :ρnᵃ, :ρqʳ, :ρnʳ, :ρqᵛ)
        haskey(μ, name) || continue
        f = Array(interior(μ[name]))
        finite = filter(isfinite, f)
        println("   ", name, ": nonfinite ", count(!isfinite, f), ", finite extrema ", isempty(finite) ? "none" : extrema(finite))
    end
    error("non-finite state")
end
add_callback!(case.simulation, iteration_guard, IterationInterval(1))
add_callback!(case.simulation, diagnostics, TimeInterval(30))
run!(case.simulation)
println("PROBE FINITE: ", string(FT), " max_Δt=", max_Δt, " scheme=", scheme)
