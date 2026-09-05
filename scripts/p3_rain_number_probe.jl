# Track the P3 rain-number runaway seen on the 192-level Covert grid: every 30 s locate the
# cell with the largest ρnʳ; once it exceeds a threshold, print the local column and the full
# P3 process-rate breakdown at that cell (Float64 CPU evaluation of the same scheme).
#   PROBE_GRID=covert julia --project scripts/p3_rain_number_probe.jl <cpu|gpu> <Float32|Float64> <dt> [minutes] [microphysics]
using BreezyLASSO, Breeze, Oceananigans, Oceananigans.Units, CUDA, Printf, Statistics
using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.Thermodynamics: StaticEnergyState, MoistureMassFractions, ThermodynamicConstants,
                             saturation_specific_humidity, PlanarLiquidSurface
using Breeze.Microphysics.PredictedParticleProperties: p3_state_tendencies, compute_p3_process_rates

arch = lowercase(ARGS[1]) == "gpu" ? GPU() : CPU()
FT = ARGS[2] == "Float64" ? Float64 : Float32
Δt = parse(Float64, ARGS[3])
minutes = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 45.0
scheme = length(ARGS) ≥ 5 ? Symbol(ARGS[5]) : :p3_n75
threshold = 1e7      # ρnʳ [m⁻³] beyond which the column is reported
reports = Ref(0)
data = joinpath(@__DIR__, "..", "data", "covert2022_bin")
z_faces = get(ENV, "PROBE_GRID", "covert") == "covert" ? covert_public_bin_vertical_faces() : lasso_ena_vertical_faces()
extra = scheme === :p3_aer2 ? (; aerosol_replenishment=:diagnostic_ccn) : NamedTuple()
get(ENV, "PROBE_MOMENTS", "plain") == "positive" && (extra = merge(extra, (; moment_advection=:positive)))
case = lasso_ena_simulation(data; preset=:covert_public_bin, arch, FT, Nx=32, Ny=32, Lx=1120, Ly=1120,
                            z_faces, microphysics=scheme, stop_time=minutes*60, Δt, max_Δt=Δt,
                            write_output=false, progress_interval=5minutes, extra...)
model = case.model
μ = model.microphysical_fields
println("microphysical fields: ", keys(μ))
prognostic_names = AM.prognostic_field_names(model.microphysics)
specific_names = map(AM.specific_field_name, prognostic_names)
zc = Array(znodes(model.grid, Center()))
zf = Array(znodes(model.grid, Face()))
ρᵣ = Array(interior(model.dynamics.reference_state.density, 1, 1, :))
pᵣ = Array(interior(model.dynamics.reference_state.pressure, 1, 1, :))
p3cpu, _ = BreezyLASSO.build_microphysics(Float64, scheme; droplet_number=75e6, surface_density=Float64(ρᵣ[1]))
constants = ThermodynamicConstants(Float64)
velocity_names = filter(n -> startswith(string(n), "w") && n !== :w, collect(keys(μ)))

function report(sim)
    m = sim.model
    ρnʳ = Array(interior(μ.ρnʳ))
    peak, idx = findmax(ρnʳ)
    i, j, k = Tuple(idx)
    @printf("t=%6.2f min | max ρnʳ %.3e at (%d,%d,%d) z=%.0f | max qʳ %.3e | max|w| %.2f\n",
            m.clock.time / 60, peak, i, j, k, zc[k], maximum(μ.qʳ), maximum(abs, m.velocities.w))
    (peak > threshold && reports[] < 4) || return nothing
    reports[] += 1
    snap = Dict(n => Array(interior(μ[n]))[i, j, :] for n in specific_names)
    T = Array(interior(m.temperature))[i, j, :]
    s = Array(interior(m.formulation.specific_energy))[i, j, :]
    qᵛ = Array(interior(μ.qᵛ))[i, j, :]
    w = Array(interior(m.velocities.w))[i, j, :]
    vel = Dict(n => Array(interior(μ[n]))[i, j, :] for n in velocity_names)
    println("   column (k, z, T, S-1, qᶜˡ, qʳ, nʳ, w(face k), ", join(string.(velocity_names), ", "), ")")
    for kk in max(k - 3, 1):min(k + 3, size(m.grid, 3))
        qs = saturation_specific_humidity(Float64(T[kk]), Float64(ρᵣ[kk]), constants, PlanarLiquidSurface())
        vels = join([@sprintf("%.3f", vel[n][kk]) for n in velocity_names], " ")
        @printf("      %3d %7.1f %8.3f %+.4e %.3e %.3e %.3e %+.3f %s\n", kk, zc[kk], T[kk], qᵛ[kk] / qs - 1,
                snap[:qᶜˡ][kk], snap[:qʳ][kk], snap[:nʳ][kk], w[kk], vels)
    end
    ρ = Float64(ρᵣ[k])
    q = MoistureMassFractions(Float64(qᵛ[k]), Float64(snap[:qᶜˡ][k] + snap[:qʳ][k]), Float64(snap[:qⁱ][k]))
    𝒰 = StaticEnergyState{Float64}(Float64(s[k]), q, Float64(zc[k]), Float64(pᵣ[k]))
    μc = NamedTuple{prognostic_names}(ntuple(n -> ρ * Float64(snap[specific_names[n]][k]), length(prognostic_names)))
    ℳ = AM.microphysical_state(p3cpu, ρ, μc, 𝒰, (; w = 0.0))
    rates = compute_p3_process_rates(p3cpu, ρ, ℳ, 𝒰, constants)
    println("   P3 process rates at the peak cell (nonzero):")
    for n in propertynames(rates)
        v = getproperty(rates, n)
        (v != 0 || !isfinite(v)) && @printf("      %-36s %+.4e\n", n, v)
    end
    r = p3_state_tendencies(p3cpu, ρ, ℳ, 𝒰, constants)
    println("   P3 tendencies per ρ:")
    for n in propertynames(r)
        v = getproperty(r, n)
        (v != 0 || !isfinite(v)) && @printf("      %-20s %+.4e\n", n, v / ρ)
    end
    reports[] == 4 && (println("PROBE STOP after 4 reports"); exit(0))
    return nothing
end

add_callback!(case.simulation, report, TimeInterval(30))
run!(case.simulation)
println("PROBE DONE")
