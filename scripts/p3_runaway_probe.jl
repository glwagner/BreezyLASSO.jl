# Catch the P3 runaway (rain/temperature explosion) at its onset: snapshot the state every
# 30 s, and when the maximum temperature exceeds the initial maximum by > 2 K, print the
# hottest cell's current and previous state and evaluate the P3 tendency bundle there.
#   julia --project scripts/p3_runaway_probe.jl <cpu|gpu> <Float32|Float64> <max_dt> [minutes] [microphysics] [off-switches]
using BreezyLASSO, Breeze, Oceananigans, Oceananigans.Units, CUDA, Printf, Statistics
using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.Thermodynamics: StaticEnergyState, MoistureMassFractions
using Breeze.Microphysics.PredictedParticleProperties: p3_state_tendencies, P3Microphysics, CloudDroplets

arch = lowercase(ARGS[1]) == "gpu" ? GPU() : CPU()
FT = ARGS[2] == "Float64" ? Float64 : Float32
max_Δt = parse(Float64, ARGS[3])
minutes = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 45.0
scheme = length(ARGS) ≥ 5 ? Symbol(ARGS[5]) : :p3_n75
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

case = lasso_ena_simulation(data; preset=:covert_public_bin, arch, FT, Nx=32, Ny=32, Lx=1120, Ly=1120,
                            z_faces=lasso_ena_vertical_faces(), microphysics=scheme, stop_time=minutes*60,
                            Δt=min(1.0, max_Δt), max_Δt, write_output=false, progress_interval=5minutes, switches...)
model = case.model
μ = model.microphysical_fields
zc = Array(znodes(model.grid, Center()))
ρᵣ = Array(interior(model.dynamics.reference_state.density, 1, 1, :))
pᵣ = Array(interior(model.dynamics.reference_state.pressure, 1, 1, :))
names = (:T, :s, :qᵛ, :qᶜˡ, :qʳ, :nʳ, :ρqʳ, :ρnʳ)
getfield_(m, n) = n === :T ? m.temperature : n === :s ? m.formulation.specific_energy : m.microphysical_fields[n]
snapshot(m) = Dict(n => Array(interior(getfield_(m, n))) for n in names)
T0max = maximum(model.temperature)
prev = Ref(snapshot(model))
p3cpu = P3Microphysics(Float64; cloud=CloudDroplets(Float64; number_concentration=75e6))
constants = ThermodynamicConstants(Float64)
triggered = Ref(false)
function watch(sim)
    m = sim.model
    Tmax = maximum(m.temperature)
    @printf("t=%6.2f min | Tmax %.2f | qʳ max %.2e | max|w| %.2f\n", m.clock.time/60, Tmax, maximum(μ.qʳ), maximum(abs, m.velocities.w))
    if Tmax > T0max + 2 && !triggered[]
        triggered[] = true
        cur = snapshot(m)
        idx = argmax(cur[:T]); i, j, k = Tuple(idx)
        println("HOT CELL (i,j,k) = ", (i, j, k), " z = ", zc[k], " p = ", pᵣ[k], " ρ = ", ρᵣ[k])
        for n in names
            @printf("   %-5s prev %.6e  now %.6e\n", n, prev[][n][i, j, k], cur[n][i, j, k])
        end
        w = Array(interior(m.velocities.w)); @printf("   w below/above: %.3f %.3f\n", w[i, j, k], w[i, j, min(k+1, end)])
        wʳ = Array(interior(μ.wʳ)); qv = cur[:qᵛ]
        println("   column at (i,j): k, z, T, qᵛ, qᶜˡ, qʳ, nʳ, wʳ(face k)")
        for kk in 1:6
            @printf("      %2d %7.1f %8.3f %.4e %.3e %.3e %.3e %.3f\n", kk, zc[kk], cur[:T][i,j,kk], qv[i,j,kk], cur[:qᶜˡ][i,j,kk], cur[:qʳ][i,j,kk], cur[:nʳ][i,j,kk], wʳ[i,j,kk])
        end
        for (label, snap) in (("prev", prev[]), ("now", cur))
            qᵛ = Float64(snap[:qᵛ][i,j,k]); qᶜˡ = Float64(snap[:qᶜˡ][i,j,k]); qʳ = Float64(snap[:qʳ][i,j,k]); nʳ = Float64(snap[:nʳ][i,j,k])
            ρ = Float64(ρᵣ[k])
            q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, 0.0)
            𝒰 = StaticEnergyState{Float64}(Float64(snap[:s][i,j,k]), q, Float64(zc[k]), Float64(pᵣ[k]))
            μc = (; ρqᶜˡ = ρ*qᶜˡ, ρqʳ = ρ*qʳ, ρnʳ = ρ*nʳ, ρqⁱ = 0.0, ρnⁱ = 0.0, ρqᶠ = 0.0, ρbᶠ = 0.0, ρqʷⁱ = 0.0)
            ℳ = AM.microphysical_state(p3cpu, ρ, μc, 𝒰, (; w = 0.0))
            r = p3_state_tendencies(p3cpu, ρ, ℳ, 𝒰, constants)
            println("   P3 tendencies at ", label, " state (per ρ, kg/kg/s or 1/kg/s):")
            for n in propertynames(r)
                v = getproperty(r, n)
                abs(v) > 0 && @printf("      %-16s %.4e\n", n, v / ρ)
            end
        end
    end
    prev[] = snapshot(m)
end
add_callback!(case.simulation, watch, TimeInterval(30))
run!(case.simulation)
println("PROBE DONE, triggered = ", triggered[])
