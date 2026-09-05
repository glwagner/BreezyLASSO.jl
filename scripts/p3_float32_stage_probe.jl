# Catch the Float32 P3-aer2 non-finite value at the Runge–Kutta stage where it first appears.
# A callback at the update-state callsite (which runs at the top of every stage, before the
# stage's tendencies are formed) inspects (1) the prognostic fields — the state the previous
# stage produced — and (2) the previous stage's tendencies Gⁿ. When one of them goes
# non-finite, the pre-stage snapshot is used to evaluate the P3 tendency bundle at that cell
# in Float32 and Float64.
#   julia --project scripts/p3_float32_stage_probe.jl [Float32|Float64] [off-switches]
using BreezyLASSO, Breeze, Oceananigans, Oceananigans.Units, Printf
using Oceananigans: Callback, UpdateStateCallsite
using Oceananigans.TimeSteppers: time_step!
using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.Thermodynamics: StaticEnergyState, MoistureMassFractions, ThermodynamicConstants
using Breeze.Microphysics.PredictedParticleProperties: p3_state_tendencies

FT = length(ARGS) ≥ 1 && ARGS[1] == "Float64" ? Float64 : Float32
off = length(ARGS) ≥ 2 ? split(ARGS[2], ",") : String[]
"all" ∈ off && (off = ["vadv", "thermo", "nudging", "sponge", "geo", "upper", "radiation", "surface", "closure", "enthalpy", "proj"])
Δt = 0.5
scheme = :p3_aer2
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
"proj" ∉ off && (switches[:aerosol_replenishment] = :diagnostic_ccn)
println("off switches: ", off)
case = lasso_ena_simulation(data; preset=:covert_public_bin, arch=CPU(), FT, Nx=8, Ny=8, Lx=280, Ly=280,
                            z_faces=lasso_ena_vertical_faces(), microphysics=scheme, stop_time=2Δt,
                            Δt, max_Δt=Δt, write_output=false, switches...)
model = case.model
μ = model.microphysical_fields
prognostic_names = AM.prognostic_field_names(model.microphysics)
specific_names = map(AM.specific_field_name, prognostic_names)
zc = Array(znodes(model.grid, Center()))
ρᵣ = Array(interior(model.dynamics.reference_state.density, 1, 1, :))
pᵣ = Array(interior(model.dynamics.reference_state.pressure, 1, 1, :))

field_of(m, n) = n === :T ? m.temperature : n === :s ? m.formulation.specific_energy : m.microphysical_fields[n]
snapshot(m) = Dict(n => copy(Array(interior(field_of(m, n)))) for n in (:T, :s, :qᵛ, specific_names...))
density_snapshot(m) = Dict(n => copy(Array(interior(f))) for (n, f) in pairs(Oceananigans.prognostic_fields(m)))

function nonfinite_report(label, fields)
    found = nothing
    for (name, a) in fields
        nb = count(!isfinite, a)
        nb == 0 && continue
        idx = findall(!isfinite, a)
        ks = sort(unique(map(c -> c[3], idx)))
        println("   $label.$name: $nb non-finite, levels $(first(ks))..$(last(ks))")
        found === nothing && (found = (name, Tuple(idx[1])))
    end
    return found
end

function evaluate_tendencies(FTe, snap, i, j, k)
    p3, _ = BreezyLASSO.build_microphysics(FTe, scheme; droplet_number=75e6, surface_density=Float64(ρᵣ[1]))
    constants = ThermodynamicConstants(FTe)
    ρ = FTe(ρᵣ[k])
    qᵛ = FTe(snap[:qᵛ][i, j, k])
    qˡ = FTe(snap[:qᶜˡ][i, j, k] + snap[:qʳ][i, j, k])
    qⁱ = FTe(snap[:qⁱ][i, j, k])
    q = MoistureMassFractions(qᵛ, qˡ, qⁱ)
    𝒰 = StaticEnergyState{FTe}(FTe(snap[:s][i, j, k]), q, FTe(zc[k]), FTe(pᵣ[k]))
    μc = NamedTuple{prognostic_names}(ntuple(n -> ρ * FTe(snap[specific_names[n]][i, j, k]), length(prognostic_names)))
    ℳ = AM.microphysical_state(p3, ρ, μc, 𝒰, (; w = zero(FTe)))
    println("   microphysical state ($FTe): ", ℳ)
    r = p3_state_tendencies(p3, ρ, ℳ, 𝒰, constants)
    println("   P3 tendencies ($FTe), per ρ:")
    for n in propertynames(r)
        v = getproperty(r, n)
        (abs(v) > 0 || !isfinite(v)) && @printf("      %-16s %.6e\n", n, v / ρ)
    end
end

const stage_counter = Ref(0)
const previous = Ref(snapshot(model))
const previous_density = Ref(density_snapshot(model))
const triggered = Ref(false)

function stage_watch(m)
    triggered[] && return nothing
    stage_counter[] += 1
    G = Dict(n => Array(interior(g)) for (n, g) in pairs(m.timestepper.Gⁿ))
    state = density_snapshot(m)
    println("stage entry $(stage_counter[]) (iteration $(m.clock.iteration), stage $(m.clock.stage)):")
    bad_G = nonfinite_report("Gⁿ(previous stage)", G)
    bad_state = nonfinite_report("state", state)
    if bad_G === nothing && bad_state === nothing
        println("   all finite")
        previous[] = snapshot(m)
        previous_density[] = state
        return nothing
    end
    triggered[] = true
    name, (i, j, k) = something(bad_G, bad_state)
    println("Cell $((i, j, k)), z = $(zc[k]) m — pre-stage state (snapshot at the previous stage entry):")
    for n in sort(collect(keys(previous[])))
        @printf("   %-5s %.6e\n", n, previous[][n][i, j, k])
    end
    println("Current density state at the cell:")
    for n in sort(collect(keys(state)))
        @printf("   %-5s %.6e\n", n, state[n][i, j, k])
    end
    println("Previous-stage tendencies at the cell:")
    for n in sort(collect(keys(G)))
        @printf("   G.%-5s %.6e\n", n, G[n][i, j, k])
    end
    evaluate_tendencies(Float32, previous[], i, j, k)
    evaluate_tendencies(Float64, previous[], i, j, k)

    # Restore the pre-stage state exactly, recompute the tendencies from it, and split the
    # non-finite tendency into its advection and P3 parts cell by cell.
    println("Restoring the pre-stage state and recomputing tendencies:")
    for (n, f) in pairs(Oceananigans.prognostic_fields(m))
        set!(f, previous_density[][n])
    end
    Oceananigans.TimeSteppers.update_state!(m)
    G2 = Dict(n => Array(interior(g)) for (n, g) in pairs(m.timestepper.Gⁿ))
    nonfinite_report("Gⁿ(recomputed)", G2)
    μ = m.microphysical_fields
    ρ_field = m.dynamics.reference_state.density
    U = m.velocities
    fields_ = Oceananigans.fields(m)
    for gname in (:ρnᵃ, :ρnᶜˡ)
        haskey(G2, gname) || continue
        idx = findall(!isfinite, G2[gname])
        isempty(idx) && continue
        println("Term split for Gⁿ.$gname at the first $(min(4, length(idx))) non-finite cells:")
        for c in idx[1:min(4, end)]
            i, j, k = Tuple(c)
            specific = AM.specific_field_name(gname)
            cfield = μ[specific]
            adv = -AM.div_ρUc(i, j, k, m.grid, m.advection[gname], ρ_field, U, cfield)
            f = m.forcing[gname]
            fval = f(i, j, k, m.grid, m.clock, fields_)
            snap = snapshot(m)
            S = snap[:qᵛ][i, j, k] / Breeze.Thermodynamics.saturation_specific_humidity(snap[:T][i, j, k], Float64(ρ_field[i, j, k]), ThermodynamicConstants(Float64), Breeze.Thermodynamics.PlanarLiquidSurface()) - 1
            @printf("   cell %s z=%.1f: advection %s | forcing %s | S=%.4e nᵃ=%.4e nᶜˡ=%.4e qᶜˡ=%.4e | column nᵃ (k-3..k+3): %s\n",
                    string((i, j, k)), zc[k], string(adv), string(fval), S, snap[:nᵃ][i, j, k], snap[:nᶜˡ][i, j, k], snap[:qᶜˡ][i, j, k],
                    string([snap[:nᵃ][i, j, kk] for kk in max(k-3, 1):min(k+3, size(m.grid, 3))]))
            evaluate_tendencies(Float32, snap, i, j, k)
        end
    end
    return nothing
end

callbacks = [Callback(stage_watch; callsite=UpdateStateCallsite())]
for iteration in 1:2
    time_step!(model, FT(Δt); callbacks)
    triggered[] && break
end
println("PROBE DONE, triggered = ", triggered[])
