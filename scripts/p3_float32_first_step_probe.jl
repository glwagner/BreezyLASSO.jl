# Float32 P3-aer2 went non-finite at iteration 1 on the GPU (jobs 847/848) while Float64 ran
# for 20 minutes. Reproduce on a small CPU grid: take one fixed step, locate the first
# non-finite cell, print its pre-step state, and evaluate the P3 tendency bundle at that
# state in Float32 and Float64 so a precision-specific overflow shows up directly.
#   julia --project scripts/p3_float32_first_step_probe.jl [Float32|Float64] [dt] [microphysics] [off-switches]
# off-switches: comma-separated subset of vadv,thermo,nudging,sponge,geo,upper,radiation,surface,
# closure,enthalpy,proj (proj = no diagnostic-CCN projection), or "all" for every one of them.
using BreezyLASSO, Breeze, Oceananigans, Oceananigans.Units, Printf
using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.Thermodynamics: StaticEnergyState, MoistureMassFractions, ThermodynamicConstants
using Breeze.Microphysics.PredictedParticleProperties: p3_state_tendencies

FT = length(ARGS) ≥ 1 && ARGS[1] == "Float64" ? Float64 : Float32
Δt = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 0.5
scheme = length(ARGS) ≥ 3 ? Symbol(ARGS[3]) : :p3_aer2
off = length(ARGS) ≥ 4 ? split(ARGS[4], ",") : String[]
"all" ∈ off && (off = ["vadv", "thermo", "nudging", "sponge", "geo", "upper", "radiation", "surface", "closure", "enthalpy", "proj"])
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
scheme === :p3_aer2 && "proj" ∉ off && (switches[:aerosol_replenishment] = :diagnostic_ccn)
println("off switches: ", off)
case = lasso_ena_simulation(data; preset=:covert_public_bin, arch=CPU(), FT, Nx=8, Ny=8, Lx=280, Ly=280,
                            z_faces=lasso_ena_vertical_faces(), microphysics=scheme, stop_time=4Δt,
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

function first_nonfinite(m)
    for (name, f) in pairs(Oceananigans.prognostic_fields(m))
        a = Array(interior(f))
        all(isfinite, a) && continue
        idx = findfirst(!isfinite, a)
        return name, Tuple(idx), count(!isfinite, a)
    end
    return nothing
end

function evaluate_tendencies(FTe, snap, i, j, k)
    p3, _ = BreezyLASSO.build_microphysics(FTe, scheme; droplet_number=75e6, surface_density=Float64(ρᵣ[1]))
    constants = ThermodynamicConstants(FTe)
    ρ = FTe(ρᵣ[k])
    qᵛ = FTe(snap[:qᵛ][i, j, k])
    qˡ = FTe(snap[:qᶜˡ][i, j, k] + snap[:qʳ][i, j, k])
    qⁱ = :qⁱ ∈ specific_names ? FTe(snap[:qⁱ][i, j, k]) : zero(FTe)
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

prev = snapshot(model)
println("Δt = $Δt, FT = $FT, scheme = $scheme, prognostics = $prognostic_names")
for iteration in 1:4
    time_step!(model, FT(Δt))
    bad = first_nonfinite(model)
    if bad === nothing
        cur = snapshot(model)
        @printf("iteration %d finite | max qᶜˡ %.3e max nᶜˡ %.3e T ∈ [%.2f, %.2f]\n", iteration,
                maximum(cur[:qᶜˡ]), maximum(cur[:nᶜˡ]), extrema(cur[:T])...)
        global prev = cur
        continue
    end
    name, (i, j, k), n_bad = bad
    println("FIRST NONFINITE: $name at iteration $iteration, cell $((i, j, k)), z = $(zc[k]) m, $n_bad cells")
    for (fname, f) in pairs(Oceananigans.prognostic_fields(model))
        a = Array(interior(f)); nb = count(!isfinite, a)
        nb > 0 && println("   $fname: $nb non-finite")
    end
    println("Pre-step state at the cell:")
    for n in keys(prev)
        @printf("   %-5s %.6e\n", n, prev[n][i, j, k])
    end
    cur = snapshot(model)
    println("Post-step state at the cell:")
    for n in keys(cur)
        @printf("   %-5s %.6e\n", n, cur[n][i, j, k])
    end
    evaluate_tendencies(Float32, prev, i, j, k)
    evaluate_tendencies(Float64, prev, i, j, k)
    break
end
println("PROBE DONE")
