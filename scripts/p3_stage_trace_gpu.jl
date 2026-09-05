# Per-stage trace of the first non-finite rain-number tendency at production scale (GPU).
# At every Runge–Kutta stage entry (update-state callsite, before the stage's tendencies are
# formed) the previous stage's tendency Gⁿ and the prognostic state are scanned; on the first
# non-finite value the pre-stage snapshot column is printed and the P3 process rates at that
# cell are evaluated on the CPU in Float64 and Float32.
#   PROBE_GRID=covert PROBE_MOMENTS=positive PROBE_NX=256 julia --project scripts/p3_stage_trace_gpu.jl <Float32|Float64> <dt> [minutes] [microphysics]
using BreezyLASSO, Breeze, Oceananigans, Oceananigans.Units, CUDA, Printf, Statistics
using Oceananigans: Callback, UpdateStateCallsite
using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.Thermodynamics: StaticEnergyState, MoistureMassFractions, ThermodynamicConstants
using Breeze.Microphysics.PredictedParticleProperties: p3_state_tendencies, compute_p3_process_rates
using Oceananigans.Advection: materialize_advection, bounds_preserving_limiter, reconstruction_extrema_x, reconstruction_extrema_y, reconstruction_extrema_z
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Architectures: on_architecture
using JLD2: jldsave

FT = ARGS[1] == "Float64" ? Float64 : Float32
Δt = parse(Float64, ARGS[2])
minutes = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 5.0
scheme = length(ARGS) ≥ 4 ? Symbol(ARGS[4]) : :p3_n75
N = parse(Int, get(ENV, "PROBE_NX", "256"))
z_faces = get(ENV, "PROBE_GRID", "covert") == "covert" ? covert_public_bin_vertical_faces() : lasso_ena_vertical_faces()
switches = Dict{Symbol, Any}()
get(ENV, "PROBE_MOMENTS", "positive") == "positive" ? (switches[:moment_advection] = :positive) : (switches[:moment_advection] = :plain)
scheme === :p3_aer2 && (switches[:aerosol_replenishment] = :diagnostic_ccn)
data = joinpath(@__DIR__, "..", "data", "covert2022_bin")
case = lasso_ena_simulation(data; preset=:covert_public_bin, arch=GPU(), FT, Nx=N, Ny=N, Lx=35N, Ly=35N, z_faces,
                            microphysics=scheme, stop_time=minutes*60, Δt, max_Δt=Δt, write_output=false,
                            progress_interval=1minute, switches...)
model = case.model
μ = model.microphysical_fields
prognostic_names = AM.prognostic_field_names(model.microphysics)
specific_names = map(AM.specific_field_name, prognostic_names)
zc = Array(znodes(model.grid, Center()))
ρᵣ = Array(interior(model.dynamics.reference_state.density, 1, 1, :))
pᵣ = Array(interior(model.dynamics.reference_state.pressure, 1, 1, :))
Nz = size(model.grid, 3)

# Only the fields that matter for the rain-number budget are snapshotted every stage.
watched = (:ρnʳ, :ρqʳ, :nʳ, :qʳ, :qᶜˡ, :qᵛ, :wʳₙ, :wʳ)
previous = Dict(n => Array(interior(μ[n])) for n in watched)
previous_T = Array(interior(model.temperature))
previous_s = Array(interior(model.formulation.specific_energy))
previous_prognostic = Dict(n => Array(interior(f)) for (n, f) in pairs(Oceananigans.prognostic_fields(model)))
stage_count = Ref(0)
done = Ref(false)

function column(fields, i, j, k)
    for kk in max(k - 3, 1):min(k + 3, Nz)
        vals = join([@sprintf("%s=%.3e", n, fields[n][i, j, kk]) for n in watched], " ")
        @printf("         k=%3d z=%7.1f T=%.3f %s\n", kk, zc[kk], previous_T[i, j, kk], vals)
    end
end

function evaluate(FTe, i, j, k)
    p3, _ = BreezyLASSO.build_microphysics(FTe, scheme; droplet_number=75e6, surface_density=Float64(ρᵣ[1]))
    constants = ThermodynamicConstants(FTe)
    ρ = FTe(ρᵣ[k])
    specific = Dict(n => Array(interior(μ[n]))[i, j, k] for n in specific_names)  # current specific fields (finite pre-stage values)
    q = MoistureMassFractions(FTe(previous[:qᵛ][i, j, k]), FTe(previous[:qᶜˡ][i, j, k] + previous[:qʳ][i, j, k]), FTe(specific[:qⁱ]))
    𝒰 = StaticEnergyState{FTe}(FTe(previous_s[i, j, k]), q, FTe(zc[k]), FTe(pᵣ[k]))
    values = Dict(n => FTe(get(previous, n, nothing) === nothing ? specific[n] : previous[n][i, j, k]) for n in specific_names)
    μc = NamedTuple{prognostic_names}(ntuple(n -> ρ * values[specific_names[n]], length(prognostic_names)))
    ℳ = AM.microphysical_state(p3, ρ, μc, 𝒰, (; w = zero(FTe)))
    println("   state ($FTe): ", ℳ)
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
    println("   P3 process rates ($FTe), nonzero or non-finite:")
    for n in propertynames(rates)
        v = getproperty(rates, n)
        (v != 0 || !isfinite(v)) && @printf("      %-36s %+.4e\n", n, v)
    end
    r = p3_state_tendencies(p3, ρ, ℳ, 𝒰, constants)
    println("   P3 tendencies ($FTe) per ρ:")
    for n in propertynames(r)
        v = getproperty(r, n)
        (v != 0 || !isfinite(v)) && @printf("      %-20s %+.4e\n", n, v / ρ)
    end
end

function stage_watch(m)
    done[] && return nothing
    stage_count[] += 1
    G = Array(interior(m.timestepper.Gⁿ.ρnʳ))
    state = Array(interior(μ.ρnʳ))
    bad_G = findall(!isfinite, G)
    bad_state = findall(!isfinite, state)
    if isempty(bad_G) && isempty(bad_state)
        for n in watched
            copyto!(previous[n], Array(interior(μ[n])))
        end
        copyto!(previous_T, Array(interior(m.temperature)))
        copyto!(previous_s, Array(interior(m.formulation.specific_energy)))
        for (n, f) in pairs(Oceananigans.prognostic_fields(m))
            copyto!(previous_prognostic[n], Array(interior(f)))
        end
        return nothing
    end
    done[] = true
    println("NON-FINITE at stage entry $(stage_count[]) (iteration $(m.clock.iteration), clock stage $(m.clock.stage), t = $(m.clock.time)):")
    println("   Gⁿ.ρnʳ non-finite cells: ", length(bad_G), "  ρnʳ non-finite cells: ", length(bad_state))
    cells = isempty(bad_G) ? bad_state : bad_G
    for c in cells[1:min(3, end)]
        i, j, k = Tuple(c)
        println("   cell ", (i, j, k), " z = ", zc[k], " m; Gⁿ.ρnʳ = ", G[i, j, k], "; pre-stage column:")
        column(previous, i, j, k)
        evaluate(Float64, i, j, k)
        evaluate(Float32, i, j, k)
    end
    θ = Array(interior(m.advection[:ρnʳ].bounds.limiter))
    println("   limiter θ non-finite cells now: ", count(!isfinite, θ))

    # Restore the pre-stage prognostic state exactly, recompute the tendencies, and split the
    # rain-number tendency into its advection and forcing parts at the failing cells.
    for (n, f) in pairs(Oceananigans.prognostic_fields(m))
        set!(f, previous_prognostic[n])
    end
    Oceananigans.TimeSteppers.update_state!(m)
    G2 = Array(interior(m.timestepper.Gⁿ.ρnʳ))
    bad2 = findall(!isfinite, G2)
    println("   Gⁿ.ρnʳ recomputed from the restored pre-stage state: non-finite cells ", length(bad2))
    grid = m.grid
    ρ_field = AM.total_density(m.dynamics)
    Uᵖ = AM.microphysical_velocities(m.microphysics, μ, :ρnʳ)
    Uᵗ = AM.sum_of_velocities(m.velocities, Uᵖ)
    fields_ = Oceananigans.fields(m)
    adv_scheme = m.advection[:ρnʳ]
    for c in bad2[1:min(3, end)]
        i, j, k = Tuple(c)
        CUDA.@allowscalar begin
            adv = -AM.div_ρUc(i, j, k, grid, adv_scheme, ρ_field, Uᵗ, μ.nʳ)
            frc = m.forcing[:ρnʳ](i, j, k, grid, m.clock, fields_)
            θz = [adv_scheme.bounds.limiter[i, j, kk] for kk in k-3:k+3]
            θx = [adv_scheme.bounds.limiter[ii, j, k] for ii in i-3:i+3]
            θy = [adv_scheme.bounds.limiter[i, jj, k] for jj in j-3:j+3]
            nz = [μ.nʳ[i, j, kk] for kk in k-3:k+3]
            nx = [μ.nʳ[ii, j, k] for ii in i-3:i+3]
            ny = [μ.nʳ[i, jj, k] for jj in j-3:j+3]
            wz = [Uᵗ.w[i, j, kk] for kk in k-2:k+2]
            wn = [μ.wʳₙ[i, j, kk] for kk in k-2:k+2]
            ux = [Uᵗ.u[ii, j, k] for ii in i-2:i+2]
            vy = [Uᵗ.v[i, jj, k] for jj in j-2:j+2]
            println("   cell ", (i, j, k), ": advection ", adv, " forcing ", frc, " Gⁿ ", G2[i, j, k])
            println("      nʳ z-stencil ", nz, "\n      nʳ x-stencil ", nx, "\n      nʳ y-stencil ", ny)
            println("      θ z ", θz, "\n      θ x ", θx, "\n      θ y ", θy)
            println("      w+wʳₙ faces ", wz, " wʳₙ ", wn, " u ", ux, " v ", vy)
        end
    end
    # The limiter's own ingredients at every non-finite θ cell, on the GPU and on the CPU from
    # the same specific field.
    θ_now = Array(interior(adv_scheme.bounds.limiter))
    nʳ_host = Array(interior(μ.nʳ))
    cpu_grid = on_architecture(CPU(), grid)
    cpu_scheme = materialize_advection(WENO(FT; order=5, bounds=(0.0, Inf)), cpu_grid)
    cpu_field = CenterField(cpu_grid)
    set!(cpu_field, nʳ_host)
    fill_halo_regions!(cpu_field)
    ω̂₁ = adv_scheme.bounds.maximum_courant_number
    for c in findall(!isfinite, θ_now)[1:min(4, end)]
        i, j, k = Tuple(c)
        gpu = CUDA.@allowscalar begin
            cᵢ = μ.nʳ[i, j, k]
            m, M = cᵢ, cᵢ
            m, M = reconstruction_extrema_x(i, j, k, grid, adv_scheme, μ.nʳ, m, M, ω̂₁)
            mx, Mx = m, M
            m, M = reconstruction_extrema_y(i, j, k, grid, adv_scheme, μ.nʳ, m, M, ω̂₁)
            my, My = m, M
            m, M = reconstruction_extrema_z(i, j, k, grid, adv_scheme, μ.nʳ, m, M, ω̂₁)
            (; cᵢ, mx, Mx, my, My, m, M, θ=bounds_preserving_limiter(i, j, k, grid, adv_scheme, μ.nʳ))
        end
        cᵢ = cpu_field[i, j, k]
        m, M = cᵢ, cᵢ
        m, M = reconstruction_extrema_x(i, j, k, cpu_grid, cpu_scheme, cpu_field, m, M, ω̂₁)
        mx, Mx = m, M
        m, M = reconstruction_extrema_y(i, j, k, cpu_grid, cpu_scheme, cpu_field, m, M, ω̂₁)
        my, My = m, M
        m, M = reconstruction_extrema_z(i, j, k, cpu_grid, cpu_scheme, cpu_field, m, M, ω̂₁)
        cpu = (; cᵢ, mx, Mx, my, My, m, M, θ=bounds_preserving_limiter(i, j, k, cpu_grid, cpu_scheme, cpu_field))
        println("   θ ingredients at ", (i, j, k), ":\n      GPU ", gpu, "\n      CPU ", cpu)
        ε₂ = convert(FT, 1e-20)
        println("      CPU denominators: M-c+ε₂ = ", cpu.M - cpu.cᵢ + ε₂, "  m-c+ε₂ = ", cpu.m - cpu.cᵢ + ε₂)
    end
    jldsave(joinpath("output", "stage_trace_$(get(ENV, "SLURM_JOB_ID", "local")).jld2"); nʳ=nʳ_host, θ=θ_now,
            ρnʳ=Array(interior(μ.ρnʳ)), ρqʳ=Array(interior(μ.ρqʳ)))
    error("non-finite rain-number tendency")
end

add_callback!(case.simulation, stage_watch, IterationInterval(1); callsite=UpdateStateCallsite())
run!(case.simulation)
println("TRACE DONE, no non-finite tendency in $(stage_count[]) stages")
