# Replay the exact recorded LES controls with only closure and output location changed.
# A reduced smoke additionally changes horizontal size/extent, duration and disables outputs.
using BreezyLASSO, Breeze, Oceananigans, CUDA, TOML, Test
using Oceananigans.Units: second

const repository = normpath(joinpath(@__DIR__, "..", ".."))
const schemes = ("one_moment", "p3_n75", "p3_aer2")

function member_specification(index)
    scheme = schemes[index % 3 + 1]
    lasso_grid = index >= 3
    suffix = lasso_grid ? "_lassogrid" : ""
    control = "covert_public_bin_$(scheme)$(suffix)_posmom_s60_theta"
    folder = lasso_grid ? "lasso_grid_theta" : "covert_grid_theta"
    record = TOML.parsefile(joinpath(repository, "results", folder, control * "_provenance.toml"))
    return (; scheme, lasso_grid, control, record)
end

function smoke(index)
    member = member_specification(index)
    z_faces = member.lasso_grid ? lasso_ena_vertical_faces() : covert_public_bin_vertical_faces()
    aerosol_replenishment = member.scheme == "p3_aer2" ? :diagnostic_ccn : nothing
    case = lasso_ena_simulation(joinpath(repository, "data", "covert2022_bin");
        preset = :covert_public_bin, arch = GPU(), FT = Float32,
        microphysics = Symbol(member.scheme), closure = nothing,
        Nx = 32, Ny = 32, Lx = 1120, Ly = 1120, z_faces,
        formulation = :LiquidIcePotentialTemperature, moment_advection = :positive,
        aerosol_replenishment, perturbation = InitialPerturbation(seed=1234),
        stop_time = 60second, write_output = false)
    @test isnothing(case.model.closure)
    @test case.config.closure == "nothing"
    @test case.config.bounded_condensate_advection
    @test case.config.moment_advection == "positive"
    @test case.simulation.Δt == 0.5
    run!(case.simulation)
    for field in values(Oceananigans.prognostic_fields(case.model))
        @test all(isfinite, interior(field))
    end
    @info "WENO-only smoke passed" index member.control iteration=case.model.clock.iteration
    return nothing
end

if ARGS == ["smoke"]
    @testset "WENO-only LES: three members on both grids" begin
        for index in 0:5
            smoke(index)
            GC.gc(true)
            CUDA.reclaim()
        end
    end
else
    index = parse(Int, only(ARGS))
    0 <= index <= 5 || error("member index must be 0:5")
    member = member_specification(index)
    output = joinpath(repository, "output", member.control * "_weno_only")
    ispath(output) && error("Refusing to overwrite existing output: $output")
    # Recorded commands use whitespace-free tokens. Only these two options are changed.
    arguments = String.(split(member.record["extra"]["command"]))
    output_option = findfirst(==("--output"), arguments)
    isnothing(output_option) && error("Control command has no output argument")
    arguments[output_option + 1] = output
    append!(arguments, ["--closure", "none"])
    empty!(ARGS)
    append!(ARGS, arguments)
    @info "Matched WENO-only production" index member.control output arguments
    cd(repository)
    include(joinpath(repository, "scripts", "run_case.jl"))
    @test isnothing(case.model.closure)
    # All physics/configuration must agree with the control, except the requested change.
    generated = TOML.parsefile(joinpath(output, "provenance.toml"))
    for (key, value) in member.record["config"]
        key in ("closure", "output_dir", "label") && continue
        @test generated["config"][key] == value
    end
end
