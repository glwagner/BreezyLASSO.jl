# Run one LASSO-ENA / Covert case from the command line.
#
#   julia --project scripts/run_case.jl --data data/covert2022_bin --preset covert_public_bin \
#         --microphysics p3_n75 --arch gpu --Nx 256 --Ny 256 --hours 6 --output output/p3_n75
#
# Every run writes <output>/provenance.toml (inputs + checksums + configuration).

using BreezyLASSO, Breeze, Oceananigans, Oceananigans.Units, CUDA, Dates, Random

function parse_args(args)
    opts = Dict{String, String}()
    i = 1
    while i ≤ length(args)
        key = args[i]
        startswith(key, "--") || error("unexpected argument $key")
        opts[key[3:end]] = args[i+1]
        i += 2
    end
    return opts
end

opts = parse_args(ARGS)
getopt(k, default) = get(opts, k, default)

data = getopt("data", "data/covert2022_bin")
preset = Symbol(getopt("preset", "covert_public_bin"))
arch = lowercase(getopt("arch", "cpu")) == "gpu" ? GPU() : CPU()
FT = getopt("float", "Float32") == "Float64" ? Float64 : Float32
seed = parse(Int, getopt("seed", "1234"))

kw = Dict{Symbol, Any}()
haskey(opts, "microphysics") && (kw[:microphysics] = Symbol(opts["microphysics"]))   # otherwise the preset decides
haskey(opts, "Nx") && (kw[:Nx] = parse(Int, opts["Nx"]))
haskey(opts, "Ny") && (kw[:Ny] = parse(Int, opts["Ny"]))
haskey(opts, "Lx") && (kw[:Lx] = parse(Float64, opts["Lx"]))
haskey(opts, "Ly") && (kw[:Ly] = parse(Float64, opts["Ly"]))
haskey(opts, "hours") && (kw[:stop_time] = parse(Float64, opts["hours"]) * 3600)
haskey(opts, "radiation") && (kw[:radiation] = opts["radiation"] == "nothing" ? nothing : Symbol(opts["radiation"]))
haskey(opts, "surface") && (kw[:surface] = Symbol(opts["surface"]))
haskey(opts, "nudging") && (kw[:wind_nudging_timescale] = opts["nudging"] == "nothing" ? nothing : parse(Float64, opts["nudging"]))
haskey(opts, "vertical_advection") && (kw[:vertical_advection] = opts["vertical_advection"] == "nothing" ? nothing : Symbol(opts["vertical_advection"]))
haskey(opts, "p3_initialization") && (kw[:p3_initialization] = Symbol(opts["p3_initialization"]))
haskey(opts, "aerosol_replenishment") && (kw[:aerosol_replenishment] = opts["aerosol_replenishment"] == "nothing" ? nothing :
                                          opts["aerosol_replenishment"] == "diagnostic_ccn" ? :diagnostic_ccn : parse(Float64, opts["aerosol_replenishment"]))
haskey(opts, "cfl") && (kw[:cfl] = parse(Float64, opts["cfl"]))
haskey(opts, "max_dt") && (kw[:max_Δt] = parse(Float64, opts["max_dt"]))
haskey(opts, "lasso_grid") && opts["lasso_grid"] == "true" && (kw[:z_faces] = lasso_ena_vertical_faces())
haskey(opts, "profile_interval") && (kw[:profile_interval] = parse(Float64, opts["profile_interval"]))
haskey(opts, "slice_interval") && (kw[:slice_interval] = parse(Float64, opts["slice_interval"]))

microphysics_label = get(opts, "microphysics", preset === :lasso_ena_official ? "p3_aer2" : "p3_n75")
output_dir = getopt("output", "output/$(preset)_$(microphysics_label)")
@info "Building $preset on $(typeof(arch)) with $FT" opts
case = lasso_ena_simulation(data; preset, arch, FT, output_dir,
                            perturbation = InitialPerturbation(seed=seed), kw...)
mkpath(output_dir)
provenance = write_provenance(joinpath(output_dir, "provenance.toml"), case;
                              extra = (; command = join(ARGS, " "), hostname = gethostname(),
                                         gpu = CUDA.functional() ? CUDA.name(CUDA.device()) : "none"))
@info "Provenance written to $provenance"
@info "Label: $(case.config.label)"
println(case.model)

wall = time()
run!(case.simulation)
@info "Finished in $(round((time() - wall) / 60, digits=1)) minutes; output in $output_dir"
