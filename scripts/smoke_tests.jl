# Staged smoke tests (CPU or GPU): forcing-only, dry dynamics, cloudy 1M / P3-N75 / P3-aer2.
#   julia --project scripts/smoke_tests.jl [cpu|gpu]
using BreezyLASSO, Breeze, Oceananigans, Oceananigans.Units, CUDA, Printf, Statistics

arch = length(ARGS) ≥ 1 && lowercase(ARGS[1]) == "gpu" ? GPU() : CPU()
data = joinpath(@__DIR__, "..", "data", "covert2022_bin")
small = (; Nx=32, Ny=32, Lx=1120, Ly=1120, z_faces=lasso_ena_vertical_faces(), FT=Float32, write_output=false, progress_interval=5minutes, max_Δt=10.0)

finite(model) = all(f -> all(isfinite, Array(interior(f))), values(Oceananigans.prognostic_fields(model)))

function report(name, case, seconds)
    t = time()
    run!(case.simulation)
    m = case.model
    @printf("%-40s finite=%s  max|w|=%.3f  LWP=%.1f g/m²  wall=%.0fs\n", name, finite(m), maximum(abs, m.velocities.w),
            1e3 * mean(Array(interior(liquid_water_path(m; species=:cloud)))), time() - t)
    return finite(m)
end

results = Bool[]
# 1. forcing-only: no closure, no radiation, no surface fluxes, one-moment control, 1 h
case = lasso_ena_simulation(data; preset=:covert_public_bin, arch, small..., microphysics=:one_moment,
                            radiation=nothing, surface=nothing, closure=nothing, stop_time=1hour, Δt=2.0)
push!(results, report("forcing-only (1M, 1 h)", case, 3600))
# 2. dry dynamics: geostrophic + nudging + subsidence + sponge, no moisture forcing
case = lasso_ena_simulation(data; preset=:covert_public_bin, arch, small..., microphysics=:one_moment,
                            radiation=nothing, surface=nothing, thermodynamic_tendencies=false,
                            wind_nudging_timescale=7200, stop_time=1hour, Δt=2.0)
push!(results, report("dry dynamics (nudging+subsidence, 1 h)", case, 3600))
# 3-5. cloudy 45 min: 1M-control, P3-N75, P3-aer2 (official-style aerosol projection as sensitivity)
for (scheme, extra) in ((:one_moment, NamedTuple()), (:p3_n75, NamedTuple()), (:p3_aer2, (; aerosol_replenishment=:diagnostic_ccn)))
    case = lasso_ena_simulation(data; preset=:covert_public_bin, arch, small..., microphysics=scheme, stop_time=45minutes, Δt=1.0, extra...)
    push!(results, report("cloudy $(scheme) (45 min)", case, 2700))
end
println(all(results) ? "ALL SMOKE TESTS FINITE" : "SMOKE TEST FAILURE")
exit(all(results) ? 0 : 1)
