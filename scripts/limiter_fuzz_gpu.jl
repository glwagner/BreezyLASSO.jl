# Fuzz Oceananigans' bounds-preserving limiter on the GPU with fields of tiny Float32 values
# (down to the subnormal range) interleaved with zeros, for three upper bounds, and compare
# with the CPU evaluation of the same fields.
#   julia --project scripts/limiter_fuzz_gpu.jl
using Oceananigans, CUDA, Random, Printf
using Oceananigans.Advection: materialize_advection, compute_bounds_preserving_limiter!
using Oceananigans.BoundaryConditions: fill_halo_regions!

FT = Float32
N = 64
grids = (GPU=RectilinearGrid(GPU(), FT; size=(N, N, N), x=(0, N), y=(0, N), z=(0, N), halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded)),
         CPU=RectilinearGrid(CPU(), FT; size=(N, N, N), x=(0, N), y=(0, N), z=(0, N), halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded)))
rng = MersenneTwister(7)
for trial in 1:6
    # log-uniform magnitudes 1e-45..1e-20 (Float32 subnormals start below 1.18e-38), 40 % zeros,
    # one trial with a decaying cloud-top-like vertical tail
    field = zeros(FT, N, N, N)
    for i in 1:N, j in 1:N, k in 1:N
        u = rand(rng)
        if trial == 6
            field[i, j, k] = k < 20 ? FT(1.5) : k < 40 ? FT(10.0^(-12 - 1.6 * (k - 20)) * (1 + 0.5rand(rng))) : zero(FT)
        elseif u < 0.4
            field[i, j, k] = zero(FT)
        else
            field[i, j, k] = FT(10.0^(-45 + 25 * rand(rng)))
        end
    end
    for (label, bounds) in (("(0, Inf)", (0.0, Inf)), ("(0, 1e15)", (0.0, 1e15)), ("(0, 1)", (0.0, 1.0)))
        results = Dict{Symbol, Any}()
        for (arch, grid) in pairs(grids)
            scheme = materialize_advection(WENO(FT; order=5, bounds), grid)
            c = CenterField(grid)
            set!(c, field)
            fill_halo_regions!(c)
            compute_bounds_preserving_limiter!(scheme, grid, c)
            results[arch] = Array(interior(scheme.bounds.limiter))
        end
        n_gpu = count(!isfinite, results[:GPU]); n_cpu = count(!isfinite, results[:CPU])
        finite = isfinite.(results[:GPU]) .& isfinite.(results[:CPU])
        @printf("trial %d bounds %-10s: non-finite θ GPU %d, CPU %d; max |GPU-CPU| %.2e\n", trial, label, n_gpu, n_cpu,
                maximum(abs.(results[:GPU][finite] .- results[:CPU][finite]); init=0f0))
        if n_gpu > 0
            idx = findall(!isfinite, results[:GPU])[1:min(3, end)]
            for c in idx
                i, j, k = Tuple(c)
                println("   GPU NaN at ", (i, j, k), " field column k-2..k+2: ", [field[i, j, kk] for kk in max(k-2,1):min(k+2,N)],
                        " x-neighbours: ", [field[mod1(ii, N), j, k] for ii in i-2:i+2], " CPU θ = ", results[:CPU][i, j, k])
            end
        end
    end
end
println("FUZZ DONE")
