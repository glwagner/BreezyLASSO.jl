# Animate the 10-minute slices of one or more runs: the cloud-liquid x–z section at y = Ly/2,
# w at 900 m and the liquid water path, with color ranges fixed over the whole series.
#   RESULTS_SUBDIR=covert_grid julia --project=analysis analysis/animate_slices.jl output/run_a output/run_b ...
# Writes results/<subdir>/animation_<run>.mp4 (6 frames per second, one frame per 10 simulated minutes).
using CairoMakie, JLD2, Oceananigans, Oceananigans.Units, Printf

runs = ARGS
results_dir = joinpath(@__DIR__, "..", "results", get(ENV, "RESULTS_SUBDIR", ""))
mkpath(results_dir)
framerate = parse(Int, get(ENV, "ANIMATION_FPS", "6"))
px_per_unit = parse(Float64, get(ENV, "FIG_PX_PER_UNIT", "1"))
compression = parse(Int, get(ENV, "ANIMATION_COMPRESSION", "28"))   # H.264 CRF; 20 is Makie's default, higher is smaller

slab(f, dims) = dropdims(Array(interior(f)); dims)

for run in runs
    label = basename(rstrip(run, '/'))
    file = only(filter(f -> endswith(f, "_slices.jld2"), readdir(run; join=true)))
    qxz = FieldTimeSeries(file, "qᶜˡ_xz")
    wxy = FieldTimeSeries(file, "w_xy")
    lwp = FieldTimeSeries(file, "lwp")
    times = qxz.times
    Nt = length(times)
    grid = qxz.grid
    x = collect(xnodes(grid, Center())) ./ 1e3
    y = collect(ynodes(grid, Center())) ./ 1e3
    z = collect(znodes(grid, Center()))
    zf = collect(znodes(grid, Face()))

    q_frames = [1e3 .* slab(qxz[n], 2) for n in 1:Nt]           # g kg⁻¹, (x, z)
    w_frames = [slab(wxy[n], 3) for n in 1:Nt]                   # m s⁻¹, (x, y) at the 900 m face
    l_frames = [1e3 .* slab(lwp[n], 3) for n in 1:Nt]            # g m⁻², (x, y)
    qmax = maximum(maximum, q_frames)
    wlim = maximum(f -> maximum(abs, f), w_frames) / 2
    lmax = maximum(maximum, l_frames)

    n = Observable(1)
    q_obs = @lift q_frames[$n]
    w_obs = @lift w_frames[$n]
    l_obs = @lift l_frames[$n]
    title = @lift @sprintf("%s — t = %.1f h after 06 UTC 18 July 2017", label, times[$n] / 3600)

    fig = Figure(size=(1300, 900), fontsize=14)
    Label(fig[0, :], title, fontsize=18, tellwidth=false)
    top = fig[1, 1] = GridLayout()
    bottom = fig[2, 1] = GridLayout()
    ax1 = Axis(top[1, 1], xlabel="x (km)", ylabel="z (m)", title="cloud liquid qᶜˡ (g kg⁻¹) at y = Ly/2")
    hm1 = heatmap!(ax1, x, z, q_obs; colormap=:Blues, colorrange=(0, qmax))
    ylims!(ax1, 0, 2500)
    Colorbar(top[1, 2], hm1)
    ax2 = Axis(bottom[1, 1], xlabel="x (km)", ylabel="y (km)", title="w at 900 m (m s⁻¹)", aspect=1)
    hm2 = heatmap!(ax2, x, y, w_obs; colormap=:balance, colorrange=(-wlim, wlim))
    Colorbar(bottom[1, 2], hm2)
    ax3 = Axis(bottom[1, 3], xlabel="x (km)", ylabel="y (km)", title="liquid water path (g m⁻²)", aspect=1)
    hm3 = heatmap!(ax3, x, y, l_obs; colormap=:viridis, colorrange=(0, lmax))
    Colorbar(bottom[1, 4], hm3)

    path = joinpath(results_dir, "animation_$label.mp4")
    record(fig, path, 1:Nt; framerate, px_per_unit, compression) do i
        n[] = i
    end
    println("wrote ", path, " (", Nt, " frames)")
end
