# Animate the saved slices of one or more runs: the cloud-liquid and rain sections at y = Ly/2,
# and w at 900 m, the liquid water path and the surface rain rate as maps.
#   RESULTS_SUBDIR=covert_grid julia --project=analysis analysis/animate_slices.jl output/run_a output/run_b ...
# Writes results/<subdir>/animation_<run>.mp4, one frame per saved slice (ANIMATION_FPS frames per second,
# default 6; 12 for the 1-minute-slice runs). ANIMATION_SIZE (default 1000x780 points), ANIMATION_FONTSIZE (12),
# FIG_PX_PER_UNIT (1) and ANIMATION_COMPRESSION (H.264 CRF, 28) trade legibility for file size.
# The color ranges are fixed across runs so members can be compared frame by frame; override with
# QCL_MAX (g/kg), QR_MAX (g/kg), W_MAX (m/s), LWP_MAX (g/m²) and RAIN_MAX (mm/day).
using CairoMakie, JLD2, Oceananigans, Oceananigans.Units, Printf

runs = ARGS
results_dir = joinpath(@__DIR__, "..", "results", get(ENV, "RESULTS_SUBDIR", ""))
mkpath(results_dir)
framerate = parse(Int, get(ENV, "ANIMATION_FPS", "6"))
px_per_unit = parse(Float64, get(ENV, "FIG_PX_PER_UNIT", "1"))
compression = parse(Int, get(ENV, "ANIMATION_COMPRESSION", "28"))
width, height = parse.(Int, split(get(ENV, "ANIMATION_SIZE", "1000x780"), "x"))
fontsize = parse(Int, get(ENV, "ANIMATION_FONTSIZE", "12"))
qcl_max = parse(Float64, get(ENV, "QCL_MAX", "1.4"))     # g kg⁻¹
qr_max = parse(Float64, get(ENV, "QR_MAX", "0.05"))      # g kg⁻¹
w_max = parse(Float64, get(ENV, "W_MAX", "1.5"))         # m s⁻¹
lwp_max = parse(Float64, get(ENV, "LWP_MAX", "350"))     # g m⁻²
rain_max = parse(Float64, get(ENV, "RAIN_MAX", "2"))     # mm day⁻¹

slab(f, dims) = dropdims(Array(interior(f)); dims)

for run in runs
    label = basename(rstrip(run, '/'))
    file = only(filter(f -> endswith(f, "_slices.jld2"), readdir(run; join=true)))
    qxz = FieldTimeSeries(file, "qᶜˡ_xz")
    rxz = FieldTimeSeries(file, "qʳ_xz")
    wxy = FieldTimeSeries(file, "w_xy")
    lwp = FieldTimeSeries(file, "lwp")
    rain = FieldTimeSeries(file, "rain")
    times = qxz.times
    Nt = length(times)
    grid = qxz.grid
    x = collect(xnodes(grid, Center())) ./ 1e3
    y = collect(ynodes(grid, Center())) ./ 1e3
    z = collect(znodes(grid, Center()))

    q_frames = [1e3 .* slab(qxz[n], 2) for n in 1:Nt]            # g kg⁻¹, (x, z)
    r_frames = [1e3 .* slab(rxz[n], 2) for n in 1:Nt]            # g kg⁻¹, (x, z)
    w_frames = [slab(wxy[n], 3) for n in 1:Nt]                   # m s⁻¹, (x, y) at the 900 m face
    l_frames = [1e3 .* slab(lwp[n], 3) for n in 1:Nt]            # g m⁻², (x, y)
    p_frames = [86400 .* slab(rain[n], 3) for n in 1:Nt]         # mm day⁻¹, (x, y)

    n = Observable(1)
    q_obs = @lift q_frames[$n]
    r_obs = @lift r_frames[$n]
    w_obs = @lift w_frames[$n]
    l_obs = @lift l_frames[$n]
    p_obs = @lift p_frames[$n]
    title = @lift @sprintf("%s — t = %.2f h after 06 UTC 18 July 2017", label, times[$n] / 3600)

    fig = Figure(size=(width, height); fontsize)
    Label(fig[0, 1], title, fontsize=fontsize + 4, tellwidth=false)

    sections = fig[1, 1] = GridLayout()
    ax1 = Axis(sections[1, 1], ylabel="z (m)", title="cloud liquid qᶜˡ (g kg⁻¹) at y = Ly/2",
               xticklabelsvisible=false)
    hm1 = heatmap!(ax1, x, z, q_obs; colormap=:Blues, colorrange=(0, qcl_max))
    Colorbar(sections[1, 2], hm1)
    ax2 = Axis(sections[2, 1], xlabel="x (km)", ylabel="z (m)", title="rain qʳ (g kg⁻¹) at y = Ly/2")
    hm2 = heatmap!(ax2, x, z, r_obs; colormap=:Purples, colorrange=(0, qr_max))
    Colorbar(sections[2, 2], hm2)
    for ax in (ax1, ax2)
        ylims!(ax, 0, 1600)
        xlims!(ax, extrema(x)...)
    end
    rowgap!(sections, 6)

    maps = fig[2, 1] = GridLayout()
    ax3 = Axis(maps[1, 1], xlabel="x (km)", ylabel="y (km)", title="w at 900 m (m s⁻¹)", aspect=DataAspect())
    hm3 = heatmap!(ax3, x, y, w_obs; colormap=:balance, colorrange=(-w_max, w_max))
    Colorbar(maps[1, 2], hm3)
    ax4 = Axis(maps[1, 3], xlabel="x (km)", title="liquid water path (g m⁻²)", aspect=DataAspect(), yticklabelsvisible=false)
    hm4 = heatmap!(ax4, x, y, l_obs; colormap=:viridis, colorrange=(0, lwp_max))
    Colorbar(maps[1, 4], hm4)
    ax5 = Axis(maps[1, 5], xlabel="x (km)", title="surface rain (mm day⁻¹)", aspect=DataAspect(), yticklabelsvisible=false)
    hm5 = heatmap!(ax5, x, y, p_obs; colormap=:Purples, colorrange=(0, rain_max))
    Colorbar(maps[1, 6], hm5)
    colgap!(maps, 8)

    rowsize!(fig.layout, 1, Relative(0.46))
    rowgap!(fig.layout, 8)

    path = joinpath(results_dir, "animation_$label.mp4")
    record(fig, path, 1:Nt; framerate, px_per_unit, compression) do i
        n[] = i
    end
    println("wrote ", path, " (", Nt, " frames)")
end
