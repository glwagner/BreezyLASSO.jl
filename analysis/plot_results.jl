# Plot the lightweight diagnostics of one or more runs.
#   julia --project=analysis analysis/plot_results.jl output/run_a output/run_b ...
# Figures are written to results/<basename>/ (profiles, time series, slices).
using CairoMakie, JLD2, Oceananigans, Oceananigans.Units, Statistics, Printf

runs = isempty(ARGS) ? ["output/covert_public_bin_p3_n75"] : ARGS
results_dir = joinpath(@__DIR__, "..", "results")
mkpath(results_dir)

function load_series(dir)
    file = only(filter(f -> endswith(f, "_timeseries.jld2"), readdir(dir; join=true)))
    lwp = FieldTimeSeries(file, "lwp"); rwp = FieldTimeSeries(file, "rwp")
    cf = FieldTimeSeries(file, "cloud_fraction"); rain = FieldTimeSeries(file, "rain_flux")
    t = lwp.times
    return (; t, lwp = [lwp[n][1, 1, 1] for n in 1:length(t)], rwp = [rwp[n][1, 1, 1] for n in 1:length(t)],
              cf = [cf[n][1, 1, 1] for n in 1:length(t)], rain = [rain[n][1, 1, 1] for n in 1:length(t)])
end

function load_profiles(dir)
    file = only(filter(f -> endswith(f, "_profiles.jld2"), readdir(dir; join=true)))
    names = ["u", "v", "w²", "θ", "T", "qᵛ", "qᶜˡ", "qʳ", "cloud_fraction"]
    return Dict(name => FieldTimeSeries(file, name) for name in names)
end

labels = [basename(rstrip(r, '/')) for r in runs]
colors = Makie.wong_colors()

# --- time series --------------------------------------------------------------------
fig = Figure(size=(1100, 800), fontsize=14)
ax1 = Axis(fig[1, 1], xlabel="hours since 06 UTC 18 July 2017", ylabel="LWP (g m⁻²)", title="Domain-mean cloud liquid water path")
ax2 = Axis(fig[1, 2], xlabel="hours", ylabel="cloud fraction", title="Cloud fraction (LWP > 5 g m⁻²)")
ax3 = Axis(fig[2, 1], xlabel="hours", ylabel="RWP (g m⁻²)", title="Rain water path")
ax4 = Axis(fig[2, 2], xlabel="hours", ylabel="mm day⁻¹", title="Surface rain rate")
for (run, label, color) in zip(runs, labels, colors)
    s = load_series(run)
    h = s.t ./ 3600
    lines!(ax1, h, 1e3 .* s.lwp; label, color)
    lines!(ax2, h, s.cf; label, color)
    lines!(ax3, h, 1e3 .* s.rwp; label, color)
    lines!(ax4, h, 86400 .* s.rain; label, color)
end
axislegend(ax1, position=:rb)
save(joinpath(results_dir, "timeseries.png"), fig)

# --- profiles -----------------------------------------------------------------------
for (run, label) in zip(runs, labels)
    p = load_profiles(run)
    times = p["θ"].times
    Nt = length(times)
    fig = Figure(size=(1300, 750), fontsize=13)
    axes = [Axis(fig[1, 1], xlabel="θˡ (K)", ylabel="z (m)"), Axis(fig[1, 2], xlabel="qᵛ (g kg⁻¹)"),
            Axis(fig[1, 3], xlabel="qᶜˡ (g kg⁻¹)"), Axis(fig[1, 4], xlabel="cloud fraction"),
            Axis(fig[2, 1], xlabel="u, v (m s⁻¹)", ylabel="z (m)"), Axis(fig[2, 2], xlabel="w² (m² s⁻²)"),
            Axis(fig[2, 3], xlabel="qʳ (g kg⁻¹)"), Axis(fig[2, 4], xlabel="T (K)")]
    for n in 1:Nt
        c = Makie.cgrad(:viridis)[(n - 1) / max(Nt - 1, 1)]
        lab = @sprintf("%.0f-%.0f h", n == 1 ? 0 : times[n-1] / 3600, times[n] / 3600)
        z = collect(znodes(p["θ"][n]))
        lines!(axes[1], vec(interior(p["θ"][n])), z; color=c, label=lab)
        lines!(axes[2], 1e3 .* vec(interior(p["qᵛ"][n])), z; color=c)
        lines!(axes[3], 1e3 .* vec(interior(p["qᶜˡ"][n])), z; color=c)
        lines!(axes[4], vec(interior(p["cloud_fraction"][n])), z; color=c)
        lines!(axes[5], vec(interior(p["u"][n])), z; color=c)
        lines!(axes[5], vec(interior(p["v"][n])), z; color=c, linestyle=:dash)
        lines!(axes[6], vec(interior(p["w²"][n])), z; color=c)
        lines!(axes[7], 1e3 .* vec(interior(p["qʳ"][n])), z; color=c)
        lines!(axes[8], vec(interior(p["T"][n])), z; color=c)
    end
    for ax in axes
        ylims!(ax, 0, 3000)
    end
    xlims!(axes[1], 288, 312)
    axislegend(axes[1], position=:rb, labelsize=9)
    Label(fig[0, :], "$label: hourly-mean profiles", fontsize=18, tellwidth=false)
    save(joinpath(results_dir, "profiles_$label.png"), fig)
end

# --- slices at the last output ------------------------------------------------------
for (run, label) in zip(runs, labels)
    file = only(filter(f -> endswith(f, "_slices.jld2"), readdir(run; join=true)))
    qxz = FieldTimeSeries(file, "qᶜˡ_xz"); wxy = FieldTimeSeries(file, "w_xy"); lwp = FieldTimeSeries(file, "lwp")
    n = length(qxz.times)
    fig = Figure(size=(1300, 900), fontsize=13)
    ax1 = Axis(fig[1, 1:2], xlabel="x (m)", ylabel="z (m)", title=@sprintf("qᶜˡ (kg kg⁻¹) at y = Ly/2, t = %.1f h", qxz.times[n] / 3600))
    hm1 = heatmap!(ax1, qxz[n]; colormap=:Blues)
    ylims!(ax1, 0, 2500)
    Colorbar(fig[1, 3], hm1)
    ax2 = Axis(fig[2, 1], xlabel="x (m)", ylabel="y (m)", title="w at 900 m (m s⁻¹)", aspect=1)
    wl = maximum(abs, wxy[n]) / 2
    hm2 = heatmap!(ax2, wxy[n]; colormap=:balance, colorrange=(-wl, wl))
    Colorbar(fig[2, 2], hm2)
    ax3 = Axis(fig[2, 3], xlabel="x (m)", ylabel="y (m)", title="LWP (g m⁻²)", aspect=1)
    hm3 = heatmap!(ax3, 1e3 * lwp[n]; colormap=:viridis)
    Colorbar(fig[2, 4], hm3)
    Label(fig[0, :], label, fontsize=18, tellwidth=false)
    save(joinpath(results_dir, "slices_$label.png"), fig)
end
# --- provenance and summary ---------------------------------------------------------
open(joinpath(results_dir, "summary.md"), "w") do io
    println(io, "| run | label | hours | final LWP (g m⁻²) | mean LWP last hour | final cloud fraction | final rain (mm day⁻¹) |")
    println(io, "|---|---|---|---|---|---|---|")
    for (run, label) in zip(runs, labels)
        s = load_series(run)
        prov = joinpath(run, "provenance.toml")
        isfile(prov) && cp(prov, joinpath(results_dir, "$(label)_provenance.toml"); force=true)
        caselabel = isfile(prov) ? something(match(r"label = \"([^\"]*)\"", read(prov, String)), (; captures=[""])).captures[1] : ""
        last_hour = s.t .≥ s.t[end] - 3600
        @printf(io, "| %s | %s | %.1f | %.1f | %.1f | %.2f | %.3f |\n", label, caselabel, s.t[end] / 3600,
                1e3 * s.lwp[end], 1e3 * mean(s.lwp[last_hour]), s.cf[end], 86400 * s.rain[end])
    end
end
println("figures written to ", abspath(results_dir))
