# The three microphysics members side by side through the whole six hours.
#   julia --project analysis/animate_microphysics.jl [ANIMATION_FPS=12 ANIMATION_COMPRESSION=30]
# The still version of this figure is analysis/microphysics_snapshot.jl. Colour ranges are fixed
# over the run and shared down each row, so the columns can be read against each other and the
# evolution read within a column.
using CairoMakie, JLD2, Oceananigans, Statistics, Printf

fps = parse(Int, get(ENV, "ANIMATION_FPS", "12"))
compression = parse(Int, get(ENV, "ANIMATION_COMPRESSION", "30"))
runs = ("one-moment" => "covert_public_bin_one_moment_posmom_s60_theta",
        "P3, Nᶜˡ = 75 cm⁻³" => "covert_public_bin_p3_n75_posmom_s60_theta",
        "P3, prognostic aerosol" => "covert_public_bin_p3_aer2_posmom_s60_theta")
QCL_MAX, QR_MAX, LWP_MAX, RAIN_MAX = 1.0, 0.02, 350.0, 0.5   # g/kg, g/kg, g/m², mm/day

series = map(runs) do (label, run)
    d = joinpath("output", run)
    (; label,
       qcl = FieldTimeSeries(joinpath(d, "lasso_ena_slices.jld2"), "qᶜˡ_xz"),
       qr  = FieldTimeSeries(joinpath(d, "lasso_ena_slices.jld2"), "qʳ_xz"),
       lwp = FieldTimeSeries(joinpath(d, "lasso_ena_slices.jld2"), "lwp"),
       rain = FieldTimeSeries(joinpath(d, "lasso_ena_slices.jld2"), "rain"))
end
t = series[1].qcl.times
x = collect(xnodes(series[1].qcl.grid, Center())) ./ 1e3
z = collect(znodes(series[1].qcl.grid, Center()))
y = collect(ynodes(series[1].lwp.grid, Center())) ./ 1e3
@info "microphysics animation" frames = length(t) hours = round(t[end]/3600, digits=2)

fig = Figure(size = (1500, 1200), fontsize = 13)
ttl = Observable("")
Label(fig[0, 1:3], ttl, fontsize = 19, tellwidth = false)
obs = [(qcl = Observable(zeros(length(x), length(z))), qr = Observable(zeros(length(x), length(z))),
        lwp = Observable(zeros(length(x), length(y))), rain = Observable(zeros(length(x), length(y))))
       for _ in series]
hm1 = hm2 = hm3 = hm4 = nothing
for (col, s) in enumerate(series)
    Label(fig[1, col], s.label, fontsize = 15, tellwidth = false, font = :bold)
    axa = Axis(fig[2, col], ylabel = col == 1 ? "z (m)" : "", title = "cloud liquid, x–z section")
    global hm1 = heatmap!(axa, x, z, obs[col].qcl; colormap = :dense, colorrange = (0, QCL_MAX))
    ylims!(axa, 0, 1600)
    axb = Axis(fig[3, col], ylabel = col == 1 ? "z (m)" : "", title = "rain, x–z section")
    global hm2 = heatmap!(axb, x, z, obs[col].qr; colormap = :Purples, colorrange = (0, QR_MAX))
    ylims!(axb, 0, 1600)
    axc = Axis(fig[4, col], ylabel = col == 1 ? "y (km)" : "", title = "liquid water path", aspect = DataAspect())
    global hm3 = heatmap!(axc, x, y, obs[col].lwp; colormap = :dense, colorrange = (0, LWP_MAX))
    axd = Axis(fig[5, col], xlabel = "x (km)", ylabel = col == 1 ? "y (km)" : "",
               title = "surface rain rate", aspect = DataAspect())
    global hm4 = heatmap!(axd, x, y, obs[col].rain; colormap = :Purples, colorrange = (0, RAIN_MAX))
end
Colorbar(fig[2, 4], hm1, label = "qᶜˡ (g kg⁻¹)")
Colorbar(fig[3, 4], hm2, label = "qʳ (g kg⁻¹)")
Colorbar(fig[4, 4], hm3, label = "LWP (g m⁻²)")
Colorbar(fig[5, 4], hm4, label = "rain (mm day⁻¹)")

mkpath("results")
record(fig, "results/animation_microphysics.mp4", 1:length(t); framerate = fps, compression) do n
    for (col, s) in enumerate(series)
        obs[col].qcl[]  = 1e3 .* Array(interior(s.qcl[n]))[:, 1, :]
        obs[col].qr[]   = 1e3 .* Array(interior(s.qr[n]))[:, 1, :]
        obs[col].lwp[]  = 1e3 .* Array(interior(s.lwp[n]))[:, :, 1]
        obs[col].rain[] = 86400 .* Array(interior(s.rain[n]))[:, :, 1]
    end
    ttl[] = @sprintf("The three microphysics members, %02d:%02d UTC 18 July 2017",
                     6 + floor(Int, t[n]/3600), round(Int, (t[n] % 3600) / 60))
end
println("wrote results/animation_microphysics.mp4  ",
        round(filesize("results/animation_microphysics.mp4")/1e6, digits = 2), " MB")
