# The three microphysics members side by side at one instant, on the Covert grid.
#   julia --project analysis/microphysics_snapshot.jl [SNAPSHOT_HOUR=6]
# Rows: cloud-liquid and rain vertical sections through the domain, liquid water path, and the
# surface rain rate. Columns: one-moment, P3 with a prescribed 75 cm⁻³ droplet number, and P3 with
# prognostic aerosol. The same colour range is used across a row, so the columns are comparable.
using CairoMakie, JLD2, Oceananigans, Statistics, Printf

hour = parse(Float64, get(ENV, "SNAPSHOT_HOUR", "6"))
runs = ("one-moment" => "covert_public_bin_one_moment_posmom_s60_theta",
        "P3, Nᶜˡ = 75 cm⁻³" => "covert_public_bin_p3_n75_posmom_s60_theta",
        "P3, prognostic aerosol" => "covert_public_bin_p3_aer2_posmom_s60_theta")

QCL_MAX, QR_MAX, LWP_MAX, RAIN_MAX = 1.0, 0.02, 350.0, 0.5   # g/kg, g/kg, g/m², mm/day
fig = Figure(size = (1500, 1180), fontsize = 13)
Label(fig[0, 1:3], @sprintf("The three microphysics members at %02d:00 UTC, 18 July 2017", 6 + hour),
      fontsize = 18, tellwidth = false)

hm1 = hm2 = hm3 = hm4 = nothing
for (col, (label, run)) in enumerate(runs)
    d = joinpath("output", run)
    qcl = FieldTimeSeries(joinpath(d, "lasso_ena_slices.jld2"), "qᶜˡ_xz")
    qr  = FieldTimeSeries(joinpath(d, "lasso_ena_slices.jld2"), "qʳ_xz")
    lwp = FieldTimeSeries(joinpath(d, "lasso_ena_slices.jld2"), "lwp")
    rain = FieldTimeSeries(joinpath(d, "lasso_ena_slices.jld2"), "rain")
    n = argmin(abs.(qcl.times ./ 3600 .- hour))
    x = collect(xnodes(qcl.grid, Center())) ./ 1e3
    z = collect(znodes(qcl.grid, Center()))
    y = collect(ynodes(lwp.grid, Center())) ./ 1e3

    Label(fig[1, col], label, fontsize = 15, tellwidth = false, font = :bold)

    axa = Axis(fig[2, col], ylabel = col == 1 ? "z (m)" : "", title = "cloud liquid, x–z section")
    global hm1 = heatmap!(axa, x, z, 1e3 .* Array(interior(qcl[n]))[:, 1, :];
                   colormap = :dense, colorrange = (0, QCL_MAX))
    ylims!(axa, 0, 1600)

    axb = Axis(fig[3, col], ylabel = col == 1 ? "z (m)" : "", title = "rain, x–z section")
    global hm2 = heatmap!(axb, x, z, 1e3 .* Array(interior(qr[n]))[:, 1, :];
                   colormap = :Purples, colorrange = (0, QR_MAX))
    ylims!(axb, 0, 1600)

    axc = Axis(fig[4, col], ylabel = col == 1 ? "y (km)" : "", title = "liquid water path",
               aspect = DataAspect())
    global hm3 = heatmap!(axc, x, y, 1e3 .* Array(interior(lwp[n]))[:, :, 1];
                   colormap = :dense, colorrange = (0, LWP_MAX))

    axd = Axis(fig[5, col], xlabel = "x (km)", ylabel = col == 1 ? "y (km)" : "",
               title = "surface rain rate", aspect = DataAspect())
    global hm4 = heatmap!(axd, x, y, 86400 .* Array(interior(rain[n]))[:, :, 1];
                   colormap = :Purples, colorrange = (0, RAIN_MAX))
end
Colorbar(fig[2, 4], hm1, label = "qᶜˡ (g kg⁻¹)")
Colorbar(fig[3, 4], hm2, label = "qʳ (g kg⁻¹)")
Colorbar(fig[4, 4], hm3, label = "LWP (g m⁻²)")
Colorbar(fig[5, 4], hm4, label = "rain (mm day⁻¹)")

mkpath("results")
save("results/microphysics_snapshot.png", fig; px_per_unit = 2)
println("wrote results/microphysics_snapshot.png")
