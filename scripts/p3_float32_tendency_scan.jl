# Where does the Float32 P3-aer2 first-step NaN originate? Scan the stage-1 tendencies Gⁿ of
# the freshly initialized model for non-finite values, then evaluate the advection and each
# forcing term of the affected scalar at the first bad cell.
#   julia --project scripts/p3_float32_tendency_scan.jl [Float32|Float64]
using BreezyLASSO, Breeze, Oceananigans, Printf
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Forcings: MultipleForcings
using Breeze.AtmosphereModels: AtmosphereModels as AM, div_ρUc

FT = length(ARGS) ≥ 1 && ARGS[1] == "Float64" ? Float64 : Float32
data = joinpath(@__DIR__, "..", "data", "covert2022_bin")
case = lasso_ena_simulation(data; preset=:covert_public_bin, arch=CPU(), FT, Nx=8, Ny=8, Lx=280, Ly=280,
                            z_faces=lasso_ena_vertical_faces(), microphysics=:p3_aer2, stop_time=2.0,
                            Δt=0.5, max_Δt=0.5, write_output=false, aerosol_replenishment=:diagnostic_ccn)
model = case.model
grid = model.grid
println("aerosol modes ($FT): ", model.microphysics.aerosol)
update_state!(model)

zc = Array(znodes(grid, Center()))
G = model.timestepper.Gⁿ
bad_cells = Dict{Symbol, Vector{CartesianIndex{3}}}()
for (name, g) in pairs(G)
    a = Array(interior(g))
    idx = findall(!isfinite, a)
    isempty(idx) && continue
    ks = sort(unique(map(c -> c[3], idx)))
    bad_cells[name] = idx
    println("Gⁿ.$name: $(length(idx)) non-finite; levels $(first(ks))..$(last(ks)) (z = $(zc[first(ks)])..$(zc[last(ks)]) m); first cell $(Tuple(idx[1]))")
end
isempty(bad_cells) && println("all stage-1 tendencies finite")

fields_ = Oceananigans.fields(model)
ρ_field = model.dynamics.reference_state.density
μ = model.microphysical_fields

function evaluate_terms(name, i, j, k)
    specific = AM.specific_field_name(name)
    c = μ[specific]
    println("Terms of Gⁿ.$name at $((i, j, k)), z = $(zc[k]) m, c = $(c[i, j, k]):")
    adv = model.advection[name]
    U = model.velocities
    println("   advection  : ", -div_ρUc(i, j, k, grid, adv, ρ_field, U, c), "  (", summary(adv), ")")
    f = model.forcing[name]
    parts = f isa MultipleForcings ? f.forcings : (f,)
    for (n, part) in enumerate(parts)
        v = part(i, j, k, grid, model.clock, fields_)
        println("   forcing $n  : ", v, "  ", summary(part))
    end
    for kk in max(k - 3, 1):min(k + 3, size(grid, 3))
        @printf("      k=%3d z=%7.1f c=%.6e T=%.3f qᵛ=%.6e w=%.3e\n", kk, zc[kk], c[i, j, kk], model.temperature[i, j, kk], μ.qᵛ[i, j, kk], U.w[i, j, kk])
    end
end

for name in (:ρnᵃ, :ρnᶜˡ, :ρqᶜˡ, :ρs, :ρqᵛ)
    haskey(bad_cells, name) || continue
    i, j, k = Tuple(bad_cells[name][1])
    evaluate_terms(name, i, j, k)
end
println("SCAN DONE")
