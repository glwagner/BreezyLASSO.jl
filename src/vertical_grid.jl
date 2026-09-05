#####
##### LASSO-ENA vertical grid: 260 levels, Δz = 25 m up to a cell center at 6012.5 m,
##### then a geometrically stretched layer reaching a top cell center of 8087.5 m
##### (modeling_methodology.html; `nz_gl = 260` in the LASSO SAM `domain.f90`).
#####
##### The official `grd` file is inside the restricted `samin` bundle, so the stretching
##### law of the top 19 levels is a reconstruction: a constant growth ratio r chosen so
##### that the last cell center lands exactly at 8087.5 m. Replace `lasso_ena_cell_centers`
##### by the archive's `grd` values once available (see `data/README.md`).
#####

"""
    lasso_ena_cell_centers(; Δz=25, uniform_top=6012.5, top=8087.5, Nz=260)

Cell-center heights of the LASSO-ENA standard vertical grid.
"""
function lasso_ena_cell_centers(; Δz=25.0, uniform_top=6012.5, top=8087.5, Nz=260)
    N_uniform = round(Int, (uniform_top - Δz / 2) / Δz) + 1
    centers = [Δz / 2 + Δz * (k - 1) for k in 1:N_uniform]
    N_stretch = Nz - N_uniform
    N_stretch ≥ 0 || throw(ArgumentError("Nz = $Nz is smaller than the uniform region ($N_uniform levels)"))
    if N_stretch > 0
        target = (top - uniform_top) / Δz # = Σ_{n=1}^{N_stretch} rⁿ
        r = geometric_growth_ratio(target, N_stretch)
        for n in 1:N_stretch
            push!(centers, centers[end] + Δz * r^n)
        end
    end
    return centers
end

# Solve Σ_{n=1}^{N} rⁿ = target for r by bisection.
function geometric_growth_ratio(target, N)
    f(r) = sum(r^n for n in 1:N) - target
    lo, hi = 1.0, 2.0
    f(lo) < 0 || return lo
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
    lasso_ena_vertical_faces(; kwargs...)

Cell-interface heights (length Nz + 1) for [`lasso_ena_cell_centers`](@ref): interfaces
halfway between consecutive centers, with the bottom face at z = 0.
"""
function lasso_ena_vertical_faces(; kwargs...)
    c = lasso_ena_cell_centers(; kwargs...)
    faces = zeros(length(c) + 1)
    faces[1] = 0
    for k in 1:length(c)-1
        faces[k+1] = (c[k] + c[k+1]) / 2
    end
    faces[end] = c[end] + (c[end] - faces[end-1])
    return faces
end

"""
    uniform_vertical_faces(Nz, top)

Uniformly spaced cell interfaces from 0 to `top`, for quick low-resolution tests.
"""
uniform_vertical_faces(Nz, top) = collect(range(0, top, length=Nz + 1))
