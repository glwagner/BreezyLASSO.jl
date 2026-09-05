#####
##### Enthalpy transport by sedimenting condensate.
#####
##### On Breeze `main` (through the pinned revision) sedimentation moves condensate mass but
##### not its energy content: `ρs` is advected by the resolved velocities alone, so rain
##### leaving a cell leaves its latent deficit behind (spurious cooling) and rain arriving
##### in a cell — including the surface cell it piles into — warms it by ℒ Δqʳ / cᵖ. In the
##### P3 runs of this case that produced a surface runaway (T > 320 K within minutes of
##### drizzle onset). Breeze PR 959 ("let falling condensate carry its enthalpy") fixes this
##### generically but is open and conflicting with main, so this file provides the same
##### physics as a density-weighted forcing on `ρs`:
#####
#####   ∂ₜ(ρs) += -∂z[ h(T) (Φ(w + wq, q) - Φ(w, q)) ]
#####
##### where Φ(·, q) is exactly the vertical mass flux the tracer tendency applies (the
##### tracer's own bounds-preserving WENO reconstruction and limiter, or the plain flux for
##### other schemes), wq the species' fall velocity, and h the static-energy content per
##### unit of condensate mass fraction at fixed temperature and fixed (anelastic) cell mass,
##### h = ∂s/∂qˣ|_T = (cˣ - cᵖᵈ) T - ℒˣ, because in Breeze's mass-fraction bookkeeping
##### s = [qᵈ cᵖᵈ + qᵛ cᵖᵛ + qˡ cˡ + qⁱ cⁱ] T + gz - ℒˡ qˡ - ℒⁱ qⁱ with qᵈ = 1 - qᵛ - qˡ - qⁱ:
##### condensate that sediments out of a cell is replaced by dry air mass. Each of the two
##### upwind-biased mass fluxes carries the content of its own donor cell (the cell the
##### flux leaves: above the face for downward, below for upward velocity), since the
##### total and resolved velocities at a face can point in different directions. A rain
##### shaft crossing an isothermal saturated column leaves T unchanged (5 mK), and cold rain
##### entering warmer air cools it by the donor-based mixing amount (both tested); the
##### gravitational part gz is left behind as dissipative heating.
#####

using Adapt: Adapt, adapt
using Oceananigans.Advection: BoundsPreservingWENO, _biased_interpolate_zᵃᵃᶠ, upwind_biased_product,
                              LeftBias, RightBias, _advective_tracer_flux_z,
                              adapt_advection_order, materialize_advection

# Oceananigans ≥ main-2026-09 stores the bounds-preserving limiter as a per-cell field
# (`BoundsPreservation`); the 0.111 release recomputes one factor per evaluating cell.
const CELL_LIMITER_OCEANANIGANS = isdefined(Oceananigans.Advection, :BoundsPreservation)

@static if CELL_LIMITER_OCEANANIGANS
    using Oceananigans.Advection: bounds_preserving_limiter
else
    using Oceananigans.Advection: _ω̂₁, _ω̂ₙ, _ε₂
end
using Oceananigans.Operators: ℑzᵃᵃᶠ, Azᶜᶜᶠ, V⁻¹ᶜᶜᶜ
using Oceananigans.Utils: SumOfArrays
using Oceananigans.Fields: ZeroField
using Breeze.AtmosphereModels: AtmosphereModels, condensate_field_names, prognostic_field_names, microphysical_velocities

struct SedimentationEnthalpyForcing{Q, W, S, C, D, P}
    mass_name :: Q        # Val of the specific mass field, e.g. Val(:qʳ)
    velocity_name :: W    # Val of the fall-velocity field, e.g. Val(:wʳ)
    advection :: S        # the tracer's own advection scheme
    thermodynamic_constants :: C
    density :: D          # reference density (materialized)
    phase :: P            # Val(:liquid) or Val(:ice)
end

Adapt.adapt_structure(to, f::SedimentationEnthalpyForcing) =
    SedimentationEnthalpyForcing(f.mass_name, f.velocity_name, adapt(to, f.advection),
                                 adapt(to, f.thermodynamic_constants), adapt(to, f.density), f.phase)

Base.summary(f::SedimentationEnthalpyForcing) =
    string("SedimentationEnthalpyForcing(", f.mass_name, " falling at ", f.velocity_name, ", ", f.phase, ")")
Base.show(io::IO, f::SedimentationEnthalpyForcing) = print(io, summary(f))

# Density-weighted tendency on ρs: must be supplied under the `ρs` key.
AtmosphereModels.is_density_tendency_forcing(::SedimentationEnthalpyForcing) = true

@inline condensate_content(::Val{:liquid}, T, constants) =
    (constants.liquid.heat_capacity - constants.dry_air.heat_capacity) * T - constants.liquid.reference_latent_heat
@inline condensate_content(::Val{:ice}, T, constants) =
    (constants.ice.heat_capacity - constants.dry_air.heat_capacity) * T - constants.ice.reference_latent_heat

# Mass fluxes (area-weighted, ρ-weighted) through the top (+) and bottom (-) faces of cell k,
# exactly as `bounded_tracer_flux_divergence_z` forms them for the bounds-preserving WENO.
@static if CELL_LIMITER_OCEANANIGANS
    # Per-cell limiter: every face reconstruction is rescaled with the factor of the cell it
    # was reconstructed from, so the flux is single-valued at shared faces. The forcing
    # recomputes θ with the model's own `bounds_preserving_limiter` (its materialized copy of
    # the scheme carries a separate, unused limiter field); the neighbour factors are mirrored
    # at the bottom and top as `fill_halo_regions!` does for the model's limiter field.
    @inline rescale(ĉ, θ, cᵢ) = θ * (ĉ - cᵢ) + cᵢ

    @inline function face_mass_fluxes(i, j, k, grid, advection::BoundsPreservingWENO, ρ, w, c)
        Nz = size(grid, 3)
        k₋ = max(k - 1, 1)
        k₊ = min(k + 1, Nz)
        θ₋ = bounds_preserving_limiter(i, j, k₋, grid, advection, c)
        θ₀ = bounds_preserving_limiter(i, j, k,  grid, advection, c)
        θ₊ = bounds_preserving_limiter(i, j, k₊, grid, advection, c)
        @inbounds c₋ = c[i, j, k-1]
        @inbounds c₀ = c[i, j, k]
        @inbounds c₊ = c[i, j, k+1]
        c₊ᴸ = rescale(_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, LeftBias,  c), θ₀, c₀)
        c₊ᴿ = rescale(_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, RightBias, c), θ₊, c₊)
        c₋ᴸ = rescale(_biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, LeftBias,  c), θ₋, c₋)
        c₋ᴿ = rescale(_biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, RightBias, c), θ₀, c₀)
        w⁺ = @inbounds w[i, j, k+1]
        w⁻ = @inbounds w[i, j, k]
        F⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) * Azᶜᶜᶠ(i, j, k+1, grid) * upwind_biased_product(w⁺, c₊ᴸ, c₊ᴿ)
        F⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ) * Azᶜᶜᶠ(i, j, k,   grid) * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)
        return F⁺, F⁻
    end
else
    # Release 0.111: one factor per evaluating cell, applied to that cell's own reconstructions.
    @inline function face_mass_fluxes(i, j, k, grid, advection::BoundsPreservingWENO, ρ, w, c)
        c_min = @inbounds advection.bounds[1]
        c_max = @inbounds advection.bounds[2]
        c₊ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, LeftBias,  c)
        c₊ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, RightBias, c)
        c₋ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, LeftBias,  c)
        c₋ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, RightBias, c)
        FT = eltype(c)
        ω̂₁ = convert(FT, _ω̂₁)
        ω̂ₙ = convert(FT, _ω̂ₙ)
        ε₂ = convert(FT, _ε₂)
        cᵢⱼ = @inbounds c[i, j, k]
        p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
        M = max(p̃, c₊ᴸ, c₋ᴿ)
        m = min(p̃, c₊ᴸ, c₋ᴿ)
        θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
        θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
        θ = min(θ_max, θ_min, one(grid))
        c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
        c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ
        w⁺ = @inbounds w[i, j, k+1]
        w⁻ = @inbounds w[i, j, k]
        F⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) * Azᶜᶜᶠ(i, j, k+1, grid) * upwind_biased_product(w⁺, c₊ᴸ, c₊ᴿ)
        F⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ) * Azᶜᶜᶠ(i, j, k,   grid) * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)
        return F⁺, F⁻
    end
end

# Any other scheme: the plain Oceananigans face flux (area included), ρ-weighted as Breeze does.
@inline function face_mass_fluxes(i, j, k, grid, advection, ρ, w, c)
    F⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) * _advective_tracer_flux_z(i, j, k+1, grid, advection, w, c)
    F⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ) * _advective_tracer_flux_z(i, j, k,   grid, advection, w, c)
    return F⁺, F⁻
end

# Content of the donor cell of a flux through face k: cell k-1 (below) when the velocity at
# the face is upward, cell k (above) when it is downward. Neighbour indices are clamped to
# 1:Nz (as Breeze's sedimentation does) so the bottom outflow and top faces never read
# unfilled halos; the direction choice is preserved.
@inline function donor_content(phase, T, i, j, k, grid, velocity, constants)
    Nz = size(grid, 3)
    k_below = max(k - 1, 1)
    k_above = min(k, Nz)
    @inbounds T_below = T[i, j, k_below]
    @inbounds T_above = T[i, j, k_above]
    T_donor = ifelse(velocity ≥ 0, T_below, T_above)
    return condensate_content(phase, T_donor, constants)
end

@inline function (f::SedimentationEnthalpyForcing)(i, j, k, grid, clock, fields)
    q = field_by_name(fields, f.mass_name)
    wq = field_by_name(fields, f.velocity_name)
    w = fields.w
    T = fields.T
    ρ = f.density
    constants = f.thermodynamic_constants
    wᵗ = SumOfArrays{2}(w, wq)
    F⁺ᵗ, F⁻ᵗ = face_mass_fluxes(i, j, k, grid, f.advection, ρ, wᵗ, q)
    F⁺, F⁻ = face_mass_fluxes(i, j, k, grid, f.advection, ρ, w, q)
    @inbounds begin
        h⁺ᵗ = donor_content(f.phase, T, i, j, k+1, grid, wᵗ[i, j, k+1], constants)
        h⁻ᵗ = donor_content(f.phase, T, i, j, k,   grid, wᵗ[i, j, k],   constants)
        h⁺  = donor_content(f.phase, T, i, j, k+1, grid, w[i, j, k+1],  constants)
        h⁻  = donor_content(f.phase, T, i, j, k,   grid, w[i, j, k],    constants)
    end
    Φ⁺ = h⁺ᵗ * F⁺ᵗ - h⁺ * F⁺
    Φ⁻ = h⁻ᵗ * F⁻ᵗ - h⁻ * F⁻
    return - V⁻¹ᶜᶜᶜ(i, j, k, grid) * (Φ⁺ - Φ⁻)
end

function AtmosphereModels.materialize_atmosphere_model_forcing(f::SedimentationEnthalpyForcing,
                                                               field, name, model_field_names, context::NamedTuple)
    name === :ρs || throw(ArgumentError("SedimentationEnthalpyForcing is a density tendency and must be supplied under `ρs`, got $name"))
    # The model materializes user advection schemes for the grid/architecture (WENO weight
    # computation type, order adaptation); mirror that so the fluxes match bit for bit.
    advection = materialize_advection(adapt_advection_order(f.advection, field.grid), field.grid)
    return SedimentationEnthalpyForcing(f.mass_name, f.velocity_name, advection, f.thermodynamic_constants,
                                        context.total_density, f.phase)
end

specific_name(ρname) = Symbol(string(ρname)[nextind(string(ρname), 1):end])
condensate_phase(ρname) = (occursin("ⁱ", string(ρname)) && !occursin("ʷ", string(ρname))) ? Val(:ice) : Val(:liquid)

const fall_velocity_names = (:wᶜˡ, :wᶜˡₙ, :wʳ, :wʳₙ, :wⁱ, :wⁱₙ, :wˢ, :wᶜⁱ)

# Ask the scheme which fall-velocity field carries `ρname` by handing it a stand-in whose
# "fields" are the field names themselves; schemes without a velocity return `nothing`.
function fall_velocity_name(microphysics, ρname)
    stub = NamedTuple{fall_velocity_names}(fall_velocity_names)
    velocities = try
        microphysical_velocities(microphysics, stub, Val(ρname))
    catch
        nothing
    end
    isnothing(velocities) && return nothing
    return velocities.w isa Symbol ? velocities.w : nothing
end

"""
    sedimentation_enthalpy_forcings(microphysics, scalar_advection; thermodynamic_constants)

One [`SedimentationEnthalpyForcing`](@ref) per sedimenting prognostic condensate mass of
`microphysics` (P3: cloud liquid, rain, ice, liquid on ice; one-moment: rain), each using
the advection scheme the model applies to that density in `scalar_advection`. Supply the
returned tuple under the `ρs` key.
"""
function sedimentation_enthalpy_forcings(microphysics, scalar_advection; thermodynamic_constants)
    forcings = ()
    for ρname in condensate_field_names(microphysics)
        ρname ∈ prognostic_field_names(microphysics) || continue
        wname = fall_velocity_name(microphysics, ρname)
        isnothing(wname) && continue
        scheme = scalar_advection[ρname]
        forcing = SedimentationEnthalpyForcing(Val(specific_name(ρname)), Val(wname), scheme,
                                               thermodynamic_constants, nothing, condensate_phase(ρname))
        forcings = (forcings..., forcing)
    end
    return forcings
end
