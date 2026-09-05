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
##### condensate that sediments out of a cell is replaced by dry air mass. With this
##### content a rain shaft crossing an isothermal, saturated column leaves the temperature
##### unchanged (tested); the gravitational part gz is left behind as dissipative heating.
#####

using Adapt: Adapt, adapt
using Oceananigans.Advection: BoundsPreservingWENO, _biased_interpolate_zᵃᵃᶠ, upwind_biased_product,
                              LeftBias, RightBias, _ω̂₁, _ω̂ₙ, _ε₂, _advective_tracer_flux_z,
                              adapt_advection_order, materialize_advection
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

# Any other scheme: the plain Oceananigans face flux (area included), ρ-weighted as Breeze does.
@inline function face_mass_fluxes(i, j, k, grid, advection, ρ, w, c)
    F⁺ = ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) * _advective_tracer_flux_z(i, j, k+1, grid, advection, w, c)
    F⁻ = ℑzᵃᵃᶠ(i, j, k,   grid, ρ) * _advective_tracer_flux_z(i, j, k,   grid, advection, w, c)
    return F⁺, F⁻
end

@inline function (f::SedimentationEnthalpyForcing)(i, j, k, grid, clock, fields)
    q = field_by_name(fields, f.mass_name)
    wq = field_by_name(fields, f.velocity_name)
    w = fields.w
    T = fields.T
    ρ = f.density
    wᵗ = SumOfArrays{2}(w, wq)
    F⁺ᵗ, F⁻ᵗ = face_mass_fluxes(i, j, k, grid, f.advection, ρ, wᵗ, q)
    F⁺, F⁻ = face_mass_fluxes(i, j, k, grid, f.advection, ρ, w, q)
    h⁺ = condensate_content(f.phase, ℑzᵃᵃᶠ(i, j, k+1, grid, T), f.thermodynamic_constants)
    h⁻ = condensate_content(f.phase, ℑzᵃᵃᶠ(i, j, k,   grid, T), f.thermodynamic_constants)
    return - V⁻¹ᶜᶜᶜ(i, j, k, grid) * (h⁺ * (F⁺ᵗ - F⁺) - h⁻ * (F⁻ᵗ - F⁻))
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
