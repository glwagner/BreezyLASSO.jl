#####
##### Simplified interactive longwave radiation as a Breeze *radiation model*.
#####
##### Status: this is the radiation of the public Covert et al. (2022) SAM configuration
##### (`doradsimple = .true., dolongwave = .true., doshortwave = .false.`) and of SAM's
##### `rad_simple.f90` (Stevens et al. 2005, DYCOMS-II RF01). It is retained as the
##### low-cost *legacy control* for the same-day Covert benchmark. The LASSO-ENA production
##### protocol uses RRTMG LW+SW, which Breeze provides through `RadiativeTransferModel`
##### (RRTMGP all-sky); see `lasso_ena_simulation(radiation=:rrtmgp)`.
#####
##### Net upward longwave flux in each column:
#####
#####   F(z) = F₀ exp(-Q(z, ∞)) + F₁ exp(-Q(0, z)) [+ cₚ ρ D (¼ z + ¾ zᵢ)(z - zᵢ)^{1/3}, z > zᵢ]
#####
##### with Q(a, b) = κ ∫ₐᵇ ρ (qᶜˡ + qⁱ) dz. The LASSO SAM (`lasso_ena_noice`) sets
##### F₀ = 113 W m⁻², F₁ = 22 W m⁻², κ = 85 m² kg⁻¹ and disables the free-tropospheric
##### term; the DYCOMS-II original is F₀ = 70 W m⁻² with the term enabled (D = 3.75×10⁻⁶ s⁻¹,
##### cₚ = 1015 J kg⁻¹ K⁻¹, zᵢ = top of the highest layer with qᵛ + qᶜˡ + qⁱ > 8 g kg⁻¹).
#####
##### The model tendency receives the flux divergence -∂F/∂z [W m⁻³] through Breeze's
##### radiation interface (`radiation_flux_divergence`), so the heating enters the static
##### energy equation exactly like RRTMGP's. (SAM divides the divergence by cₚ,spec = 1015
##### and adds it to t = T + gz/cp, a 1% different heat capacity; we keep the
##### energy-consistent form.)
#####

using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index
using Oceananigans: Field, Center, Face, CenterField, IterationInterval, fields
using Oceananigans.Fields: ZFaceField, ZeroField
using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Utils: launch!
using Breeze.AtmosphereModels: AtmosphereModels

struct SimpleLongwaveRadiation{FT, F, H, S}
    F₀ :: FT
    F₁ :: FT
    κ :: FT
    D :: FT
    cₚ :: FT
    qᵗ_inversion :: FT
    free_troposphere_term :: Bool
    flux :: F                 # ZFaceField: net upward longwave flux [W m⁻²]
    flux_divergence :: H      # CenterField: -∂F/∂z [W m⁻³], read by the energy tendency
    schedule :: S
end

"""
    SimpleLongwaveRadiation(grid; F₀=113, F₁=22, κ=85, D=3.75e-6, cₚ=1015, qᵗ_inversion=8e-3,
                            free_troposphere_term=false, schedule=IterationInterval(1))

Simple interactive longwave radiation model (see the file header). Defaults follow the
LASSO-ENA SAM `rad_simple.f90`; pass `F₀=70, free_troposphere_term=true` for the DYCOMS-II
form. Pass the result as `radiation` to `AtmosphereModel`.
"""
function SimpleLongwaveRadiation(grid; F₀=113, F₁=22, κ=85, D=3.75e-6, cₚ=1015, qᵗ_inversion=8e-3,
                                 free_troposphere_term=false, schedule=IterationInterval(1))
    FT = eltype(grid)
    flux = ZFaceField(grid)
    flux_divergence = CenterField(grid)
    return SimpleLongwaveRadiation(FT(F₀), FT(F₁), FT(κ), FT(D), FT(cₚ), FT(qᵗ_inversion),
                                   free_troposphere_term, flux, flux_divergence, schedule)
end

Adapt.adapt_structure(to, r::SimpleLongwaveRadiation) =
    SimpleLongwaveRadiation(r.F₀, r.F₁, r.κ, r.D, r.cₚ, r.qᵗ_inversion, r.free_troposphere_term,
                            adapt(to, r.flux), adapt(to, r.flux_divergence), adapt(to, r.schedule))

Base.summary(r::SimpleLongwaveRadiation) =
    string("SimpleLongwaveRadiation(F₀=", r.F₀, ", F₁=", r.F₁, ", κ=", r.κ,
           r.free_troposphere_term ? ", DYCOMS free-troposphere term" : "", ")")
Base.show(io::IO, r::SimpleLongwaveRadiation) = print(io, summary(r))

# Breeze calls this through `update_radiation!` on the schedule (and at iteration 0).
function AtmosphereModels._update_radiation!(r::SimpleLongwaveRadiation, model)
    grid = model.grid
    model_fields = fields(model)
    ρ = model.dynamics.reference_state.density
    qᵛ = model_fields.qᵛ
    qᶜˡ = model_fields.qᶜˡ
    qⁱ = ice_mass_fraction(model_fields)
    launch!(grid.architecture, grid, :xy, _compute_simple_longwave_flux!,
            r.flux, r.flux_divergence, grid, ρ, qᵛ, qᶜˡ, qⁱ,
            r.F₀, r.F₁, r.κ, r.D, r.cₚ, r.qᵗ_inversion, r.free_troposphere_term)
    return nothing
end

# Warm-phase schemes carry no ice field; P3 does (`qⁱ`).
ice_mass_fraction(model_fields) = haskey(model_fields, :qⁱ) ? model_fields.qⁱ : ZeroField()

@inline face_density(i, j, k, grid, ρ, Nz) = ifelse(k > Nz, @inbounds(ρ[i, j, Nz]), ℑzᵃᵃᶠ(i, j, k, grid, ρ))

@inline condensate(i, j, k, qᶜˡ, qⁱ) = @inbounds max(qᶜˡ[i, j, k], 0) + max(qⁱ[i, j, k], 0)

# Net upward longwave flux at face k given the optical depths above (Q∞) and below (Q₀) it.
@inline function simple_longwave_face_flux(i, j, k, grid, ρ, Nz, Q∞, Q₀, zᵢ, itop, F₀, F₁, D, cₚ, free_troposphere_term)
    FT = typeof(Q∞)
    zᶠ = znode(i, j, k, grid, Center(), Center(), Face())
    Fᵏ = F₀ * exp(-Q∞) + F₁ * exp(-Q₀)
    above = (k > itop) & free_troposphere_term
    Δzᵢ = max(zᶠ - zᵢ, zero(FT))
    ρᶠ = face_density(i, j, k, grid, ρ, Nz)
    Fᶠᵗ = cₚ * ρᶠ * D * (FT(0.25) * zᶠ + FT(0.75) * zᵢ) * cbrt(Δzᵢ)
    return Fᵏ + ifelse(above, Fᶠᵗ, zero(FT))
end

@kernel function _compute_simple_longwave_flux!(F, H, grid, ρ, qᵛ, qᶜˡ, qⁱ, F₀, F₁, κ, D, cₚ, qᵗ_inversion, free_troposphere_term)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    FT = eltype(F)

    # Total cloud optical depth of the column and the inversion face index
    Q∞ = zero(FT)
    itop = 1
    for k in 1:Nz
        qᶜ = condensate(i, j, k, qᶜˡ, qⁱ)
        qᵗ = @inbounds qᵛ[i, j, k] + qᶜ
        Q∞ += κ * @inbounds(ρ[i, j, k]) * qᶜ * Δzᶜᶜᶜ(i, j, k, grid)
        itop = ifelse(qᵗ > qᵗ_inversion, k + 1, itop)
    end
    zᵢ = znode(i, j, itop, grid, Center(), Center(), Face())

    # Sweep upward accumulating the optical depth from the surface: the flux at face k
    # uses the optical depth below face k, then the layer k contribution moves from Q∞ to Q₀.
    Q₀ = zero(FT)
    for k in 1:Nz
        @inbounds F[i, j, k] = simple_longwave_face_flux(i, j, k, grid, ρ, Nz, Q∞, Q₀, zᵢ, itop,
                                                         F₀, F₁, D, cₚ, free_troposphere_term)
        ΔQ = κ * @inbounds(ρ[i, j, k]) * condensate(i, j, k, qᶜˡ, qⁱ) * Δzᶜᶜᶜ(i, j, k, grid)
        Q∞ -= ΔQ
        Q₀ += ΔQ
    end
    @inbounds F[i, j, Nz+1] = simple_longwave_face_flux(i, j, Nz+1, grid, ρ, Nz, Q∞, Q₀, zᵢ, itop,
                                                        F₀, F₁, D, cₚ, free_troposphere_term)

    # Heating rate density: -∂F/∂z
    for k in 1:Nz
        @inbounds H[i, j, k] = - (F[i, j, k+1] - F[i, j, k]) / Δzᶜᶜᶜ(i, j, k, grid)
    end
end
