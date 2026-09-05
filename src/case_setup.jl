#####
##### The LASSO-ENA 18 July 2017 case driver.
#####
##### Protocol (lasso-ena.svcs.arm.gov modeling_methodology + the lasso_ena_noice SAM):
#####   anelastic, doubly periodic 25.6 km × 25.6 km at 100 m, 260 levels (Δz = 25 m to
#####   6012.5 m, stretched to 8087.5 m), Coriolis at the ENA latitude, time-height
#####   large-scale forcing (tls, qls, wls, uls/vls, ug/vg), domain-mean wind nudging with
#####   τ = 2 h (n0), upper-30% sponge, surface fluxes from SST (flxsst) or prescribed,
#####   RRTMG(P) LW+SW radiation, and microphysics staged as
#####   1M-control → P3-N75 → P3-aer2 (production).
#####

using Dates: Dates, DateTime
using TOML: TOML
using Random: MersenneTwister
using Printf: @sprintf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interior
using Oceananigans.Grids: znodes
using Breeze
using Breeze.Microphysics.PredictedParticleProperties: CloudDroplets, AerosolActivation, AerosolMode
using CloudMicrophysics: CloudMicrophysics
using RRTMGP: RRTMGP
using NCDatasets: NCDatasets
using ClimaComms: ClimaComms

"""
    epoch_from_day_of_year(day0; year=2017)

UTC `DateTime` of the fractional day-of-year `day0` (SAM `day0`; 1.0 = 1 January 00 UTC):
199.25 → 2017-07-18T06:00:00.
"""
function epoch_from_day_of_year(day0; year=2017)
    whole = floor(Int, day0)
    seconds = round(Int, (day0 - whole) * 86400)
    return DateTime(year, 1, 1) + Dates.Day(whole - 1) + Dates.Second(seconds)
end

"""
    lasso_aerosol_modes(FT; setting=:aer2, reference_density, kwargs...)

The two lognormal aerosol modes of the LASSO-ENA spectral-bin configuration converted to
P3 `AerosolMode`s. LASSO quotes number *concentrations* per cm³:

| setting | mode 1 (r = 0.018 μm, σ = 1.53) | mode 2 (r = 0.066 μm, σ = 1.78) |
|---------|---------------------------------|---------------------------------|
| aer1    | 138 cm⁻³                        | 140.5 cm⁻³                      |
| aer2    | 276 cm⁻³                        | 281 cm⁻³                        |
| aer3    | 552 cm⁻³                        | 562 cm⁻³                        |

P3 takes the number *per unit mass* of air, `nᵃ = 10⁶ N / ρ` [kg⁻¹]. The LASSO HUJI-SBM
initializes its CCN with a constant *mixing ratio* with height
(`MICRO_HUJISBM/microphysics.f90`: `FCCN0 = FCCNR_mp * rhocgs(k)/rhocgs(1)`, i.e. a
surface concentration `N` scaled by `ρ(z)/ρ(0)`), so the conversion uses the surface
reference density: `number_mixing_ratio = 1e6 N / ρ(0)`, uniform in z. The chemistry is
set explicitly to the HUJI-SBM values (aerosol density 1790 kg m⁻³, molecular weight
0.115 kg mol⁻¹, van 't Hoff factor 3) rather than P3's implicit defaults, and recorded.
"""
function lasso_aerosol_modes(FT=Float64; setting=:aer2, reference_density,
                             aerosol_density=1790, molecular_weight_aerosol=0.115, vant_hoff_factor=3, kwargs...)
    N₁, N₂ = setting === :aer1 ? (138.0, 140.5) :
             setting === :aer2 ? (276.0, 281.0) :
             setting === :aer3 ? (552.0, 562.0) :
             throw(ArgumentError("unknown LASSO aerosol setting $setting (aer1, aer2, aer3)"))
    n₁ = 1e6 * N₁ / reference_density
    n₂ = 1e6 * N₂ / reference_density
    chemistry = (; aerosol_density, molecular_weight_aerosol, vant_hoff_factor)
    mode1 = AerosolMode(FT; number_mixing_ratio=n₁, mean_radius=0.018e-6, geometric_std=1.53, chemistry..., kwargs...)
    mode2 = AerosolMode(FT; number_mixing_ratio=n₂, mean_radius=0.066e-6, geometric_std=1.78, chemistry..., kwargs...)
    return (mode1, mode2), (; N₁, N₂, n₁, n₂, reference_density, chemistry...)
end

one_moment_extension() = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)

function build_microphysics(FT, scheme; droplet_number, surface_density, aerosol_kwargs=NamedTuple())
    if scheme === :one_moment
        ext = one_moment_extension()
        cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
        return ext.OneMomentCloudMicrophysics(FT; cloud_formation), (; scheme)
    elseif scheme === :p3_n75 || scheme === :p3_prescribed
        cloud = CloudDroplets(FT; number_concentration=droplet_number)
        return P3Microphysics(FT; cloud), (; scheme, droplet_number)
    elseif scheme ∈ (:p3_aer1, :p3_aer2, :p3_aer3)
        setting = Symbol(string(scheme)[4:end])
        modes, conversion = lasso_aerosol_modes(FT; setting, reference_density=surface_density, aerosol_kwargs...)
        aerosol = AerosolActivation(modes...)
        cloud = CloudDroplets(FT; number_concentration=droplet_number) # only the initial droplet number
        return P3Microphysics(FT; cloud, aerosol), (; scheme, setting, conversion...)
    else
        throw(ArgumentError("unknown microphysics scheme $scheme"))
    end
end

is_p3(microphysics) = microphysics isa Breeze.Microphysics.PredictedParticleProperties.PredictedParticlePropertiesMicrophysics

"""
    scalar_advection_schemes(order, microphysics, moisture_name; bounded_condensates=true)

Advection scheme per prognostic scalar, following Breeze's `examples/rico.jl`: the static
energy uses plain WENO; every microphysical *water-mass* tracer (the vapor / equilibrium
moisture and all condensate masses `ρq*`) uses bounds-preserving WENO with bounds `(0, 1)`;
dimensional number and volume moments (`ρn*`, `ρb*`), whose magnitudes are not bounded by
one, use plain WENO (P3's `SpeciesBorrowing` clamps their negative undershoots).

Known issue (documented, tracked): the bounds-preserving WENO of the pinned Oceananigans
limits only the upwind reconstruction of the evaluating cell, so the two cells sharing a
face can apply different fluxes when its limiter fires and mass is not conserved there. With
P3's fast sedimentation this piled phantom rain into the surface cell of the 45-minute
GPU probes (surface `qʳ` reached 9.5 g kg⁻¹, versus < 0.03 g kg⁻¹ with plain WENO), and the
forcing-free rain-shaft budget in `scripts/mass_budget_probe.jl` records the residual.
Oceananigans `main` stores the limiter as a cell field and rescales every face
reconstruction with its own cell's factor, which restores conservation; that path is being
validated in an isolated environment. `bounded_condensates = false` (plain WENO for the
condensate masses) is retained as a diagnostic sensitivity only.
"""
function scalar_advection_schemes(order, microphysics, moisture_name; bounded_condensates=true)
    weno = WENO(; order)
    bounded = WENO(; order, bounds=(0, 1))
    moisture = Symbol("ρ", moisture_name)
    names = (:ρs, moisture, Breeze.AtmosphereModels.prognostic_field_names(microphysics)...)
    schemes = map(names) do name
        s = string(name)
        name === moisture ? bounded :
        (occursin("ρq", s) && bounded_condensates) ? bounded : weno
    end
    return NamedTuple{names}(schemes)
end

"""
    build_case(data_dir; kwargs...)

Core builder behind [`lasso_ena_simulation`](@ref): assemble the simulation from a
directory holding SAM `snd`, `lsf`, `sfc` (and optionally `prm`, `grd`) files with fully
explicit settings. Returns a `NamedTuple` with the `simulation`, `model`, input profiles,
forcing time series, and the configuration record used for provenance.

Keyword arguments:

- `arch = CPU()`, `FT = Float32`
- `Nx = 256, Ny = 256, Lx = Ly = 25600`, `z_faces = lasso_ena_vertical_faces()`
- `day0`: fractional day of year at model time 0 (default: first sounding record)
- `epoch`: UTC date-time at model time 0 (for RRTMGP); default derived from `day0` in 2017
- `moisture_basis = :mixing_ratio`: SAM files carry mixing ratios (per kg dry air); converted
  to Breeze mass fractions for the initial/target profiles and through the exact Jacobian for
  the `qls` source; `:mass_fraction` passes them through
- `latitude = 39.0916`, `longitude = -28.0257` (ENA C1); the namelist `latitude0` wins if present
- `microphysics`: `:one_moment` (1M-control), `:p3_n75` (prescribed droplet number), `:p3_aer2`
  (production; also `:p3_aer1`, `:p3_aer3`)
- `droplet_number = 75e6` [m⁻³]: prescribed Nᶜˡ for `:p3_n75` and the initial in-cloud droplet number
- `radiation`: `:rrtmgp` (all-sky LW+SW, production), `:simple` (LASSO SAM rad_simple, Covert-era
  legacy control), `:dycoms` (Stevens et al. 2005 form), or `nothing`
- `radiation_interval = 60` s, `liquid_effective_radius = 10e-6`, `ice_effective_radius = 30e-6`
- `surface`: `:prescribed_fluxes` (SFC_FLX_FXD, Covert) or `:bulk_sst` (LASSO flxsst)
- `wind_nudging_timescale = 7200` (LASSO n0), `nothing` to disable
- `vertical_advection`: `:full_field` (SAM subsidence.f90), `:mean_profile` (Breeze
  `SubsidenceForcing`), or `nothing`
- `sponge = SAMSponge()`, `closure = :smagorinsky_lilly` (built at the run precision) or any Oceananigans closure / `nothing`, `advection_order = 5`
- `stop_time = 9hours`, `Δt = 1`, `max_Δt = 10`, `cfl = 0.7`
- `perturbation = InitialPerturbation()`
- `output_dir`, `output_prefix`, `profile_interval = 1hour`, `timeseries_interval = 60`,
  `slice_interval = 10minutes`, `slice_height = 900`
- `exclude_subsurface_levels = false` (SAM interpolates through below-surface levels)
- `temperature_neutral_evaporation = true`
"""
function build_case(data_dir;
                              arch = CPU(),
                              FT = Float32,
                              Nx = 256, Ny = 256, Lx = 25600, Ly = 25600,
                              z_faces = lasso_ena_vertical_faces(),
                              day0 = nothing,
                              epoch = nothing,
                              moisture_basis = :mixing_ratio,
                              latitude = 39.0916,
                              longitude = -28.0257,
                              microphysics = :one_moment,
                              droplet_number = 75e6,
                              radiation = :rrtmgp,
                              radiation_interval = 60,
                              liquid_effective_radius = 10e-6,
                              ice_effective_radius = 30e-6,
                              surface_albedo = 0.07,
                              surface_emissivity = 0.98,
                              background_atmosphere = BackgroundAtmosphere(CO₂ = 405e-6, CH₄ = 1.85e-6, N₂O = 330e-9),
                              surface = :prescribed_fluxes,
                              wind_nudging_timescale = 7200,
                              translation_velocity = (0.0, 0.0),
                              vertical_advection = :full_field,
                              geostrophic = true,
                              thermodynamic_tendencies = true,
                              upper_boundary_relaxation = true,
                              sponge = SAMSponge(),
                              closure = :smagorinsky_lilly,
                              advection_order = 5,
                              stop_time = 9hours,
                              Δt = 1.0,
                              max_Δt = 10.0,
                              cfl = 0.7,
                              perturbation = InitialPerturbation(),
                              p3_initialization = :condensate_free,
                              initial_droplet_number = nothing,
                              aerosol_replenishment = nothing,
                              sedimentation_enthalpy = true,
                              bounded_condensate_advection = nothing,
                              label = "unlabeled",
                              output_dir = "output",
                              output_prefix = "lasso_ena",
                              profile_interval = 1hour,
                              timeseries_interval = 60,
                              slice_interval = 10minutes,
                              slice_height = 900,
                              progress_interval = 10minutes,
                              exclude_subsurface_levels = false,
                              temperature_neutral_evaporation = true,
                              write_output = true)

    Oceananigans.defaults.FloatType = FT
    closure = closure === :smagorinsky_lilly ? SmagorinskyLilly(FT) : closure

    #####
    ##### Inputs
    #####

    snd_path = joinpath(data_dir, "snd")
    lsf_path = joinpath(data_dir, "lsf")
    sfc_path = joinpath(data_dir, "sfc")
    prm_path = joinpath(data_dir, "prm")
    soundings = read_sam_sounding(snd_path)
    lsf = read_sam_large_scale_forcing(lsf_path)
    sfc = read_sam_surface_forcing(sfc_path)
    namelist = isfile(prm_path) ? read_sam_namelist(prm_path) : Dict{String, NamelistValue}()
    haskey(namelist, "latitude0") && (latitude = namelist["latitude0"])
    haskey(namelist, "longitude0") && (longitude = namelist["longitude0"])
    day0 = isnothing(day0) ? soundings[1].day : day0
    epoch = isnothing(epoch) ? epoch_from_day_of_year(day0) : epoch

    sounding = soundings[1]
    profiles = SoundingProfiles(sounding; exclude_subsurface_levels, moisture_basis)

    #####
    ##### Grid, reference state, dynamics
    #####

    Nz = length(z_faces) - 1
    grid = RectilinearGrid(arch, FT; size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=z_faces,
                           halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(FT)
    constants64 = ThermodynamicConstants(Float64)

    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = profiles.surface_pressure,
                                     potential_temperature = z -> profiles(:θ, z),
                                     vapor_mass_fraction = z -> profiles(:qᵗ, z))
    dynamics = AnelasticDynamics(reference_state)
    coriolis = FPlane(FT; latitude)

    z_centers = Array(znodes(grid, Center()))
    ρᵣ = Array(interior(reference_state.density, 1, 1, :))
    pᵣ = Array(interior(reference_state.pressure, 1, 1, :))
    surface_density = ρᵣ[1]

    #####
    ##### Microphysics
    #####

    microphysics_model, microphysics_record = build_microphysics(FT, microphysics; droplet_number, surface_density)
    moisture_name = Breeze.AtmosphereModels.moisture_specific_name(microphysics_model)

    momentum_advection = WENO(order=advection_order)
    bounded_condensate_advection = something(bounded_condensate_advection, true)
    scalar_advection = scalar_advection_schemes(advection_order, microphysics_model, moisture_name;
                                                bounded_condensates=bounded_condensate_advection)

    #####
    ##### Large-scale forcing
    #####

    forcing_profiles = LargeScaleForcingProfiles(grid, lsf, z_centers, pᵣ; day0)

    # SAM solves for winds relative to the translating frame (namelist ug, vg): it subtracts
    # the pair from the initial winds, the nudging targets ul0/vl0 and the geostrophic
    # profiles ug0/vg0 (forcing.f90, setdata.f90) and adds it back for the surface wind
    # (surface.f90). Vertical advection, the sponge on u - ū and Coriolis on the relative
    # wind are frame-invariant. Nonzero translation is supported for prescribed fluxes only.
    uᶠ, vᶠ = FT.(translation_velocity)
    translating = !(uᶠ == 0 && vᶠ == 0)
    translating && surface === :bulk_sst &&
        throw(ArgumentError("a translating frame needs the surface wind ū + (ug, vg); Breeze's bulk drag uses the model wind, so use surface = :prescribed_fluxes or translation_velocity = (0, 0)"))
    shifted(fts, shift) = shift == 0 ? fts : shifted_profile_time_series(fts, shift)
    ug_frame = shifted(forcing_profiles.ug, uᶠ)
    vg_frame = shifted(forcing_profiles.vg, vᶠ)
    uls_frame = shifted(forcing_profiles.uls, uᶠ)
    vls_frame = shifted(forcing_profiles.vls, vᶠ)

    geostrophic_forcing = geostrophic ? time_varying_geostrophic_forcings(ug_frame, vg_frame) : (; u=nothing, v=nothing)
    thermodynamic = thermodynamic_tendencies ?
        large_scale_thermodynamic_forcings(forcing_profiles.tls, forcing_profiles.qls;
                                           microphysics=microphysics_model,
                                           thermodynamic_constants=constants,
                                           moisture_name, moisture_basis) :
        NamedTuple{(:s, moisture_name)}((nothing, nothing))

    vadv = vertical_advection === :full_field ? LargeScaleVerticalAdvection(forcing_profiles.wls) :
           vertical_advection === :mean_profile ? SubsidenceForcing(mean_profile_subsidence_velocity(grid, forcing_profiles)) :
           nothing

    nudging_u = isnothing(wind_nudging_timescale) ? nothing : MeanProfileNudging(uls_frame; timescale=wind_nudging_timescale)
    nudging_v = isnothing(wind_nudging_timescale) ? nothing : MeanProfileNudging(vls_frame; timescale=wind_nudging_timescale)

    upper = if upper_boundary_relaxation
        targets = SoundingTargetProfiles(grid, soundings, z_centers, pᵣ; day0, moisture_basis)
        upper_boundary_relaxation_forcings(targets.T, targets.q; microphysics=microphysics_model,
                                           thermodynamic_constants=constants, moisture_name)
    else
        NamedTuple{(:s, moisture_name)}((nothing, nothing))
    end

    compact(args...) = Tuple(a for a in args if !isnothing(a))

    forcing = Dict{Symbol, Tuple}()
    forcing[:u] = compact(geostrophic_forcing.u, nudging_u, vadv, sponge)
    forcing[:v] = compact(geostrophic_forcing.v, nudging_v, vadv, sponge)
    forcing[:w] = compact(sponge)
    forcing[:s] = compact(thermodynamic.s, vadv, upper.s)
    forcing[moisture_name] = compact(thermodynamic[moisture_name], vadv, upper[moisture_name])
    if vertical_advection === :full_field
        for ρname in Breeze.AtmosphereModels.prognostic_field_names(microphysics_model)
            name = Symbol(string(ρname)[nextind(string(ρname), 1):end])
            forcing[name] = (vadv,)
        end
    end
    prognostic_aerosol = is_p3(microphysics_model) && !isnothing(microphysics_model.aerosol)
    if aerosol_replenishment isa Number
        prognostic_aerosol || throw(ArgumentError("aerosol_replenishment needs P3 with AerosolActivation"))
        n_initial = sum(mode.number_mixing_ratio for mode in microphysics_model.aerosol.modes)
        replenishment = AerosolReplenishment(n_initial; timescale=aerosol_replenishment)
        forcing[:nᵃ] = (get(forcing, :nᵃ, ())..., replenishment)
    elseif aerosol_replenishment === :diagnostic_ccn
        prognostic_aerosol || throw(ArgumentError("aerosol_replenishment=:diagnostic_ccn needs P3 with AerosolActivation"))
    elseif !isnothing(aerosol_replenishment)
        throw(ArgumentError("aerosol_replenishment must be nothing, :diagnostic_ccn, or a relaxation timescale"))
    end
    if sedimentation_enthalpy
        # Stand-in for Breeze PR 959: sedimenting condensate carries its static-energy content
        forcing[:ρs] = sedimentation_enthalpy_forcings(microphysics_model, scalar_advection; thermodynamic_constants=constants)
    end
    forcing = NamedTuple(name => value for (name, value) in forcing if !isempty(value))

    #####
    ##### Surface
    #####

    surface_series = surface_time_series(grid, sfc, day0)
    Tₛ = Field{Center, Center, Nothing}(grid)
    set!(Tₛ, FT(sfc.sst[1]))
    sst_updater = SeaSurfaceTemperatureUpdater(Tₛ, surface_series.times, FT.(sfc.sst))

    stress_record = nothing
    boundary_conditions = if isnothing(surface)
        NamedTuple()
    elseif surface === :prescribed_fluxes
        bcs, stress_record = prescribed_surface_flux_boundary_conditions(grid, sfc, day0; thermodynamic_constants=constants,
                                                                         surface_density, moisture_name, temperature_neutral_evaporation,
                                                                         frame_velocity=(uᶠ, vᶠ))
        bcs
    elseif surface === :bulk_sst
        bulk_surface_flux_boundary_conditions(grid, Tₛ; moisture_name)
    else
        throw(ArgumentError("unknown surface mode $surface"))
    end

    #####
    ##### Radiation
    #####

    radiation_model = if radiation === :rrtmgp
        RadiativeTransferModel(grid, AllSkyOptics(), constants;
                               surface_temperature = Tₛ,
                               surface_albedo, surface_emissivity, background_atmosphere,
                               solar_position = ApparentSolarPosition(coordinate=(longitude, latitude), epoch),
                               schedule = TimeInterval(radiation_interval),
                               liquid_effective_radius = ConstantRadiusParticles(liquid_effective_radius),
                               ice_effective_radius = ConstantRadiusParticles(ice_effective_radius))
    elseif radiation === :simple
        SimpleLongwaveRadiation(grid; schedule=IterationInterval(1))
    elseif radiation === :dycoms
        SimpleLongwaveRadiation(grid; F₀=70, free_troposphere_term=true, schedule=IterationInterval(1))
    elseif isnothing(radiation)
        nothing
    else
        throw(ArgumentError("unknown radiation option $radiation"))
    end

    #####
    ##### Model
    #####

    model = AtmosphereModel(grid; formulation = :StaticEnergy, dynamics, coriolis, closure,
                            microphysics = microphysics_model, radiation = radiation_model,
                            momentum_advection, scalar_advection, forcing, boundary_conditions,
                            thermodynamic_constants = constants)

    #####
    ##### Initial condition
    #####

    columns = initial_state_columns(profiles, z_centers, pᵣ; constants=constants64)
    ϵ = perturbation_array(Nx, Ny, z_centers, perturbation)
    δT = perturbation.amplitude_T
    δq = perturbation.amplitude_q
    column(values) = reshape(values, 1, 1, Nz)
    # setperturb.f90 adds ±δq to SAM's dry-basis vapor; with moisture_basis = :mixing_ratio the
    # perturbation is applied to r = q/qᵈ (qᵈ = 1 - q - qᶜ, with qᶜ the condensate held fixed)
    # and converted back, otherwise to q directly.
    perturbed_moisture(q, ϵ, qᶜ=0.0) = moisture_basis === :mixing_ratio ?
        (r = q / (1 - q - qᶜ) + δq * ϵ; max(0, r * (1 - qᶜ) / (1 + r))) : max(0, q + δq * ϵ)

    u₀ = repeat(column(columns.u .- uᶠ), Nx, Ny, 1)
    v₀ = repeat(column(columns.v .- vᶠ), Nx, Ny, 1)

    if is_p3(microphysics_model)
        if p3_initialization === :condensate_free
            # SAM HUJI-SBM `micro_init`: all condensate bins empty, qᵗ all vapor, cloud forms
            # through the scheme's own activation/condensation in the first steps.
            T₀ = column(columns.T_condensate_free) .+ δT .* ϵ
            qᵛ₀ = perturbed_moisture.(column(columns.qᵗ), ϵ)
            set!(model; T=T₀, qᵛ=qᵛ₀, u=u₀, v=v₀)
        elseif p3_initialization === :equilibrium
            # Deliberate P3 choice: warm-phase equilibrium partition (identical to the 1M
            # control's first saturation adjustment), with an in-cloud droplet number.
            T₀ = column(columns.T) .+ δT .* ϵ
            qᵛ₀ = perturbed_moisture.(column(columns.qᵛ), ϵ, column(columns.qᶜˡ))
            qᶜˡ₀ = repeat(column(columns.qᶜˡ), Nx, Ny, 1)
            if !isnothing(microphysics_model.aerosol)
                isnothing(initial_droplet_number) &&
                    throw(ArgumentError("p3_initialization=:equilibrium with aerosol activation needs `initial_droplet_number` [m⁻³]"))
                nᶜˡ₀ = repeat(column(initial_droplet_number ./ ρᵣ .* (columns.qᶜˡ .> 0)), Nx, Ny, 1)
                set!(model; T=T₀, qᵛ=qᵛ₀, qᶜˡ=qᶜˡ₀, nᶜˡ=nᶜˡ₀, u=u₀, v=v₀)
            else
                set!(model; T=T₀, qᵛ=qᵛ₀, qᶜˡ=qᶜˡ₀, u=u₀, v=v₀)
            end
        else
            throw(ArgumentError("unknown p3_initialization $p3_initialization (:condensate_free or :equilibrium)"))
        end
    else
        Π = columns.T ./ columns.θˡ
        θ₀ = column(columns.θˡ) .+ δT .* ϵ ./ column(Π)
        qᵗ₀ = perturbed_moisture.(column(columns.qᵗ), ϵ)
        set!(model; θ=θ₀, qᵗ=qᵗ₀, u=u₀, v=v₀)
    end

    #####
    ##### Simulation
    #####

    simulation = Simulation(model; Δt, stop_time)
    conjure_time_step_wizard!(simulation; cfl, max_Δt)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
    add_callback!(simulation, sst_updater, IterationInterval(1))
    if aerosol_replenishment === :diagnostic_ccn
        n_initial = sum(mode.number_mixing_ratio for mode in microphysics_model.aerosol.modes)
        add_callback!(simulation, DiagnosticCCNProjection(model, n_initial), IterationInterval(1))
    end
    if !isnothing(stress_record)
        stress_updater = prescribed_stress_updater(stress_record, model.velocities)
        stress_updater(simulation) # initialize the stress from the initial wind
        add_callback!(simulation, stress_updater, IterationInterval(1))
    end
    add_callback!(simulation, ProgressMessenger(model), TimeInterval(progress_interval))

    if write_output
        mkpath(output_dir)
        add_output_writers!(simulation; output_dir, output_prefix, profile_interval,
                            timeseries_interval, slice_interval, slice_height)
    end

    config = (; label, arch=string(typeof(arch)), FT=string(FT), Nx, Ny, Nz, Lx, Ly, day0, epoch=string(epoch),
                moisture_basis=string(moisture_basis),
                latitude, longitude, microphysics=string(microphysics), droplet_number,
                radiation=string(radiation), radiation_interval, liquid_effective_radius, ice_effective_radius,
                surface=string(surface), wind_nudging_timescale=something(wind_nudging_timescale, 0),
                translation_velocity_u=uᶠ, translation_velocity_v=vᶠ, translation_frame_applied=translating,
                sam_translation_u=Float64(get(namelist, "ug", 0.0)), sam_translation_v=Float64(get(namelist, "vg", 0.0)),
                geostrophic, thermodynamic_tendencies,
                surface_albedo, surface_emissivity,
                background_CO₂=background_atmosphere.CO₂, background_CH₄=background_atmosphere.CH₄,
                background_N₂O=background_atmosphere.N₂O, background_O₃=string(background_atmosphere.O₃),
                write_output, output_dir=abspath(output_dir), output_prefix, profile_interval, timeseries_interval,
                slice_interval, slice_height, progress_interval,
                vertical_advection=string(vertical_advection), upper_boundary_relaxation,
                sponge=isnothing(sponge) ? "nothing" : summary(sponge), closure=isnothing(closure) ? "nothing" : summary(closure),
                advection_order, stop_time, Δt_initial=Δt, cfl, max_Δt,
                time_stepping = max_Δt == Δt ? "fixed Δt = $Δt s" : "adaptive (initial Δt = $Δt s, CFL wizard cfl = $cfl, max_Δt = $max_Δt s); SAM used fixed dt from the namelist",
                exclude_subsurface_levels, temperature_neutral_evaporation,
                perturbation=string(perturbation), p3_initialization=string(p3_initialization),
                initial_droplet_number=something(initial_droplet_number, 0),
                aerosol_replenishment=string(aerosol_replenishment), sedimentation_enthalpy, bounded_condensate_advection,
                namelist_latitude=get(namelist, "latitude0", NaN),
                microphysics_record...)

    grd_path = joinpath(data_dir, "grd")
    inputs = (; snd=snd_path, lsf=lsf_path, sfc=sfc_path, prm=isfile(prm_path) ? prm_path : nothing,
                grd=isfile(grd_path) ? grd_path : nothing)

    return (; simulation, model, grid, config, inputs, namelist, soundings, lsf, sfc, profiles,
              forcing_profiles, surface_series, columns, surface_temperature=Tₛ)
end

# Mean-profile alternative (Breeze SubsidenceForcing): snapshot of wls at the first record,
# on cell faces. Kept only as a documented sensitivity; it is not time-dependent.
function mean_profile_subsidence_velocity(grid, forcing_profiles)
    wˢ = Field{Nothing, Nothing, Face}(grid)
    z = Array(znodes(grid, Center()))
    w = Array(interior(forcing_profiles.wls[1], 1, 1, :))
    set!(wˢ, ζ -> interpolate_profile(z, w, ζ))
    return wˢ
end

#####
##### Output
#####

function add_output_writers!(simulation; output_dir, output_prefix, profile_interval,
                             timeseries_interval, slice_interval, slice_height)
    model = simulation.model
    grid = model.grid
    u, v, w = model.velocities
    μ = model.microphysical_fields
    qᶜˡ = cloud_liquid(model)
    qʳ = rain_mass_fraction(model)
    qᵛ = μ.qᵛ
    θ = liquid_ice_potential_temperature(model)
    T = model.temperature
    s = model.formulation.specific_energy

    profile_fields = (; u, v, w² = w^2, uw = u * w, vw = v * w, θ, T, s, qᵛ, qᶜˡ, qʳ,
                        cloud_fraction = cloud_fraction_profile(model))
    haskey(μ, :nᶜˡ) && (profile_fields = merge(profile_fields, (; nᶜˡ = μ.nᶜˡ)))
    haskey(μ, :nᵃ) && (profile_fields = merge(profile_fields, (; nᵃ = μ.nᵃ)))
    haskey(μ, :qⁱ) && (profile_fields = merge(profile_fields, (; qⁱ = μ.qⁱ)))
    if !isnothing(model.radiation)
        profile_fields = merge(profile_fields, (; radiative_flux_divergence = model.radiation.flux_divergence))
    end
    # Fields that are already horizontal means (cloud fraction) are written as they are:
    # Oceananigans main refuses to average over dimensions a field no longer has.
    profiles = NamedTuple(name => (horizontally_reduced(f) ? f : Average(f, dims=(1, 2)))
                          for (name, f) in pairs(profile_fields))

    simulation.output_writers[:profiles] =
        JLD2Writer(model, profiles; filename = joinpath(output_dir, output_prefix * "_profiles.jld2"),
                   schedule = AveragedTimeInterval(profile_interval), overwrite_existing = true)

    lwp = liquid_water_path(model; species=:cloud)
    rwp = liquid_water_path(model; species=:rain)
    rain = surface_rain_flux(model)
    timeseries = (; lwp = Average(lwp, dims=(1, 2)),
                    rwp = Average(rwp, dims=(1, 2)),
                    cloud_fraction = cloud_fraction(model),
                    rain_flux = Average(rain, dims=(1, 2)))

    simulation.output_writers[:timeseries] =
        JLD2Writer(model, timeseries; filename = joinpath(output_dir, output_prefix * "_timeseries.jld2"),
                   schedule = TimeInterval(timeseries_interval), overwrite_existing = true)

    z = Array(znodes(grid, Center()))
    k = searchsortedfirst(z, slice_height)
    j = max(1, size(grid, 2) ÷ 2)
    slices = (; qᶜˡ_xz = view(qᶜˡ, :, j, :), qʳ_xz = view(qʳ, :, j, :), w_xz = view(w, :, j, :),
                qᶜˡ_xy = view(qᶜˡ, :, :, k), w_xy = view(w, :, :, k), lwp, rain)

    simulation.output_writers[:slices] =
        JLD2Writer(model, slices; filename = joinpath(output_dir, output_prefix * "_slices.jld2"),
                   schedule = TimeInterval(slice_interval), overwrite_existing = true)
    return nothing
end

horizontally_reduced(f) = (loc = Oceananigans.Fields.location(f); loc[1] === Nothing && loc[2] === Nothing)

#####
##### Provenance
#####

"""
    write_provenance(path, case; extra=NamedTuple())

Write a TOML provenance record: input file paths and SHA-256 checksums, Breeze (pinned
revision and checkout state), Oceananigans and Julia versions, the Manifest checksum, the
LASSO SAM reference commit, and the configuration record (values normalized to TOML
scalars). Round-trips through `TOML.parsefile`.
"""
function write_provenance(path, case; extra=NamedTuple())
    inputs = Dict{String, Any}()
    for (name, file) in pairs(case.inputs)
        isnothing(file) && continue
        inputs[string(name)] = abspath(file)
        inputs[string(name, "_sha256")] = file_sha256(file)
    end
    manifest = joinpath(dirname(dirname(pathof(BreezyLASSO))), "Manifest.toml")
    software = Dict{String, Any}(
        "Breeze" => string(Base.pkgversion(Breeze)),
        "Breeze_source" => breeze_source_description(),
        "Oceananigans" => string(Base.pkgversion(Oceananigans)),
        "Oceananigans_source" => oceananigans_source_description(),
        "BreezyLASSO_source" => package_source_description(BreezyLASSO),
        "julia" => string(VERSION),
        "lasso_sam_reference" => "https://code.arm.gov/lasso/lasso-ena-codes/lasso_sam_sbm.git branch lasso_ena_noice commit 12d02446a2147388dc89d828e6e0553106abea0f")
    isfile(manifest) && (software["BreezyLASSO_manifest_sha256"] = file_sha256(manifest))
    record = Dict{String, Any}(
        "generated" => string(Dates.now()),
        "preset" => string(get(case, :preset, "none")),
        "inputs" => inputs,
        "software" => software,
        "config" => Dict{String, Any}(string(k) => toml_value(v) for (k, v) in pairs(case.config)),
        "extra" => Dict{String, Any}(string(k) => toml_value(v) for (k, v) in pairs(extra)))
    open(path, "w") do io
        TOML.print(io, record)
    end
    return path
end

toml_value(x::Union{Bool, Integer, AbstractFloat, AbstractString}) = x
toml_value(x::Symbol) = string(x)
toml_value(::Nothing) = "nothing"
toml_value(x::Tuple) = [toml_value(v) for v in x]
toml_value(x::AbstractVector) = [toml_value(v) for v in x]
toml_value(x::NamedTuple) = Dict{String, Any}(string(k) => toml_value(v) for (k, v) in pairs(x))
toml_value(x) = string(x)

# A pinned dependency's git revision (from the active Manifest) and the dirty state of a
# `dev`ed checkout, if that is how the package is being loaded.
breeze_source_description() = package_source_description(Breeze)
oceananigans_source_description() = package_source_description(Oceananigans)

function package_source_description(package::Module)
    dir = Base.pkgdir(package)
    name = string(nameof(package))
    # BreezyLASSO's own Manifest carries the `[sources]` pins; the active project's Manifest is
    # the fallback (inside `Pkg.test` the active project is a sandbox).
    manifests = (joinpath(dirname(dirname(pathof(BreezyLASSO))), "Manifest.toml"),
                 Base.active_project() === nothing ? "" : joinpath(dirname(Base.active_project()), "Manifest.toml"))
    rev = "unknown"
    block_pattern = Regex("(?s)\\[\\[deps\\.$name\\]\\]\\n(.*?)(?=\\n\\[\\[|\\z)")
    for manifest in manifests
        isfile(manifest) || continue
        block = match(block_pattern, read(manifest, String))
        isnothing(block) && continue
        repo_rev = match(r"repo-rev = \"([^\"]+)\"", block.captures[1])
        repo_url = match(r"repo-url = \"([^\"]+)\"", block.captures[1])
        (isnothing(repo_rev) || isnothing(repo_url)) && continue
        rev = string(repo_url.captures[1], "@", repo_rev.captures[1])
        break
    end
    dirty = ""
    if isdir(joinpath(dir, ".git"))
        status = try
            read(`git -C $dir status --porcelain`, String)
        catch
            ""
        end
        head = try
            strip(read(`git -C $dir rev-parse HEAD`, String))
        catch
            "unknown"
        end
        dirty = string(" (checkout ", head, isempty(strip(status)) ? ", clean)" : ", DIRTY)")
    end
    return string(rev, " at ", dir, dirty)
end

#####
##### Presets
#####

"""
    read_sam_grd(path)

Read a SAM `grd` file (one scalar-level height per line, optionally more lines than the
model uses) and return the cell-center heights.
"""
function read_sam_grd(path)
    values = Float64[]
    for line in eachline(path)
        t = split(line)
        isempty(t) && continue
        v = tryparse(Float64, replace(t[1], r"[dD]" => "e"))
        isnothing(v) || push!(values, v)
    end
    isempty(values) && error("no heights found in $path")
    return values
end

"""
    faces_from_centers(centers)

Cell interfaces halfway between consecutive centers (bottom face at 0), for a SAM `grd`.
"""
function faces_from_centers(centers)
    faces = zeros(length(centers) + 1)
    for k in 1:length(centers)-1
        faces[k+1] = (centers[k] + centers[k+1]) / 2
    end
    faces[end] = centers[end] + (centers[end] - faces[end-1])
    return faces
end

"""
    covert_public_bin_vertical_faces(; Nz=192, top=20000, Δz=10, uniform_top=1500)

A labelled **reconstruction** of the 192-level, 20-km-top vertical grid of the Covert et al.
(2022) SAM runs (the public repository advertises but does not ship its `grd`/`domain.f90`):
uniform `Δz = 10 m` from the surface to `uniform_top` (150 cells, covering the surface layer
and the inversion), then the remaining 42 cells stretched with a constant growth ratio so
that the top face lands exactly at `top`. Exactly `Nz + 1` faces are returned; pass the
archive's `grd` (via `faces_from_centers(read_sam_grd(path))`) once it is available.
"""
function covert_public_bin_vertical_faces(; Nz=192, top=20000.0, Δz=10.0, uniform_top=1500.0)
    N_uniform = round(Int, uniform_top / Δz)
    N_stretch = Nz - N_uniform
    N_stretch ≥ 1 || throw(ArgumentError("Nz = $Nz leaves no stretched cells above $uniform_top m"))
    faces = collect(0.0:Δz:uniform_top)
    r = geometric_growth_ratio((top - uniform_top) / Δz, N_stretch)
    for n in 1:N_stretch
        push!(faces, faces[end] + Δz * r^n)
    end
    faces[end] = top
    length(faces) == Nz + 1 || error("vertical grid construction produced $(length(faces) - 1) cells, expected $Nz")
    return faces
end

function require_namelist!(namelist, required, preset)
    missing_keys = [k for k in required if !haskey(namelist, k)]
    isempty(missing_keys) || throw(ArgumentError("preset $preset requires namelist parameters $(missing_keys) (found $(sort(collect(keys(namelist)))))"))
    return nothing
end

function require_namelist_value!(namelist, key, value, preset)
    namelist[key] == value || throw(ArgumentError("preset $preset requires $key = $value in the namelist, found $(namelist[key])"))
    return nothing
end

parse_caseid(caseid) = parse.(Int, split(lowercase(String(caseid)), "x"))

"""
    lasso_ena_simulation(data_dir; preset, kwargs...)

Build the simulation for one of the validated presets, applying its namelist-derived
defaults and refusing incompatible switches. Any explicit keyword overrides the preset
(with a warning), so sensitivity runs stay traceable in the provenance record.

- `:covert_public_bin` — the runnable *development benchmark* from the public files of
  the Covert et al. (2022) bin-paper repository: `caseid = 256x256x192` at `dx = dy = 35 m`
  (8.96 km), `day0 = 199.25` (18 July 2017 06 UTC), `nstop × dt = 6 h`, prescribed
  H/LE/τ (`SFC_FLX_FXD`, `SFC_TAU_FXD`), SAM `rad_simple` longwave only, no wind nudging,
  `doupperbound`, `dodamping`. Not the 864²×192, 30.24-km, 06-15 UTC configuration of the
  published paper, and **not** an official LASSO-ENA reproduction.
- `:lasso_ena_official` — the official protocol from a `samin` bundle directory
  (`snd`, `lsf`, `sfc`, `prm`, `grd`): grid from `grd` and the namelist, bulk fluxes from
  SST, RRTMGP LW+SW, `tauls` wind nudging (n0), P3-aer2 with the SBM `diagCCN`
  aerosol-reservoir projection, duration `nstop × dt`.
"""
function lasso_ena_simulation(data_dir; preset = :covert_public_bin, kwargs...)
    prm_path = joinpath(data_dir, "prm")
    isfile(prm_path) || throw(ArgumentError("preset $preset needs the namelist $(prm_path)"))
    namelist = read_sam_namelist(prm_path)
    overrides = Dict{Symbol, Any}(kwargs)

    if preset === :covert_public_bin
        require_namelist!(namelist, ["caseid", "dx", "dy", "dt", "nstop", "day0", "latitude0", "longitude0",
                                     "sfc_flx_fxd", "sfc_tau_fxd", "doradsimple", "dolongwave", "doshortwave",
                                     "donudging_uv", "dolargescale", "dosfcforcing", "doupperbound", "dodamping"], preset)
        require_namelist_value!(namelist, "sfc_flx_fxd", true, preset)
        require_namelist_value!(namelist, "sfc_tau_fxd", true, preset)
        require_namelist_value!(namelist, "doradsimple", true, preset)
        require_namelist_value!(namelist, "dolongwave", true, preset)
        require_namelist_value!(namelist, "doshortwave", false, preset)
        require_namelist_value!(namelist, "donudging_uv", false, preset)
        require_namelist_value!(namelist, "dolargescale", true, preset)
        require_namelist_value!(namelist, "dosfcforcing", true, preset)
        nx, ny, nz = parse_caseid(namelist["caseid"])
        defaults = (; label = "Covert-public-bin development benchmark (not an official LASSO-ENA reproduction)",
                      Nx = nx, Ny = ny, Lx = nx * namelist["dx"], Ly = ny * namelist["dy"],
                      z_faces = covert_public_bin_vertical_faces(; Nz=nz),
                      day0 = Float64(namelist["day0"]),
                      latitude = Float64(namelist["latitude0"]), longitude = Float64(namelist["longitude0"]),
                      stop_time = Float64(namelist["nstop"] * namelist["dt"]),
                      Δt = Float64(namelist["dt"]),
                      max_Δt = Float64(namelist["dt"]),
                      translation_velocity = (Float64(get(namelist, "ug", 0.0)), Float64(get(namelist, "vg", 0.0))),
                      microphysics = :p3_n75,
                      radiation = :simple,
                      surface = :prescribed_fluxes,
                      wind_nudging_timescale = nothing,
                      vertical_advection = :full_field,
                      upper_boundary_relaxation = Bool(namelist["doupperbound"]),
                      sponge = Bool(namelist["dodamping"]) ? SAMSponge() : nothing,
                      aerosol_replenishment = nothing)
        nz == length(defaults.z_faces) - 1 ||
            throw(ArgumentError("Covert-public-bin preset: namelist caseid has $nz levels but the reconstructed vertical grid has $(length(defaults.z_faces) - 1)"))
    elseif preset === :lasso_ena_official
        grd_path = joinpath(data_dir, "grd")
        isfile(grd_path) || throw(ArgumentError("preset $preset needs the vertical grid file $(grd_path) from the samin bundle"))
        require_namelist!(namelist, ["dx", "dy", "dt", "nstop", "day0", "latitude0", "longitude0",
                                     "sfc_flx_fxd", "ocean", "dolargescale", "dosfcforcing", "donudging_uv", "tauls",
                                     "doupperbound", "dodamping", "read_in_geostrophic_wind", "nrad"], preset)
        require_namelist_value!(namelist, "sfc_flx_fxd", false, preset)
        require_namelist_value!(namelist, "ocean", true, preset)
        require_namelist_value!(namelist, "dolargescale", true, preset)
        require_namelist_value!(namelist, "dosfcforcing", true, preset)
        require_namelist_value!(namelist, "donudging_uv", true, preset)
        centers = read_sam_grd(grd_path)
        lsf_columns = read_sam_large_scale_forcing(joinpath(data_dir, "lsf"))[1].has_geostrophic_columns
        Bool(namelist["read_in_geostrophic_wind"]) == lsf_columns ||
            throw(ArgumentError("namelist read_in_geostrophic_wind = $(namelist["read_in_geostrophic_wind"]) but the lsf file " *
                                (lsf_columns ? "has" : "lacks") * " the ug/vg columns (with .false. the LASSO SAM aliases ug/vg to uls/vls, which the reader reproduces)"))
        sam_ug = Float64(get(namelist, "ug", 0.0))
        sam_vg = Float64(get(namelist, "vg", 0.0))
        nx = haskey(namelist, "nx_gl") ? namelist["nx_gl"] : 256
        ny = haskey(namelist, "ny_gl") ? namelist["ny_gl"] : 256
        nz = haskey(namelist, "nz_gl") ? namelist["nz_gl"] : length(centers)
        defaults = (; label = string("LASSO-ENA official protocol (samin bundle; RRTMGP columns end at the LES top: Breeze analog of RRTMG, not padded to TOA; ",
                                     "namelist translation frame (ug, vg) = ($sam_ug, $sam_vg) NOT applied because bulk fluxes need the ground-relative wind)"),
                      Nx = nx, Ny = ny, Lx = nx * namelist["dx"], Ly = ny * namelist["dy"],
                      z_faces = faces_from_centers(centers[1:nz]),
                      day0 = Float64(namelist["day0"]),
                      latitude = Float64(namelist["latitude0"]), longitude = Float64(namelist["longitude0"]),
                      stop_time = Float64(namelist["nstop"] * namelist["dt"]),
                      Δt = Float64(namelist["dt"]),
                      max_Δt = Float64(namelist["dt"]),
                      translation_velocity = (0.0, 0.0), # bulk fluxes use the model wind; a nonzero frame is refused
                      microphysics = :p3_aer2,
                      radiation = :rrtmgp,
                      radiation_interval = Float64(namelist["nrad"] * namelist["dt"]),
                      surface = :bulk_sst,
                      wind_nudging_timescale = Float64(namelist["tauls"]),
                      vertical_advection = :full_field,
                      upper_boundary_relaxation = Bool(namelist["doupperbound"]),
                      sponge = Bool(namelist["dodamping"]) ? SAMSponge() : nothing,
                      aerosol_replenishment = :diagnostic_ccn)
    else
        throw(ArgumentError("unknown preset $preset (:covert_public_bin or :lasso_ena_official)"))
    end

    for (name, value) in overrides
        if haskey(defaults, name) && defaults[name] != value
            @warn "preset $preset: overriding $name = $(defaults[name]) with $value (recorded in provenance)"
        end
    end
    settings = merge(defaults, NamedTuple(overrides))
    settings = merge(settings, (; label = string(settings.label, isempty(overrides) ? "" : " [overrides: $(join(string.(keys(overrides)), ", "))]")))
    case = build_case(data_dir; settings...)
    return merge(case, (; preset))
end
