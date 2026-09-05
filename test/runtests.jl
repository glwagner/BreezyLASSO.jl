using Test
using BreezyLASSO
using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Units: Time
using Statistics
using Random
using TOML
using Dates: DateTime

const FIXTURES = joinpath(@__DIR__, "fixtures")
const COVERT_DIR = joinpath(@__DIR__, "..", "data", "covert2022_bin")
const HAVE_COVERT = isfile(joinpath(COVERT_DIR, "snd")) && isfile(joinpath(COVERT_DIR, "lsf")) &&
                    isfile(joinpath(COVERT_DIR, "sfc")) && isfile(joinpath(COVERT_DIR, "prm"))

# The materialized forcing of a prognostic: Breeze wraps specific-keyed forcings in
# SpecificForcing (and several in MultipleForcings).
inner(f::Breeze.Forcings.SpecificForcing) = f.forcing
inner(f) = f

test_grid(; Nx=8, Ny=8, Nz=24, Lz=6000) =
    RectilinearGrid(CPU(), Float64; size=(Nx, Ny, Nz), x=(0, 800), y=(0, 800), z=(0, Lz),
                    halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded))

@testset "BreezyLASSO" begin

@testset "SAM input files" begin
    @testset "sounding on pressure levels" begin
        snd = read_sam_sounding(joinpath(FIXTURES, "snd_pressure"))
        @test length(snd) == 2
        @test snd[1].day == 199.25 && snd[2].day == 199.5
        @test snd[1].surface_pressure == 101930
        @test all(isnan, snd[1].z)               # -9999 → missing heights, records retained
        @test snd[1].p[1] == 104000 && snd[1].θ[1] == 292.2 && snd[1].q[1] ≈ 11.2e-3
        z = record_heights(snd[1])
        @test z[1] < 0                            # below-surface level kept (SAM interpolates through it)
        @test issorted(z)
        @test issorted([r.day for r in snd])      # time monotonicity
    end

    @testset "large-scale forcing layouts" begin
        lsf7 = read_sam_large_scale_forcing(joinpath(FIXTURES, "lsf_7col"))
        @test length(lsf7) == 2
        @test !lsf7[1].has_geostrophic_columns
        @test lsf7[1].ug == lsf7[1].uls && lsf7[1].vg == lsf7[1].vls   # LASSO fallback: ug/vg alias uls/vls
        @test lsf7[2].tls[1] == -8e-5 && lsf7[2].wls[2] == -0.006
        @test issorted([r.day for r in lsf7])

        lsf9 = read_sam_large_scale_forcing(joinpath(FIXTURES, "lsf_9col_height"))
        @test lsf9[1].has_geostrophic_columns
        @test lsf9[1].ug == [3.0, 5.0, 9.0] && lsf9[1].vg == [-6.0, -7.0, -8.0]
        @test lsf9[1].tls[1] == -4e-5                                  # Fortran D exponent
        @test all(isfinite, lsf9[1].z)                                 # height-coordinate record

        @test_throws ErrorException read_sam_large_scale_forcing(joinpath(FIXTURES, "lsf_malformed"))
    end

    @testset "surface forcing and namelist" begin
        sfc = read_sam_surface_forcing(joinpath(FIXTURES, "sfc"))
        @test sfc.day == [199.25, 199.375, 199.5]
        @test sfc.latent_heat_flux[2] == 112.197 && sfc.kinematic_stress[1] == 0.0625
        prm = read_sam_namelist(joinpath(FIXTURES, "prm"))
        @test prm["caseid"] == "16x16x24"
        @test prm["sfc_flx_fxd"] === true && prm["doshortwave"] === false
        @test prm["latitude0"] == 39.05 && prm["ug"] == 5.0 && prm["vg"] == -8.0
        @test prm["nstop"] == 43200 && prm["dt"] == 0.5 && prm["day0"] == 199.25
        @test prm["tauls"] == 10800.0
        @test !haskey(prm, "comment")
        @test eltype(values(prm)) <: Union{Bool, Int, Float64, String}
    end

    @testset "hydrostatic heights (setdata.f90)" begin
        p = [104000.0, 100000.0, 90000.0]
        θ = [292.2, 292.2, 298.5]
        z = sam_hydrostatic_heights(p, θ, 101930.0)
        T1 = 292.2 * (1.04)^(287 / 1004)
        @test z[1] ≈ 287 / 9.81 * T1 * log(101930 / 104000)
        @test z[2] > 0 && z[3] > z[2]
    end

    @testset "day conversion and checksum" begin
        @test day_to_seconds(199.5, 199.25) == 21600
        @test length(file_sha256(joinpath(FIXTURES, "sfc"))) == 64
    end

    if HAVE_COVERT
        @testset "Covert public-bin regression" begin
            snd = read_sam_sounding(joinpath(COVERT_DIR, "snd"))
            lsf = read_sam_large_scale_forcing(joinpath(COVERT_DIR, "lsf"))
            sfc = read_sam_surface_forcing(joinpath(COVERT_DIR, "sfc"))
            prm = read_sam_namelist(joinpath(COVERT_DIR, "prm"))
            @test length(snd) == 4 && length(lsf) == 5 && length(sfc.day) == 5
            @test snd[1].surface_pressure == 101930
            z = record_heights(snd[1])
            @test isapprox(z[1:3], [-5.9, 1138.1, 1147.5]; atol=0.1)     # bin-paper snd: 1020, 891, 890 hPa
            bulk = joinpath(@__DIR__, "..", "data", "covert2022_bulk", "snd")
            if isfile(bulk)
                zb = record_heights(read_sam_sounding(bulk)[1])
                @test isapprox(zb[1:3], [-173.8, 35.8, 249.2]; atol=0.1)  # bulk-paper snd: 1040, 1015, 990 hPa
            end
            @test !lsf[1].has_geostrophic_columns
            @test prm["caseid"] == "256x256x192" && prm["dx"] == 35.0 && prm["nstop"] * prm["dt"] == 21600
        end
    else
        @info "Covert public inputs not present in data/; skipping the regression fixture (run scripts/fetch_inputs.jl)"
    end
end

@testset "SAM column interpolation (forcing.f90)" begin
    # pressure coordinate: linear in p; above the record top tls → 0, winds hold
    p = [105000.0, 90000.0, 70000.0]
    y = [1.0, 2.0, 3.0]
    pq = [100000.0, 80000.0, 60000.0, 50000.0]
    @test sam_interpolate_column(p, y, pq; pressure_grid=true, above=:zero) ≈ [4/3, 2.5, 0.0, 0.0]
    @test sam_interpolate_column(p, y, pq; pressure_grid=true, above=:hold) ≈ [4/3, 2.5, 2.5, 2.5]
    # below the first level: linear extrapolation through the first two levels (SAM coef < 0)
    @test sam_interpolate_column(p, y, [110000.0]; pressure_grid=true) ≈ [1 - 5000/15000]
    # height coordinate: linear in z
    z = [0.0, 1000.0, 3000.0]
    @test sam_interpolate_column(z, y, [500.0, 2000.0, 4000.0]; pressure_grid=false, above=:zero) ≈ [1.5, 2.5, 0.0]
    @test sam_interpolate_column(z, y, [500.0, 2000.0, 4000.0]; pressure_grid=false, above=:hold) ≈ [1.5, 2.5, 2.5]

    grid = test_grid()
    snd = read_sam_sounding(joinpath(FIXTURES, "snd_pressure"))
    lsf7 = read_sam_large_scale_forcing(joinpath(FIXTURES, "lsf_7col"))
    zc = Array(znodes(grid, Center()))
    pᵣ = 101930 .* exp.(-zc ./ 8000)
    profiles = LargeScaleForcingProfiles(grid, lsf7, zc, pᵣ; day0=199.25)
    @test profiles.times == [0.0, 21600.0]
    # top of the domain is above the 700 hPa record top: tls/qls/wls zero, winds held
    @test profiles.tls[1, 1, grid.Nz, Time(0.0)] == 0
    @test profiles.wls[1, 1, grid.Nz, Time(0.0)] == 0
    # winds hold the value of the level below (SAM: uu(iz) = uu(iz-1)) above the record top
    @test profiles.uls[1, 1, grid.Nz, Time(0.0)] == profiles.uls[1, 1, grid.Nz - 1, Time(0.0)]
    @test 6 < profiles.uls[1, 1, grid.Nz, Time(0.0)] ≤ 10
    @test profiles.ug[1, 1, grid.Nz, Time(0.0)] == profiles.uls[1, 1, grid.Nz, Time(0.0)]
    # linear time interpolation between the two records
    @test profiles.tls[1, 1, 1, Time(10800.0)] ≈ (profiles.tls[1, 1, 1, Time(0.0)] + profiles.tls[1, 1, 1, Time(21600.0)]) / 2
    lsf9 = read_sam_large_scale_forcing(joinpath(FIXTURES, "lsf_9col_height"))
    profiles9 = LargeScaleForcingProfiles(grid, lsf9, zc, pᵣ; day0=199.25)
    @test profiles9.has_geostrophic_columns
    @test profiles9.ug[1, 1, 1, Time(0.0)] != profiles9.uls[1, 1, 1, Time(0.0)]
    targets = SoundingTargetProfiles(grid, snd, zc, pᵣ; day0=199.25)
    @test targets.T[1, 1, 1, Time(0.0)] ≈ 292.2 * (pᵣ[1] / 1e5)^(287 / 1004)
    @test targets.q[1, 1, 1, Time(0.0)] ≈ 11.2e-3 / (1 + 11.2e-3)       # mixing ratio → mass fraction
    @test SoundingTargetProfiles(grid, snd, zc, pᵣ; day0=199.25, moisture_basis=:mass_fraction).q[1, 1, 1, Time(0.0)] ≈ 11.2e-3
    @test mass_fraction_from_mixing_ratio(0.0112) ≈ 0.0112 / 1.0112
    @test epoch_from_day_of_year(199.25) == DateTime(2017, 7, 18, 6)
    @test epoch_from_day_of_year(199.0) == DateTime(2017, 7, 18, 0)
    @test epoch_from_day_of_year(200.5) == DateTime(2017, 7, 19, 12)
end

@testset "Vertical grid and sponge" begin
    c = lasso_ena_cell_centers()
    @test length(c) == 260
    @test c[1] == 12.5 && c[241] == 6012.5 && c[end] ≈ 8087.5
    @test all(diff(c[1:241]) .≈ 25)
    @test issorted(diff(c[241:end]))
    f = lasso_ena_vertical_faces()
    @test length(f) == 261 && f[1] == 0 && issorted(f)
    fc = covert_public_bin_vertical_faces()
    @test length(fc) == 193 && fc[1] == 0 && fc[end] == 20000 && issorted(fc)
    @test all(diff(fc[1:151]) .≈ 10)
    rates = sam_sponge_rates(c, c)
    @test rates[end] ≈ 1 / 60
    @test count(>(0), rates) ≥ 1
    kbase = findfirst(>(0), rates)
    @test rates[kbase] ≈ 1 / 1800
    @test rates[kbase - 1] == 0
    @test c[end] - c[kbase + 1] < 0.3 * c[end]
end

@testset "Initial perturbation" begin
    zc = collect(12.5:25:1000)
    ϵ = perturbation_array(4, 3, zc, InitialPerturbation(depth=600, seed=7))
    ϵ′ = perturbation_array(4, 3, zc, InitialPerturbation(depth=600, seed=7))
    @test ϵ == ϵ′                                   # deterministic
    @test all(abs.(ϵ) .≤ 1)
    below = zc .≤ 600
    @test all(ϵ[:, :, .!below] .== 0)
    @test any(ϵ[:, :, below] .!= 0)
    T = 290 .+ 0.1 .* ϵ
    q = 0.01 .+ 0.025e-3 .* ϵ
    δT = vec(T[:, :, below] .- 290); δq = vec(q[:, :, below] .- 0.01)
    @test cor(δT, δq) ≈ 1
end

@testset "Saturation partition (Breeze thermodynamics)" begin
    T, qᵛ, qᶜˡ = saturation_partition(292.2, 11.2e-3, 101930.0)
    @test qᶜˡ == 0 && qᵛ == 11.2e-3
    T, qᵛ, qᶜˡ = saturation_partition(292.2, 11.2e-3, 89000.0)
    @test qᶜˡ > 0 && qᵛ + qᶜˡ ≈ 11.2e-3
    @test T > 292.2 * (0.89)^(287 / 1004)           # latent heating
end

@testset "Aerosol conversion" begin
    modes, conversion = lasso_aerosol_modes(Float64; setting=:aer2, reference_density=1.2)
    @test conversion.N₁ == 276 && conversion.N₂ == 281
    @test modes[1].number_mixing_ratio ≈ 276e6 / 1.2
    @test modes[2].mean_radius == 0.066e-6 && modes[2].geometric_std == 1.78
    @test_throws ArgumentError lasso_aerosol_modes(; setting=:aer9, reference_density=1.2)
end

@testset "Forcing operators in a model" begin
    grid = test_grid(; Nz=24, Lz=6000)
    zc = Array(znodes(grid, Center()))
    constants = ThermodynamicConstants(Float64)
    reference_state = ReferenceState(grid, constants; surface_pressure=101930, potential_temperature=z -> 292 + 0.004z)
    dynamics = AnelasticDynamics(reference_state)
    pᵣ = Array(interior(reference_state.pressure, 1, 1, :))
    times = [0.0, 3600.0]
    tls = profile_time_series(grid, times, [fill(-5e-5, 24), fill(-5e-5, 24)])
    qls = profile_time_series(grid, times, [fill(1e-7, 24), fill(1e-7, 24)])
    wls = profile_time_series(grid, times, [fill(-0.01, 24), fill(-0.01, 24)])
    uls = profile_time_series(grid, times, [fill(8.0, 24), fill(8.0, 24)])
    ug = profile_time_series(grid, times, [fill(3.0, 24), fill(5.0, 24)])
    vg = profile_time_series(grid, times, [fill(-6.0, 24), fill(-6.0, 24)])
    microphysics = SaturationAdjustment(Float64; equilibrium=WarmPhaseEquilibrium())
    thermo = large_scale_thermodynamic_forcings(tls, qls; microphysics, thermodynamic_constants=constants, moisture_name=:qᵉ)

    @testset "tls/qls physical-temperature invariant" begin
        for basis in (:mass_fraction, :mixing_ratio)
            thermo_b = large_scale_thermodynamic_forcings(tls, qls; microphysics, thermodynamic_constants=constants,
                                                          moisture_name=:qᵉ, moisture_basis=basis)
            model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics, thermodynamic_constants=constants,
                                    forcing=(; s=thermo_b.s, qᵉ=thermo_b.qᵉ))
            set!(model; T=290.0, qᵗ=8e-3)
            T₀ = copy(interior(model.temperature)); q₀ = copy(interior(model.microphysical_fields.qᵛ))
            Δt = 10.0
            for _ in 1:10
                time_step!(model, Δt)
            end
            ΔT = interior(model.temperature) .- T₀
            Δq = interior(model.microphysical_fields.qᵛ) .- q₀
            @test all(isapprox.(ΔT, -5e-5 * 100; rtol=1e-3))   # dT/dt = tls while vapor is forced
            if basis === :mass_fraction
                @test all(isapprox.(Δq, 1e-7 * 100; rtol=1e-6))    # dqᵛ/dt = qls verbatim
            else
                # SAM mixing-ratio source R: dqᵛ/dt = qᵈ² / (1 - qᶜ) R with qᶜ = 0 here
                expected = @. (1 - q₀)^2 * 1e-7 * 100
                @test all(isapprox.(Δq, expected; rtol=1e-3))
                @test all(Δq .< 1e-7 * 100)                        # ~1.6 % below the verbatim rate
            end
        end
    end

    @testset "tls/qls Jacobian in cloudy cells (P3 and one-moment moisture)" begin
        using Breeze.Thermodynamics: StaticEnergyState, MoistureMassFractions, temperature, mixture_heat_capacity
        # Apply the forcing kernels' s and qᵛ rates for Δt to a cloudy state (condensate fixed) and
        # recover T from Breeze's own state: dT/dt must be tls, dqᵛ/dt the mapped source.
        R = -1e-7; tls_value = 5e-5; Δt = 100.0
        for (label, qˡ) in (("cloudy", 5e-4), ("clear", 0.0))
            T = 288.0; qᵛ = 9e-3; p = 90000.0; z = 900.0
            q = MoistureMassFractions(qᵛ, qˡ, 0.0)
            s = mixture_heat_capacity(q, constants) * T + constants.gravitational_acceleration * z - constants.liquid.reference_latent_heat * qˡ
            qᵈ = 1 - qᵛ - qˡ
            dqᵛ = qᵈ^2 / (1 - qˡ) * R                       # moisture_basis = :mixing_ratio
            ds = mixture_heat_capacity(q, constants) * tls_value + (1850 - 1005) * T * dqᵛ
            q₁ = MoistureMassFractions(qᵛ + dqᵛ * Δt, qˡ, 0.0)
            𝒰₁ = StaticEnergyState{Float64}(s + ds * Δt, q₁, z, p)
            T₁ = temperature(𝒰₁, constants)
            @test isapprox((T₁ - T) / Δt, tls_value; rtol=2e-3)
            # ... whereas the verbatim rate or a cᵖᵐ-only mapping would not
            𝒰₂ = StaticEnergyState{Float64}(s + mixture_heat_capacity(q, constants) * tls_value * Δt, q₁, z, p)
            @test !isapprox((temperature(𝒰₂, constants) - T) / Δt, tls_value; rtol=2e-3)
            # the kernels reproduce the same rates in a one-moment model with this cloudy state
            if label == "cloudy"
                ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
                one_moment = ext.OneMomentCloudMicrophysics(Float64; cloud_formation=SaturationAdjustment(Float64; equilibrium=WarmPhaseEquilibrium()))
                tls_f = profile_time_series(grid, times, [fill(tls_value, 24), fill(tls_value, 24)])
                qls_f = profile_time_series(grid, times, [fill(R, 24), fill(R, 24)])
                thermo_1m = large_scale_thermodynamic_forcings(tls_f, qls_f; microphysics=one_moment, thermodynamic_constants=constants, moisture_name=:qᵉ)
                model_1m = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics=one_moment, thermodynamic_constants=constants,
                                           forcing=(; s=thermo_1m.s, qᵉ=thermo_1m.qᵉ))
                set!(model_1m; θ=290.0, qᵗ=12e-3)             # saturated at 290 K near the surface → cloud
                Oceananigans.TimeSteppers.update_state!(model_1m)
                f1 = Oceananigans.fields(model_1m)
                k1 = findfirst(k -> f1.qˡ[1, 1, k] > 1e-5, 1:grid.Nz)
                @test !isnothing(k1)
                fs1 = inner(model_1m.forcing.ρs); fq1 = inner(model_1m.forcing.ρqᵉ)
                qᵛ1 = f1.qᵛ[1, 1, k1]; qˡ1 = f1.qˡ[1, 1, k1]; T1 = f1.T[1, 1, k1]; qᵈ1 = 1 - qᵛ1 - qˡ1
                @test fq1(1, 1, k1, grid, model_1m.clock, f1) ≈ qᵈ1^2 / (1 - qˡ1) * R
                @test fs1(1, 1, k1, grid, model_1m.clock, f1) ≈ mixture_heat_capacity(MoistureMassFractions(qᵛ1, qˡ1, 0.0), constants) * tls_value + (1850 - 1005) * T1 * qᵈ1^2 / (1 - qˡ1) * R
            end
            # ... and in a P3 model with this state
            if label == "cloudy"
                using Breeze.Microphysics.PredictedParticleProperties: CloudDroplets
                p3 = P3Microphysics(Float64; cloud=CloudDroplets(Float64; number_concentration=75e6))
                tls_f = profile_time_series(grid, times, [fill(tls_value, 24), fill(tls_value, 24)])
                qls_f = profile_time_series(grid, times, [fill(R, 24), fill(R, 24)])
                thermo_p3 = large_scale_thermodynamic_forcings(tls_f, qls_f; microphysics=p3, thermodynamic_constants=constants, moisture_name=:qᵛ)
                model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics=p3, thermodynamic_constants=constants,
                                        forcing=(; s=thermo_p3.s, qᵛ=thermo_p3.qᵛ))
                set!(model; T=T, qᵛ=qᵛ, qᶜˡ=qˡ)
                Oceananigans.TimeSteppers.update_state!(model)
                fields = Oceananigans.fields(model)
                fs = inner(model.forcing.ρs); fq = inner(model.forcing.ρqᵛ)
                k = 8
                Tk = fields.T[1, 1, k]; qᵛk = fields.qᵛ[1, 1, k]; qˡk = fields.qᶜˡ[1, 1, k]
                qᵈk = 1 - qᵛk - qˡk
                @test fq(1, 1, k, grid, model.clock, fields) ≈ qᵈk^2 / (1 - qˡk) * R
                qk = MoistureMassFractions(qᵛk, qˡk, 0.0)
                @test fs(1, 1, k, grid, model.clock, fields) ≈ mixture_heat_capacity(qk, constants) * tls_value + (1850 - 1005) * Tk * qᵈk^2 / (1 - qˡk) * R
            end
        end
    end

    @testset "mean-profile nudging leaves eddies alone" begin
        nudge = MeanProfileNudging(uls; timescale=7200)
        model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics, thermodynamic_constants=constants,
                                forcing=(; u=nudge))
        set!(model; T=290.0, qᵗ=5e-3, u=(x, y, z) -> 4 + 0.5 * sin(2π * x / 800))
        Oceananigans.TimeSteppers.update_state!(model)
        ρ = interior(reference_state.density)
        sf = model.forcing.ρu
        @test sf isa Breeze.Forcings.SpecificForcing
        f = inner(sf)
        @test f isa MeanProfileNudging
        # kernel value is horizontally uniform (mean-based), not pointwise
        vals = [f(i, 1, 1, grid, model.clock, Oceananigans.fields(model)) for i in 1:grid.Nx]
        @test all(v ≈ vals[1] for v in vals)
        @test vals[1] ≈ (8 - 4) / 7200
        @test sf(1, 1, 1, grid, model.clock, Oceananigans.fields(model)) ≈ (8 - 4) / 7200 * ρ[1, 1, 1]
    end

    @testset "time-varying geostrophic forcing" begin
        geo = time_varying_geostrophic_forcings(ug, vg)
        model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics, thermodynamic_constants=constants,
                                coriolis=FPlane(f=1e-4), forcing=(; u=geo.u, v=geo.v))
        set!(model; T=290.0, qᵗ=5e-3)
        fu = inner(model.forcing.ρu)
        fv = inner(model.forcing.ρv)
        model.clock.time = 0.0
        @test fv(1, 1, 1, grid, model.clock, Oceananigans.fields(model)) ≈ 1e-4 * 3.0
        model.clock.time = 3600.0
        @test fv(1, 1, 1, grid, model.clock, Oceananigans.fields(model)) ≈ 1e-4 * 5.0
        @test fu(1, 1, 1, grid, model.clock, Oceananigans.fields(model)) ≈ -1e-4 * (-6.0)
        # the nudging target (uls) is distinct from the geostrophic wind (ug)
        @test uls[1, 1, 1, Time(0.0)] != ug[1, 1, 1, Time(0.0)]
    end

    @testset "full-field upwind vertical advection" begin
        vadv = LargeScaleVerticalAdvection(wls)
        model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics, thermodynamic_constants=constants,
                                forcing=(; s=vadv, u=vadv))
        set!(model; T=(x, y, z) -> 290 - 0.005z + 0.5 * (x > 400), qᵗ=5e-3, u=(x, y, z) -> 0.001z)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)
        fs = inner(model.forcing.ρs)
        fields = Oceananigans.fields(model)
        s = fields.s
        Δz = 6000 / 24
        for i in (1, 8), k in (2, 12)
            # wls < 0 → upwind from above: -(w) (s[k+1] - s[k]) / Δz
            expected = -(-0.01) * (s[i, 1, k+1] - s[i, 1, k]) / Δz
            @test fs(i, 1, k, grid, model.clock, fields) ≈ expected
        end
        @test fs(1, 1, 1, grid, model.clock, fields) == 0       # SAM skips the bottom cell
        @test fs(1, 1, 24, grid, model.clock, fields) == 0      # ... and the top cell
        # pointwise: differs between the two halves of the domain
        @test fs(1, 1, 12, grid, model.clock, fields) != fs(8, 1, 12, grid, model.clock, fields) skip=true
        fu = inner(model.forcing.ρu)
        @test fu(1, 1, 12, grid, model.clock, fields) ≈ 0.01 * 0.001
    end

    @testset "SAM sponge and upper-boundary relaxation" begin
        sponge = SAMSponge()
        targets_T = profile_time_series(grid, times, [fill(280.0, 24), fill(280.0, 24)])
        targets_q = profile_time_series(grid, times, [fill(1e-3, 24), fill(1e-3, 24)])
        upper = upper_boundary_relaxation_forcings(targets_T, targets_q; microphysics, thermodynamic_constants=constants, moisture_name=:qᵉ)
        model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics, thermodynamic_constants=constants,
                                forcing=(; u=sponge, v=sponge, w=sponge, s=upper.s, qᵉ=upper.qᵉ))
        set!(model; T=290.0, qᵗ=5e-3, u=(x, y, z) -> 5 + sin(2π * x / 800), w=0)
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies=false)
        fields = Oceananigans.fields(model)
        fu = inner(model.forcing.ρu)
        @test fu(1, 1, 24, grid, model.clock, fields) ≈ -(1 / 60) * (fields.u[1, 1, 24] - 5)
        @test fu(1, 1, 1, grid, model.clock, fields) == 0
        fq = inner(model.forcing.ρqᵉ)
        @test fq(1, 1, 24, grid, model.clock, fields) ≈ -(5e-3 - 1e-3) / 3600
        @test fq(1, 1, 23, grid, model.clock, fields) ≈ -(5e-3 - 1e-3) / 3600
        @test fq(1, 1, 22, grid, model.clock, fields) == 0
        fs = inner(model.forcing.ρs)
        @test fs(1, 1, 24, grid, model.clock, fields) < 0        # T = 290 relaxed toward 280
        @test fs(1, 1, 22, grid, model.clock, fields) == 0

        # forcing-only step: the top levels follow dT/dt = -(T - Tg0)/τ, dqᵛ/dt = -(q - qg0)/τ
        model2 = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics, thermodynamic_constants=constants,
                                 forcing=(; s=upper.s, qᵉ=upper.qᵉ))
        set!(model2; T=290.0, qᵗ=5e-3)
        T₀ = copy(interior(model2.temperature)); q₀ = copy(interior(model2.microphysical_fields.qᵛ))
        for _ in 1:5
            time_step!(model2, 10.0)
        end
        ΔT = interior(model2.temperature) .- T₀
        Δq = interior(model2.microphysical_fields.qᵛ) .- q₀
        # exponential relaxation of the *actual* initial state toward the targets
        expected_ΔT = @. -(T₀[:, :, 23:24] - 280) * (1 - exp(-50 / 3600))
        expected_Δq = @. -(q₀[:, :, 23:24] - 1e-3) * (1 - exp(-50 / 3600))
        @test all(isapprox.(ΔT[:, :, 23:24], expected_ΔT; rtol=5e-3))
        @test all(isapprox.(Δq[:, :, 23:24], expected_Δq; rtol=5e-3))
        @test all(abs.(ΔT[:, :, 1:22]) .< 1e-8)
    end
end

@testset "Prescribed surface stress is uniform and wind-aligned" begin
    grid = test_grid(; Nz=24, Lz=6000)
    constants = ThermodynamicConstants(Float64)
    reference_state = ReferenceState(grid, constants; surface_pressure=101930, potential_temperature=292)
    dynamics = AnelasticDynamics(reference_state)
    sfc = read_sam_surface_forcing(joinpath(FIXTURES, "sfc"))
    bcs, stress = prescribed_surface_flux_boundary_conditions(grid, sfc, 199.25; thermodynamic_constants=constants,
                                                              surface_density=1.2, moisture_name=:qᵉ)
    microphysics = SaturationAdjustment(Float64; equilibrium=WarmPhaseEquilibrium())
    model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics, thermodynamic_constants=constants,
                            boundary_conditions=bcs)
    set!(model; T=290.0, qᵗ=5e-3, u=(x, y, z) -> 3 + 2 * sin(2π * x / 800), v=(x, y, z) -> -4 + cos(2π * y / 800))
    simulation = Simulation(model; Δt=1.0, stop_time=1.0)
    updater = prescribed_stress_updater(stress, model.velocities)
    updater(simulation)
    τˣ = interior(stress.τˣ); τʸ = interior(stress.τʸ)
    @test all(τˣ .≈ τˣ[1, 1, 1]) && all(τʸ .≈ τʸ[1, 1, 1])
    ū = mean(interior(model.velocities.u, :, :, 1)); v̄ = mean(interior(model.velocities.v, :, :, 1))
    U = max(1, sqrt(ū^2 + v̄^2))
    @test τˣ[1, 1, 1] ≈ -1.2 * 0.0625 * ū / U
    @test τʸ[1, 1, 1] ≈ -1.2 * 0.0625 * v̄ / U
    # energy flux includes the temperature-neutral evaporation term
    ℒ = constants.liquid.reference_latent_heat
    H = model.formulation.energy_density.boundary_conditions.bottom.condition[1, 1, 1, Time(0.0)]
    @test H ≈ 11.5361 + (1850 - 1005) * 294.937 * 85.8638 / ℒ
end

@testset "Scalar advection: bounded water masses, plain energy and moments" begin
    using Oceananigans.Advection: BoundsPreservingWENO
    is_bounded(scheme) = scheme isa BoundsPreservingWENO
    for (scheme, moisture) in ((:p3_aer2, :qᵛ), (:p3_n75, :qᵛ), (:one_moment, :qᵗ))
        microphysics, _ = BreezyLASSO.build_microphysics(Float64, scheme; droplet_number=75e6, surface_density=1.17)
        schemes = BreezyLASSO.scalar_advection_schemes(5, microphysics, moisture)
        @test !is_bounded(schemes.ρs)
        @test is_bounded(schemes[Symbol("ρ", moisture)])
        for name in keys(schemes)
            name === :ρs && continue
            s = string(name)
            if occursin("ρq", s)
                @test is_bounded(schemes[name])   # every water-mass tracer
            else
                @test !is_bounded(schemes[name])  # number (ρn*) and volume (ρb*) moments
            end
        end
        # diagnostic sensitivity: plain WENO for the condensate masses only
        plain = BreezyLASSO.scalar_advection_schemes(5, microphysics, moisture; bounded_condensates=false)
        @test is_bounded(plain[Symbol("ρ", moisture)])
        @test all(!is_bounded(plain[n]) for n in keys(plain) if n != Symbol("ρ", moisture))
    end
    aer2, _ = BreezyLASSO.build_microphysics(Float64, :p3_aer2; droplet_number=75e6, surface_density=1.17)
    schemes = BreezyLASSO.scalar_advection_schemes(5, aer2, :qᵛ)
    @test all(is_bounded, (schemes.ρqᶜˡ, schemes.ρqʳ, schemes.ρqⁱ, schemes.ρqᶠ, schemes.ρqʷⁱ))
    @test all(!is_bounded, (schemes.ρnᶜˡ, schemes.ρnʳ, schemes.ρnⁱ, schemes.ρbᶠ, schemes.ρnᵃ))
    # positivity-only limiter for the moments: lower bound 0, no upper bound
    positive = BreezyLASSO.scalar_advection_schemes(5, aer2, :qᵛ; positive_moments=true)
    @test all(is_bounded, (positive.ρnᶜˡ, positive.ρnʳ, positive.ρnⁱ, positive.ρbᶠ, positive.ρnᵃ))
    @test positive.ρnʳ.bounds.minimum_value == 0 && isinf(positive.ρnʳ.bounds.maximum_value)
    @test positive.ρqʳ.bounds.maximum_value == 1
    @test !is_bounded(positive.ρs)
end

@testset "Sedimenting rain carries its enthalpy" begin
    using Breeze.Thermodynamics: saturation_specific_humidity, PlanarLiquidSurface
    using Breeze.Microphysics.PredictedParticleProperties: CloudDroplets
    grid = test_grid(; Nz=40, Lz=1000)
    constants = ThermodynamicConstants(Float64)
    reference_state = ReferenceState(grid, constants; surface_pressure=101300, potential_temperature=290)
    dynamics = AnelasticDynamics(reference_state)
    zc = Array(znodes(grid, Center()))
    ρᵣ = Array(interior(reference_state.density, 1, 1, :))
    p3 = P3Microphysics(Float64; cloud=CloudDroplets(Float64; number_concentration=75e6))
    scalar_advection = BreezyLASSO.scalar_advection_schemes(5, p3, :qᵛ)
    T₀ = 285.0
    qsat = [saturation_specific_humidity(T₀, ρᵣ[k], constants, PlanarLiquidSurface()) for k in 1:40]
    col(v) = repeat(reshape(v, 1, 1, 40), 8, 8, 1)
    qʳ₀ = [600 ≤ z ≤ 750 ? 2e-3 : 0.0 for z in zc]
    nʳ₀ = qʳ₀ ./ (4/3 * π * 1000 * (0.5e-3)^3)
    function rain_column(with_enthalpy)
        forcing = with_enthalpy ? (; ρs = sedimentation_enthalpy_forcings(p3, scalar_advection; thermodynamic_constants=constants)) : NamedTuple()
        model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics=p3, thermodynamic_constants=constants,
                                momentum_advection=WENO(order=5), scalar_advection, forcing)
        set!(model; T=T₀, qᵛ=col(qsat), qʳ=col(qʳ₀), nʳ=col(nʳ₀))
        Tᵢ = copy(interior(model.temperature))
        for _ in 1:60
            time_step!(model, 1.0)
        end
        return interior(model.temperature) .- Tᵢ
    end
    ΔT_without = rain_column(false)
    ΔT_with = rain_column(true)
    @test maximum(abs, ΔT_without) > 1      # rain arriving without its enthalpy warms by ~ℒΔqʳ/cᵖ
    @test maximum(abs, ΔT_with) < 0.05       # with the transport the column stays isothermal

    # Donor-cell content: cold rain from above the jump (T = 280 K) falling into warm air
    # (285 K) cools the receiving cells by Δq (cˡ - cᵖᵈ)(T_cold - T_warm) / cᵖᵐ of the rain
    # that has passed, which a face-centered content would halve at the jump.
    T_jump = [z > 600 ? 280.0 : 285.0 for z in zc]
    qsat_jump = [saturation_specific_humidity(T_jump[k], ρᵣ[k], constants, PlanarLiquidSurface()) for k in 1:40]
    forcing = (; ρs = sedimentation_enthalpy_forcings(p3, scalar_advection; thermodynamic_constants=constants))
    model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics=p3, thermodynamic_constants=constants,
                            momentum_advection=WENO(order=5), scalar_advection, forcing)
    set!(model; T=col(T_jump), qᵛ=col(qsat_jump), qʳ=col(qʳ₀), nʳ=col(nʳ₀))
    Tᵢ = copy(interior(model.temperature)); qʳᵢ = copy(interior(model.microphysical_fields.qʳ))
    for _ in 1:8   # ~8 s: the rain front crosses the jump at 612.5 m but stays above the bottom
        time_step!(model, 1.0)
    end
    ΔT = interior(model.temperature) .- Tᵢ
    Δqʳ = interior(model.microphysical_fields.qʳ) .- qʳᵢ
    k_warm = findlast(≤(600), zc)                       # first warm cell below the jump
    cᵖᵐ = 1005 * (1 - qsat_jump[k_warm]) + 1850 * qsat_jump[k_warm]
    expected = Δqʳ[1, 1, k_warm] * (4181 - 1005) * (280 - 285) / cᵖᵐ
    @test Δqʳ[1, 1, k_warm] > 1e-4                      # rain has arrived in the warm cell
    @test ΔT[1, 1, k_warm] < 0                           # ... and cooled it (donor-cell content)
    # the cell's cumulative inflow exceeds the net gain (rain also leaves below), so the
    # cooling is larger than the net-gain estimate but bounded by the inflow over the step
    @test expected * 4 < ΔT[1, 1, k_warm] < expected
    @test abs(ΔT[1, 1, k_warm - 1]) < 5e-3               # rain leaving at 285 K does not cool the next cell
    @test all(abs.(ΔT[:, :, 1:k_warm-3]) .< 0.05)        # only trace effects below the front
    forcings = sedimentation_enthalpy_forcings(p3, scalar_advection; thermodynamic_constants=constants)
    @test length(forcings) == 4               # cloud liquid, rain, ice, liquid on ice
    @test Breeze.AtmosphereModels.is_density_tendency_forcing(forcings[1])
end

@testset "Simple longwave radiation model" begin
    grid = test_grid(; Nz=24, Lz=3000)
    constants = ThermodynamicConstants(Float64)
    reference_state = ReferenceState(grid, constants; surface_pressure=101930, potential_temperature=292)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment(Float64; equilibrium=WarmPhaseEquilibrium())
    radiation = SimpleLongwaveRadiation(grid; schedule=IterationInterval(1))
    model = AtmosphereModel(grid; formulation=:StaticEnergy, dynamics, microphysics, thermodynamic_constants=constants, radiation)
    set!(model; θ=(x, y, z) -> z < 1000 ? 290 : 296 + 0.003z, qᵗ=(x, y, z) -> z < 1000 ? 11e-3 : 3e-3)
    Oceananigans.TimeSteppers.update_state!(model)
    F = interior(radiation.flux); H = interior(radiation.flux_divergence)
    @test all(isfinite, F) && all(isfinite, H)
    @test F[1, 1, 1] ≈ 22 + 113 * exp(-85 * sum(interior(reference_state.density) .* max.(interior(model.microphysical_fields.qˡ, 1, 1, :), 0) .* (3000 / 24)))
    @test any(H .< 0)                                  # cloud-top cooling
    @test radiation.schedule isa IterationInterval
    # radiation updates on the schedule after iteration 0
    model.clock.iteration = 1
    fill!(radiation.flux_divergence, 0)
    Breeze.AtmosphereModels.update_radiation!(radiation, model)
    @test any(interior(radiation.flux_divergence) .!= 0)
end

if HAVE_COVERT
    @testset "Covert public-bin preset builds and steps" begin
        case = lasso_ena_simulation(COVERT_DIR; preset=:covert_public_bin, Nx=8, Ny=8, Lx=280, Ly=280,
                                    z_faces=collect(range(0, 6000, length=25)), microphysics=:one_moment,
                                    stop_time=4.0, write_output=false, progress_interval=100)
        @test occursin("Covert-public-bin", case.config.label)
        @test case.config.surface == "prescribed_fluxes" && case.config.radiation == "simple"
        @test case.config.wind_nudging_timescale == 0
        @test case.simulation.stop_time == 4.0
        run!(case.simulation)
        @test all(f -> all(isfinite, interior(f)), values(Oceananigans.prognostic_fields(case.model)))
        @test_throws ArgumentError lasso_ena_simulation(COVERT_DIR; preset=:lasso_ena_official)
        # preset fidelity: fixed SAM time step and namelist translation frame
        @test case.config.Δt_initial == 0.5 && case.config.max_Δt == 0.5
        @test case.config.translation_velocity_u == 5.0 && case.config.translation_velocity_v == -8.0
        # provenance round-trips through TOML
        path = joinpath(mktempdir(), "provenance.toml")
        write_provenance(path, case; extra=(; note="test", tuple=(1, 2)))
        record = TOML.parsefile(path)
        @test record["preset"] == "covert_public_bin"
        @test record["config"]["microphysics"] == "one_moment"
        @test haskey(record["inputs"], "snd_sha256") && haskey(record["inputs"], "prm_sha256")
        @test record["software"]["Breeze_source"] isa String
        @test occursin("877618e", record["software"]["Oceananigans_source"])
        @test occursin("5fc404c", record["software"]["Breeze_source"])
        @test record["extra"]["tuple"] == [1, 2]
        # output writers build (profiles of already-averaged fields, time series, slices)
        written = lasso_ena_simulation(COVERT_DIR; preset=:covert_public_bin, arch=CPU(), FT=Float32, Nx=8, Ny=8, Lx=280, Ly=280,
                                       z_faces=collect(range(0, 6000, length=25)), microphysics=:p3_aer2,
                                       aerosol_replenishment=:diagnostic_ccn, stop_time=1.0, write_output=true,
                                       output_dir=mktempdir(), progress_interval=100)
        @test Set(keys(written.simulation.output_writers)) ⊇ Set((:profiles, :timeseries, :slices))
        @test :cloud_fraction ∈ keys(written.simulation.output_writers[:profiles].outputs)
    end
end

end # BreezyLASSO
