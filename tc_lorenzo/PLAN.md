# Hurricane Lorenzo (2019): an ERA5-nested tropical-cyclone downscaling toward 100 m

Claude, 12 September 2026. Working directory `~/tc_hindcast`; results and plans here in
BreezyLASSO.jl at Greg's request. Codex is on tmux pane 24.

## 1. The storm, and why this one

Searched IBTrACS v04r01 North Atlantic, 1990–2024, for storms whose peak intensity coincided with
maximum isolation from land. The `DIST2LAND` column answers the constraint directly.

| year | storm | peak | lat | lon | kt | dist to land |
|---|---|---|---:|---:|---:|---:|
| **2019** | **LORENZO** | 29 Sep 03 UTC | 24.3°N | 45.0°W | **140** | **2229 km** |
| 2024 | Kirk | 4 Oct 00 | 21.2 | −47.1 | 130 | 1876 |
| 2004 | Karl | 21 Sep 06 | 19.6 | −47.3 | 125 | 1706 |
| 2010 | Julia | 15 Sep 12 | 17.7 | −32.2 | 120 | 1605 |
| 2003 | Isabel | 11 Sep 18 | 21.5 | −54.8 | 145 | 1180 |
| 2021 | Sam | 26 Sep 18 | 14.1 | −50.3 | 135 | 1012 |

**Lorenzo is the most isolated intense Atlantic cyclone in the modern record**: Category 5,
140 kt, 2229 km from the nearest land. Isabel and Sam are comparably strong but sit 1000–1200 km
out, which is too close for a 20° domain. Lorenzo gives more than 1000 km of margin on every side,
so **the domain is pure ocean and `SurfacePartition` is not needed** — prescribed SST alone, which
is what Greg asked for.

2019 also means high-quality ERA5, and the storm is well observed by geostationary and polar
satellites for later validation.

## 2. Window and domain

Track from IBTrACS, three-hourly:

```
28 Sep 00 UTC  20.3N 44.2W  105 kt   952 mb   moving NNW at 9 kt
29 Sep 00 UTC  23.8N 45.0W  130 kt   936 mb
29 Sep 03 UTC  24.3N 45.0W  140 kt   925 mb   <- Category 5 peak
```

**Window: 28 Sep 00 UTC → 30 Sep 00 UTC (48 h).** This brackets a re-intensification from 100 to
140 kt — a rapid-intensification event, which is both the most interesting target and the most
demanding test of the microphysics and the closure.

**Outer domain: 55–35°W × 13–33°N**, 20° × 20°, centred near the track mean. The storm translates
about 650 km northward inside it, staying clear of every boundary. Nearest land: Cabo Verde at
15–17°N, 22–25°W, ten degrees east of the eastern boundary; the Lesser Antilles six degrees west of
the western one.

## 3. The refinement ladder — revised, on measured allocations

The first version of this section carried an arithmetic error and a guessed cell budget. Both are
replaced here by Codex's exact-shape memory census, validated against real GPU allocations: a
mixed-phase 1M + TKE model at 64³×150 allocates 240,583,440 bytes on an A100, matching the
CPU-shape prediction exactly, and the same agreement holds at 128×128×150. The GPU also carries a
radiation `TransposedStateCache` of `4·ncol·(5·nz+1)` bytes that a CPU-shape count misses.

Two corrections to what was written before:

* the old table claimed 118 GB was "beyond 4 × 80 GB" — 4 × 80 GB is 320 GB, and the claim was
  simply wrong;
* the level count is **86**, not the ~100–150 assumed, because that is what the stated vertical
  spacing produces. The driver no longer accepts a bare level count at all (§6).

Outer domain: 2047 km zonal at 23°N by 2224 km meridional. Device capacity is **79.14 GiB
accessible** per A100, not the nominal 80.

| stage | spacing | domain | cells (Nz 86) | GPUs | GiB/rank | fits |
|---|---|---|---:|---:|---:|---|
| L0 | 6 km | 20° × 20° | 10.9 M | 1 | ~6 | yes |
| L1 | 3 km | 20° × 20° | 43.5 M | 1 | ~23 | yes |
| L2 | 1.5 km | 20° × 20° | 174 M | 4 | ~22 | yes |
| L3 | 750 m | 10° × 10° | 174 M | 4 | ~22 | yes |
| L4 | 100 m | 2° × 2° | 392 M | 4 | **49.8** | yes, subject to peak |
| L4 | 100 m | 2° × 2°, Nz 150 | 683 M | 4 | **84.3** | **no** — exceeds 79.14 |
| L4′ | 100 m | 4° × 4°, Nz 150 | 2.7 G | 32 | 40.8 | Perlmutter |
| L4′ | 100 m | 4° × 4°, Nz 150 | 2.7 G | 64 (40 GiB) | 20.5 | Perlmutter |

The headline changes: **100 m is reachable locally at 86 levels on a 2° × 2° box** and is not
reachable at 150 levels. That makes the fine-stage memory a *choice about vertical resolution*
rather than a hard wall — and it is the cheap 86-level mesh that deserves the scepticism, because at
100 m horizontal a 500 m cell aloft is a 1:5 aspect ratio and an eyewall updraft resolved that way
is not an LES in the vertical.

The independent limit is **residence time**, and it is what actually rules the small box out for
production. Lorenzo translates at 9 kt, about 400 km per day, and a 2° box is 205 km across: a fixed
2° window loses the storm in well under a day. The working plan is therefore a fixed **4° × 6°**
nest, recentred between stages, holding roughly 12 h per stage — larger fixed nests recentred
between refinements rather than new moving-nest machinery.

Each stage restarts from the previous stage's saved state rather than from rest, so the vortex is
inherited and only the newly resolved scales spin up.

## 4. Blockers: one closed, one open, one changed shape

### 4.1 Multi-GPU — transport verified, the CALLER is broken

The `MPIError(17)` on a bare `LatitudeLongitudeGrid` with one `CenterField` and one
`fill_halo_regions!` was confirmed on a compute node (job 1077):

```
mca:mpi:base:param:mpi_built_with_cuda_support:value:false
```

The cluster's OpenMPI is not CUDA-aware and there is no UCX, so Oceananigans' device-resident
communication buffers are handed to an MPI that cannot read them. The resolution is neither of the
two routes proposed before: Oceananigans' **NCCL extension** bypasses MPI for the halo exchange
entirely. `NCCLDistributed(GPU(); partition = Partition(x, y))` provides
`distributed_fill_halo_event!` with corner exchanges, and host-scalar collectives still work through
the MPI handle the NCCL communicator carries. NERSC supports NCCL over Slingshot through
`nccl-plugin`, so the same path carries to Perlmutter.

The driver takes `--arch nccl --ranks px,py`.

**One trap, found by testing rather than by reading.** The exchange still failed until the Slurm
binding was fixed. With `--gpus-per-task=1` each rank gets a cgroup holding one GPU, so every rank
sees `ndevices == 1` and calls `device!(0)`, and NCCL's peer-to-peer IPC import fails **mid-run**:

```
transport/p2p.cc:290 (ncclP2pImportShareableBuffer) NCCL WARN Cuda failure 101 'invalid device ordinal'
```

The devices really are distinct — a UUID check passes — which is what makes it confusing. Each task
must see every GPU on the node; Oceananigans then assigns `device!(node_rank % ndevices)` itself.
With that removed, the halo exchange is **bitwise exact** at Center, XFace and YFace locations,
including the asynchronous deferred-unpack path with three fields in flight — on 2 ranks (job 1092)
and on 4 ranks in a 2 × 2 partition (job 1094), where each rank also exchanges a diagonal corner.
The test poisons every halo with NaN and predicts each cell from its GLOBAL INDEX, so a skipped
exchange leaves NaN rather than a plausible number and a mis-assembled corner is an O(1) error;
predicting from coordinates instead would have forced a tolerance, because two ranks can round a
node position differently in the last bit.

**What remained untested was a real model step under NCCL — and it fails.** A probe reproducing
Breeze's own `update_state!` call sequence on two GPUs shows that after asynchronous halo fills are
launched on density, momentum AND tracers, only density and momentum are synchronised: the run is
left with `pending_unpacks = 3` and the **tracer neighbour halo still holding its poison value**. An
explicit `synchronize_communication!(tracer)` replaces it with the correct neighbour value and clears
the queue. So the exchange is right and the completion is missing from the caller — which means every
distributed Breeze run reads stale tracer halos: moisture, every microphysical species, and the
closure's TKE.

This is worth stating carefully because the two facts are easy to conflate. **A bitwise-exact
transport gate and a broken model are entirely compatible**: our halo test exercised Oceananigans'
exchange with synthetic fields and passed on 2 and 4 ranks including corners, while the defect sits
one level up in how Breeze calls it. Multi-GPU is therefore blocked again, for a completely different
reason than the MPI problem it replaced, and the same defect would meet the coupled-model benchmark
on Perlmutter.

The minimal fix is `async = false` in Breeze's `update_state!`, or equivalently an explicit
completion immediately after the asynchronous fill — identical ordering. Genuine overlap would need
separate interior work and boundary completion, which Breeze does not currently schedule, so there is
no performance being protected by the present arrangement.

The same check belongs in the Perlmutter readiness list as a named test rather than an assumption:
it presents as a CUDA error in the middle of the first exchange, not as a setup failure.

### 4.2 Float32 at fine resolution — OPEN, unchanged

At ENA, Float32 produced NaNs at 3 km three times (17 min, 36 min, 3.4 h) where Float64 ran clean;
the cost of the Float64 workaround is 1.55×. The first non-finite value appeared in dry density and
61 complete columns failed in one step. The generator is still unlocalized. My proposed mechanism —
cancellation in the bottom continuity flux — was refuted: at an impermeable flat bottom the lower
face flux is zero, and a pinned-solver probe passed 26 assertions with finite inputs in both
precisions. What the probe did establish is that a single injected NaN at one face contaminates all
113 levels of a column, which is why the original "level 1, dry density" reading was a scan-order
artifact and not a surface-origin failure.

The TC differs from ENA in ways that could help or hurt: no terrain, no `SurfacePartition`, a warm
deep tropical column rather than a shallow capped boundary layer. `--nancheck` is in the driver from
the start, reporting every bad field with distinct column counts and level ranges rather than a
column-major `first(findall)`.

### 4.3 Restart — the missing method found and written; the test is the real work

The gap was one level lower than it looked. `prognostic_state` resolves cleanly through
`Simulation` → `EarthSystemModel` → `NestedModel`, and then falls through the universal fallback
`prognostic_state(obj) = obj` **at the Breeze `AtmosphereModel`, which defines neither method**. So
the checkpointer would be asked to serialize an entire model and pickup would raise a `MethodError`.

`~/tc_hindcast/breeze_checkpointing.jl` supplies the two methods, taking the state from Breeze's own
`prognostic_fields(model)` so that changing microphysics or adding a closure changes what is
checkpointed without the patch knowing. Both Breeze timesteppers return `nothing`, on the claim that
`U⁰`, `Gⁿ` and the acoustic substepper's perturbation fields are intra-step scratch.

That claim is tested, not asserted: `restart_round_trip.jl` runs the same coarse nest three ways —
continuous for 2n steps, stopped at n with a checkpoint, and picked up from that checkpoint — and
compares the two end states field by field. The pass condition is **exact** equality, since it is
identical arithmetic on identical inputs.

Cross-grid restart is a further step: `regrid!` conserves Center fields on spherical grids to
2.8 × 10⁻¹⁵ but throws on XFace momentum, so momentum transfer between stages needs its own strategy.

### 4.4 AIVA is not usable, on this pin OR on 0.112 — new, and it reorders §5

Greg asked for AIVA and for Oceananigans 0.112's bounded WENO. Four separate findings say no, and
the last of them is a bug rather than a configuration problem.

**On the current pin, AIVA is a silent no-op with bounded WENO.** A dispatch probe shows the
bounded-WENO `update_advection!` method is the more specific one and wins, so the AIVA timestep
update never runs: the probe measures `dt = 0.0` after an update that requested 0.125. Oceananigans
0.112 (#5933) restructures the call so both run, and measures 0.125. Since every water mass carries
`bounds = (0, 1)`, enabling AIVA on this pin disables exactly the update it needs on exactly the
fields we care about most.

**A blind 0.112 bump is also wrong.** The pinned revision carries two Float32 protections absent from
the tagged 0.112 source: the lower limiter denominator's minus-epsilon guard against 0/0 at tiny
undershoots, and a ZWENO weight-ratio cap against Float32 overflow — the same class of protection as
the failure mode in §4.2. Losing them while changing the limiter's structure would make any new
Float32 failure uninterpretable. A preservation patch exists and passes `git apply --check` against
the tag, unapplied.

**And on 0.112 itself, bounded + AIVA is unsound.** `bounded_tracer_flux_divergence_z` reads the raw
face velocities and never applies AIVA's `explicit_velocity_scale`, which Breeze's bounded `div_ρUc`
calls directly. Measured on the tagged implementation at Δz 1 m, Δt 1 s, explicit CFL 0.1, total
w = −10 m/s, on a smooth tracer whose limiter is one:

| operator | explicit flux relative to fully explicit |
|---|---:|
| ordinary WENO + AIVA | 0.0100 |
| bounded WENO + AIVA | **1.0** |
| required | 0.01 |

One forward-Euler explicit stage then takes an outlet cell of a pulse bounded in [0, 1] from
**1 to −9**, where ordinary AIVA takes it to 0.9 — a bounds-preserving scheme producing a value nine
times outside its own bounds.

**The implicit half never sees the fall speed either.** The explicit tendency forms
`sum_of_velocities(velocities, microphysical_velocities(…))`, but `scalar_substep!` hands
`tendency_transport_velocities(model)` to every species' `implicit_step!` and SSP-RK3 passes the
resolved velocities unchanged. With resolved w = 0 and a 10 m/s fall speed the correct implicit
remainder is −9.9 and the supplied one is 0; with a +12 updraft against the same fall speed it is
+1.9 against +11.9. So the two halves of the split do not transport the same thing.

The consequence for this plan is concrete: the **sedimentation CFL is a real constraint**, not a
conservatism to be removed, and L0's step is set by it — τ ≈ Δz / fall speed ≈ 5 s at Δz 50 m, so
the wizard at a summed-Courant target of 0.25 settles near Δt ≈ 1.25 s, roughly 138,000 steps for
48 h. **No AIVA speedup is in any budget in this plan.**

AIVA is therefore a four-gate item: #5933's dispatch fix, both Float32 protections preserved,
bounded vertical fluxes routed through a conservative AIVA velocity split with each species' total
velocity supplied to its implicit solve, and only then a timescale change — with the summed explicit
Courant budget respected, since dropping the vertical term from the wizard does not make AIVA's
explicit vertical fraction vanish.

## 5. Order of work

1. **ERA5 pre-stage** for 26 Sep 12 UTC – 30 Sep 00 UTC: the nine pressure-level variables the
   nested parent requests, over the padded parent box, plus surface pressure, SST and skin
   temperature over the child box, through the batched `Downloads.download(::MetadataSet)` backend.
   The window starts 36 h before the nominal L0 start so finer stages can branch earlier than L0
   does rather than chasing a peak L0 has already passed.
2. **L0 at 6 km, Float32, 48 h** — the baseline, and the first test of whether Float32 survives a
   deep tropical convective case at all.
3. **Same-grid restart qualified** by the round-trip test above, then **cross-grid transfer**.
4. **NCCL halo validation** on 2 then 4 GPUs, then **L1 at 3 km** — the first resolution at which
   ENA's Float32 failure appeared.
5. **Perlmutter staging and a first benchmark**, in parallel with the local ladder rather than after
   it, since the 4° × 4° stages were never going to run here.
6. **0.112 in isolation**, carrying both Float32 protections, then AIVA behind its dispatch and
   conservation tests. AIVA's sedimentation-Courant relief is worth more in a cyclone than at ENA —
   the measured horizontal advective timescale is 300 s against 4.918 s for the built-in automatic
   scan at Δx 3 km, Δz 50 m, u 10 m/s, rain fall speed −10 m/s, a factor of 61 — which is exactly
   why it must be qualified rather than switched on.

## 6. Answers to the questions this plan opened

**Vertical grid.** Resolved by removing the question rather than answering it with a number. The
driver exposes the spacing that generates the mesh — `--dz` (surface spacing, default 50 m),
`--dz-constant-extent` (1000 m), `--dz-max` (500 m), `--stretching` (0.06 per cell), `--ztop`
(22 km) — and takes `Nz = length(z)`. A bare level count is no longer accepted, because one that
disagreed with the spacing would either fail in the grid constructor or, worse, silently describe a
different atmosphere. All five values are recorded in each run's `provenance.toml` alongside the
resulting `Nz`, so a stage's vertical resolution is documented as the spacing that produced it.

**SST.** ERA5 hourly sea-surface temperature, streamed three time levels at a time rather than
cached, over the child box. It gives the storm no cold wake, which is a real limitation for a 48 h
intensification study — Lorenzo's own upwelling would cool the surface beneath it and damp the
intensification we are trying to reproduce. The prognostic ocean Greg named as a future goal is
exactly the fix, and this is the diagnostic that will show whether it matters.
