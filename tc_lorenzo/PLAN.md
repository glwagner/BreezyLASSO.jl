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

## 3. The refinement ladder, with honest arithmetic

Outer domain is 2047 km zonal (at 23°N) by 2224 km meridional. Assuming ~100 vertical levels to a
20 km top:

| stage | spacing | domain | cells | Float32 fields | where |
|---|---|---|---:|---:|---|
| L0 | 6 km | 20° × 20° | 12.6 M | ~2 GB | 1 GPU |
| L1 | 3 km | 20° × 20° | 50.6 M | ~8 GB | 1 GPU |
| L2 | 1.5 km | 20° × 20° | 202 M | ~32 GB | 1 GPU (tight) or 4 |
| L3 | 750 m | 10° × 10° | 202 M | ~32 GB | 4 GPUs |
| L4 | 100 m | 2° × 2° | 739 M (Nz 150) | ~118 GB | **beyond 4 × 80 GB** |
| L4′ | 100 m | 4° × 4° | 2.96 G | ~473 GB | Perlmutter |

**So 100 m is not reachable locally on any useful domain**, which is consistent with Greg's
expectation that it moves to Perlmutter. Locally the realistic floor is **750 m on a reduced domain
with 4 GPUs**, or 1.5 km on the full domain. Everything finer is a Perlmutter job.

Each stage restarts from the previous stage's saved state rather than from rest, so the vortex is
inherited and only the newly resolved scales spin up.

## 4. Three blockers, all already measured, none speculative

### 4.1 Multi-GPU is blocked on this cluster — hard gate on L2 and beyond

Oceananigans' distributed halo exchange fails with `MPIError(17)` (truncation) on a **bare
`LatitudeLongitudeGrid` with one `CenterField` and one `fill_halo_regions!`** across two GPU ranks.
Reproducer: `~/ena_hindcast/dist_halo_mwe.jl`. Nothing of ours is involved.

Cause now confirmed on a compute node (job 1077):

```
mca:mpi:base:param:mpi_built_with_cuda_support:value:false
```

The cluster's OpenMPI 4.1.x is **not CUDA-aware**, there is no UCX, and Julia's bundled MPICH is
not CUDA-aware either. Oceananigans allocates its communication buffers with
`on_architecture(arch, …)`, so on `Distributed(GPU())` they are `CuArray`s handed straight to MPI,
which cannot read device pointers. Two routes, neither of which I can take unilaterally:

- **(a) a CUDA-aware MPI build** (UCX + OpenMPI with CUDA) — a systems task;
- **(b) stage the halo buffers through host memory** — a small, well-defined Oceananigans patch,
  slower but functional, and a sensible upstream contribution.

**Until one of these exists, only single-GPU stages (L0, L1, and L2 at the memory limit) are
runnable here.** This should be settled before the ladder reaches L3, not when it gets there.

### 4.2 Float32 fails at fine resolution — direct threat to Greg's Float32 requirement

At ENA, Float32 produced NaNs at 3 km three times (17 min, 36 min, 3.4 h); Float64 ran clean.
The first non-finite value was in the dry density, and 61 complete columns failed in one step. The
generator is still unlocalized and is with Codex for stage-local capture; the mechanism is not
cancellation in the bottom continuity flux, which was my hypothesis and was refuted. Cost of the
workaround is 1.55×.

Relevant differences here that may help or hurt: no terrain, no `SurfacePartition`, warm deep
tropics rather than a shallow capped boundary layer, and **Oceananigans 0.112's bounded WENO**,
which Greg specifically wants and which we have not yet tried. It is plausible the ENA failure was
configuration-specific; it is not safe to assume so. **The NaN localiser (`--nancheck`) goes into
the TC driver from the start**, and L1 at 3 km is the first place to watch.

### 4.3 No restart support — required by the ladder, does not exist

`run_hindcast.jl` has no `Checkpointer` and no pickup path. The whole refinement strategy depends
on restarting a finer grid from a coarser saved state, which additionally needs **interpolation
between grids**, not just a checkpoint reload. This is the largest piece of new code in the project
and it is on the critical path from L1 onward.

## 5. Order of work

1. **ERA5 acquisition** for 27 Sep 18 UTC – 30 Sep 06 UTC over 57–33°W × 11–35°N (a padded box),
   pressure levels plus SST and surface fields. The CDS credential works.
2. **A TC driver** adapted from `run_hindcast.jl`: prescribed ocean only, no land, no terrain,
   1M microphysics, TKE closure, `--nancheck` available, Oceananigans 0.112 bounded WENO.
3. **L0 at 6 km, Float32**, 48 h. The baseline, and the first test of whether Float32 survives a
   deep tropical convective case at all.
4. **Checkpoint and cross-grid restart**, written and tested between L0 and L1 — the enabling
   capability, and worth doing carefully because everything downstream uses it.
5. **L1 at 3 km**, the first resolution at which ENA's Float32 failure appeared.
6. **AIVA** evaluated at L1: its sedimentation-Courant relief is worth more here than at ENA,
   because a TC has heavy rain and the explicit fall-speed bound will pin the step.
7. L2/L3 only once the MPI question in §4.1 is resolved.

## 6. Open questions for Greg

- Vertical grid: ENA used 40 m to 2 km for a stratocumulus deck. A TC needs the full depth and a
  well-resolved inflow layer; I propose ~50 m near the surface stretching to ~500 m aloft with a
  20 km top, but this drives both cost and the Float32 risk and is worth agreeing early.
- Prescribed SST from ERA5 hourly skin temperature, or a fixed climatological field? Hourly ERA5
  gives the storm no cold wake, which matters for a 48 h intensification study. The prognostic
  ocean Greg mentions as a future goal is exactly the fix.
