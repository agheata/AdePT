<!--
SPDX-FileCopyrightText: 2026 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Geometry validation: region handoff and GPU boundary crossings

When only some regions are transported on the GPU (see
{ref}`runtime-parameters-specify-the-regions-where-the-gpu-is-used`), a track
leaving them is returned to Geant4 with a touchable built from the VecGeom state
reached on the GPU. Geant4 relocates it from that path with its final direction.
This is correct only if the end point lies within Geant4's tolerance band of the
crossed surfaces and the direction agrees with the crossing; otherwise Geant4
puts the track back in the volume it left, and the track bounces between CPU and
GPU.

AdePT provides two optional checks: one for this handoff, one for every boundary
crossing made on the GPU. They only observe: findings are counted and logged,
the job is never stopped, and the simulation results are unchanged.

## Enabling the checks

The checks are compiled only with `-DADEPT_GEOMETRY_VALIDATION=ON` (default
`OFF`, which leaves the transport code unchanged; the commands then only print a
notice). They are enabled at run time with:

```console
/adept/validateHandoff true
/adept/validateCrossings true
/adept/geometryValidationOutput <prefix>          # optional: write the records
/adept/geometryValidationRecordCapacity 10000     # optional: records kept per check
```

With the option `ON` and no check enabled there is no measurable cost; with both
checks the run-time ratio without/with is 0.96 (CMS 2018, `EcalRegion` on the GPU).

## Handoff to Geant4

For every track leaving the GPU regions, a separate navigator repeats the
relocation Geant4 does when tracking resumes, and checks:

- *tolerance band*: the end point is not strictly inside a volume being left nor
  strictly outside a volume being entered, and lies on a crossed surface;
- *direction*: the final direction against the normal of the crossed surface;
- *relocation*: Geant4 confirms the GPU path (a), or locates the track elsewhere
  for a geometric reason: the direction points back through the crossed surface
  (b), another volume starts on that surface (c), or the handoff push crosses a
  further surface. Anything else is *unexplained* (d).

```{figure} images/geometry_validation_handoff.png
:name: fig-validation-handoff
:alt: Classification of the tracks leaving the GPU regions.
:align: center
:width: 95%

(a) confirmed path; (b) direction pointing back and (c) a further volume
starting on the crossed surface, both legitimate; (d) errors.
```

A band error or an unexplained relocation is excused only when it comes from the
geometry: the end point, or a point up to 100 `kTolerance` along the direction,
is strictly inside two volumes of different branches (overlap) or inside a
volume and outside its ancestor (extrusion). The job ends with
`HandoffValidation: PASSED` when every error is excused this way, `FAILED`
otherwise, with the first cases printed.

## Boundary crossings on the GPU

Every crossing made by the GPU transport (solid navigation only, not
`ADEPT_USE_SURF`) is checked after the relocation, with the VecGeom solids of
the volumes left and entered, against a +-`kTolerance` band. The report gives:

- the depth of the landings on the wrong side, as a histogram in `kTolerance`
  units, and the *off-band* landings beyond `kTolerance`;
- the *bounce-backs*, where the final direction points back through the crossed
  surface;
- the crossings made in *zero-step loops* and the steps ended by the stuck-track
  push;
- the *state mismatches*: at the start of a step, or at the end of a step without
  crossing, the point lies more than `kTolerance` outside the volume of its
  navigation state or inside one of its daughters, i.e. the track changed volume
  without a relocation.

```{figure} images/geometry_validation_crossings.png
:name: fig-validation-crossings
:alt: Landing classes of the GPU boundary crossings.
:align: center
:width: 95%

Landing classes against the +-`kTolerance` band, and bounce-backs.
```

Overlaps are proven as for the handoff. This report is a measurement: it does
not pass or fail.

## Records, plots and replay

With `/adept/geometryValidationOutput <prefix>` the job writes
`<prefix>_crossings.json` and `<prefix>_handoff.json`: the counters, the
histograms and a replayable record of each off-band landing, zero-step-loop
crossing, state mismatch and handoff not confirmed by Geant4. The scripts are in
`test/regression/scripts/python_scripts`:

```console
plot_geometry_validation.py --prefix <prefix> --output-dir plots
replay_geometry_validation.py --prefix <prefix> --summary
replay_geometry_validation.py --replay BuildProducts/bin/geometryValidationReplay \
  --gdml <file.gdml> --prefix <prefix> --index <i> [--boolean-threshold n]
```

The plot script draws the histograms. `--summary` groups the records by kind
and volumes, largest group first, with record indices to replay.
`geometryValidationReplay` (built with example1) repeats a record on the CPU,
with the same GDML file and the same VecGeom: it classifies the end point with
every crossed solid, recomputes the step and the relocation, and, for a
handoff, repeats the Geant4 relocation. `--boolean-threshold` must repeat the
VecGeom Boolean threshold of the run, if it set one.

## Interpreting the records

A group of records with the same kind and volumes usually has one cause, and
replaying one or two of its records is enough. Before any replay:

- an off-band group with an overlap proof is a geometry error;
- an off-band group whose tracks all come after a stuck-track push is a
  consequence of the zero-step loop before the push;
- a zero-step-loop group with one or two records per track and no stuck-track
  push is a track crossing touching volumes, not an error;
- every other off-band landing and every state mismatch is worth a replay.

| What the replay shows | Cause |
| --- | --- |
| The recomputed step reproduces the recorded end point, and a solid left or entered classifies it beyond its band | That solid's distance lands outside its own surface band: a solid error. |
| `pushed>`: the volume entered classifies the pushed pre-step point as outside, or gives a `DistanceToIn` beyond the band, although the step entered it | The solid answers wrongly from a point within its surface band. |
| The recomputed end point differs from the recorded one by more than `kTolerance` (straight steps) | The GPU and the CPU disagree: device code, navigation state, or a different Boolean threshold. |
| `relocation ... DIFFERENT from the recorded state`, with an overlap proof | A geometry overlap or extrusion. |
| `relocation ... DIFFERENT from the recorded state` after a zero step, no overlap proof | Touching volumes: the relocation goes one level further. |
| Zero-step loop: the volume entered is `Inside surface` with a large `DistanceToOut`, and the relocation is the recorded state | A legitimate zero-length crossing between touching volumes. |
| State mismatch with a displacement larger than the pre-step safety | The multiple-scattering displacement left the safety sphere: the safety was too large, or the displacement was not limited by it. |
| State mismatch with a displacement within the pre-step safety | The straight step missed a boundary: replay the step to find the solid. |
| State mismatch at a step start | The previous step left the track in the wrong state: replay the earlier records of the track. |

For a solid error, the replay prints the solid parameters and the local point
and direction with 17 digits: enough for a VecGeom unit test against a
`long double` reference.

## Example 1 tests

With `ADEPT_GEOMETRY_VALIDATION=ON` and examples built, the `example1-handoff`
test runs the CMS 2018 geometry with only `EcalRegion` on the GPU, a 3.8 T field
and both checks, and passes on `HandoffValidation: PASSED`. More geometries
defining their own regions are added at configuration time, one test each:

```console
-DADEPT_HANDOFF_VALIDATION_GEOMETRIES="<file.gdml>=<Region>[,<Region>...];..."
-DADEPT_HANDOFF_VALIDATION_CUDA_STACK=<bytes>     # default 8192
```
