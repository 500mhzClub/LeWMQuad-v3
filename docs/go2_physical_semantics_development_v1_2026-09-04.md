# Physical-interface development assay: completed bounded result

Date: 4 September 2026. Repository baseline `1f7dd8e`; new development source is
uncommitted and individually SHA-256-bound in each completed run's result.

## Outcome

The independent Genesis primitive-box assay passes all **15 checks**. The final
explicit source/synthetic suite passes **116 tests** (22 new assay tests plus
the preceding 94). These are separate counts: tests exercise acceptance and
failure behavior; the 15 checks apply to actual simulator output.

This is progress on rendering/physics interface qualification, **not** Go2
sensor calibration, gait execution, maze navigation or evidence of JEPA benefit.
No trained model, benchmark data, frozen experiment output or physical robot was
used. Existing frozen sources and results are unchanged.

Implementation:

- [Independent runner](../scripts/run_physical_semantics_development_v1.py).
- [Proper optical-frame conversion and measurements](../lewm/physical_semantics.py).
- [Positive and deliberately failing synthetic cases](../lewm/tests/test_physical_semantics_development.py).
- [Final measured result and source bindings](../.generated/physical_semantics_development_v1_attempt_004/result.json).

The final output directory also contains launch metadata, four PNGs and all
three 401-sample position traces. It is ignored development output, not a
tracked result publication. The conclusions and provenance are retained here.
Final `result.json` SHA-256:
`c504c2af6f66882ad4f336493865d6f54da6a5dbf069ce92c0b7083665400515`.

## Method and fixed criteria

Three fresh, separately destroyed Genesis scenes use the same seed, CPU rigid
physics, 0.002 s step, zero gravity, and 400 steps (0.8 s). A 0.2 m cube starts at
`[0.5,0,0.5]` m with initial velocity `[1,0,0]` m/s. This directly initialized
velocity is a fixture input, **not** a Go2 command or locomotion-controller test.
The cube is invisible to RGB but present in collision physics.

The scenes differ only by absence of a wall, presence of a red wall, or presence
of the same wall with a green render surface. Wall center is `[1.2,0,0.5]` m and
size `[0.2,1,1]` m. Render and physical geometry are instantiated together.
Contacts come from `probe.get_contacts()` after every physics step, not from
analytic intersection labels. Red and green use identical collision materials.

The wall-free expected endpoint is `[1.3,0,0.5]` m. An ideal initial contact is
at 0.5 s: the cube's front face travels 0.5 m to the wall's near face. Criteria
were implemented before the first successful run: free-motion endpoint error
below 0.02 m; contact in steps 225–275; wall-case final x in `(0.95,1.03)` m;
identical color-case contact step and trajectories within absolute `1e-5` m;
and image mean absolute change above 5/255. The endpoint interval is a narrow
fixture check, not a general physical safety limit. It permits rebound and does
not prove the body has come to rest.

A blue calibration cube at `[2,0.4,0.7]` m is measured from its rendered pixels
in the clear scene. Camera position is `[0,0,0.5]` m, native resolution 640×480,
with declared horizontal FOV 78.323°. The runner uses the existing
`genesis_vertical_fov_deg` adapter. An independent pinhole calculation uses a
proper RDF optical frame; marker centroid error must remain below 3 pixels at
camera yaw 0° and +10°. The renderer does not supply the expected centroid.

Finally, setting the probe back to its initial position with zero velocity must
leave position error and speed below `1e-5` after one physics step. This checks
that explicit simulator reset operation, not a commanded stop or exact Go2
snapshot restoration.

## Measured result

| Measurement | Result |
|---|---|
| Clear-scene final x | 1.2999897003 m; no contact |
| Red/green first contact | Step 252 = 0.504 s in both |
| Red/green final x | 0.9621547461 m in both; rebound is allowed |
| Appearance-only dynamics comparison | Identical reported trajectories and contact step |
| Zero-yaw projection error | 0.1845036815 pixels |
| +10° yaw projection error | 0.0400708200 pixels |
| Reset position error / speed | Zero in all three cases |
| Final status | 15/15 checks pass |

The red-wall image was also visually inspected: the wall is clearly present.
The final repetition's complete case payloads, including pixel hashes and
position traces, equal those of the preceding successful attempt. This checks
repeatability of this small fixture only, not across machines or robot states.

## Iteration history and infrastructure

Every attempt uses a fresh output directory; none was overwritten or deleted.

1. `attempt_001`: native EGL initialization crashed with process exit 139 after
   warning that software rendering was being forced on an explicitly selected
   hardware device. No scientific result was produced. A separately labeled
   launcher-failure record preserves the observed exit and stderr.
2. `attempt_002`: the installed OSMesa backend created a context but hit an
   assertion inside Genesis's offscreen-render path. The runner recorded
   `INFRASTRUCTURE_FAILURE` and preserved launch metadata.
3. `attempt_003`: enumeration identified device 2 as advertising
   `EGL_MESA_device_software`; selecting it explicitly completed all checks.
4. `attempt_004`: repeated successfully after renaming the misleading
   `stops_before_wall` check to `bounded_wall_endpoint`. The predicate, numeric
   tolerances, geometry and physics did not change. No failed scientific gate
   was relaxed. Launch metadata and per-case traces are now persisted before
   later stages, so subsequent native crashes cannot masquerade as completion.

Runtime: Genesis 0.4.6, NumPy 2.4.6, Torch 2.12.0+rocm7.2, PyOpenGL 3.1.10.
Physics was explicitly CPU; the successful renderer used the enumerated Mesa
software EGL device. No package or system-library changes were made.

To reproduce, use a **new, nonexistent** output directory. Device indices are
machine-specific: verify the software-device identity before reusing index 2.

```sh
ulimit -c 0
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1 \
PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=2 \
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
timeout 240 .generated/venvs/genesis_rocm_0_4_6_v1/bin/python \
  scripts/run_physical_semantics_development_v1.py \
  --output-dir .generated/physical_semantics_development_v1_fresh_reproduction
```

The runner refuses an existing output directory. A timeout or native crash may
leave only launch metadata or partial traces; missing `result.json` is incomplete,
never a pass. Ordinary exceptions produce an infrastructure-failure result.

## What remains, and the concrete handoff blocker

This assay establishes primitive-scene render/contact consistency and verifies
the horizontal-to-vertical FOV conversion at two poses. It does not run the
active Go2 scene builder end to end, verify textured-mesh collision equivalence,
test safety-retracted camera poses, or validate actual IMU/joint timing,
command-to-body delay, gait stopping, or consecutive edge transitions. Those
remain explicit qualification tasks; do not promote this result to cover them.

The existing stratified handoff successor's
[`require_runtime_source_freeze`](../scripts/run_physical_handoff_stratified_generator_successor_v1.py)
requires HEAD to be its exact freeze commit **and every tracked/untracked
worktree entry to be clean**. HEAD has the required freeze subject, but this
turn's and the preceding turn's development files are untracked. Therefore the
official runner cannot execute as-is. No ignore rule, guard, frozen source,
checkout or benchmark custody rule was changed to evade this condition.

The next operator decision is how to retain these new files while supplying
the existing runner its required clean frozen worktree. One possible narrowly
scoped approach is temporarily relocating only the explicitly named new
development files to an ignored staging location, then restoring them after
execution; that relocation has **not** been performed. Whole-tree exports and
unauthorized access to protected material remain prohibited.

After the frozen run can legitimately proceed, preserve its generator stopping
rules. An inadequate teacher-qualified panel means model performance remains
unmeasured; an adequate panel allows separating candidate feasibility, target
semantics and observation-domain transfer. Hardware sensor calibration requires
the actual platform and a separately agreed safe operating procedure.
