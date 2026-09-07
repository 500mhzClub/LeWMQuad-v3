# Contact measurement integrity correction

5 September 2026 (UK). Frozen source baseline: `1f7dd8e`.

## Confirmed defect and scope

The inherited `_GenesisPhysicalSession._disallowed_contact` in
`scripts/run_physical_graph_edge_handoff_qualification_v1.py` flattens link IDs
but indexes the unflattened force array. The active builder calls
`scene.build(n_envs=1)`. Installed Genesis 0.4.6 returns contact arrays shaped
`[environment, contact, ...]` whenever `solver.n_envs != 0`; the runner's
`_as_np` preserves that shape.

For force shape `[1,N,3]`, the first contact's force norm can include every
contact, and later contacts can receive `force_magnitude_n=None`. Consequently,
a zero-force body/wall geometric contact plus an allowed nonzero foot/floor
contact is incorrectly marked disallowed. Both contact orderings are reproduced
by the actual frozen method in
`lewm/tests/test_frozen_contact_force_batch_diagnostic.py`. Identical unbatched
inputs classify correctly. Validity masks also require explicit handling.

This is a measurement defect, not evidence that the JEPA or teacher controller
failed. The previously reported first-20 trace summary (19 flags; median flag
onset 0.422 seconds) remains an accurate summary of persisted booleans, **not
verified collision incidence**. Raw per-contact forces/link identities were not
persisted, so those labels cannot simply be corrected retrospectively. Earlier
physical lineages sharing this detector need a scope-specific audit; do not
silently promote their reported counts to independently verified measurements.

## Interruption and custody

The owned supervisor was interrupted at 23:15:42 UTC on 4 September, after
16 streams completed and while `TURNING_JUNCTION-00` was active. Its exec session
96850 exited 130, stopped its child, and restored all ten staged development
files. A separate hash check confirmed all ten restorations. Both processes
were verified absent. No material, logs, partial candidates or tracked source
were deleted or overwritten. No generator reduction, panel, ranker, or held-out
evaluation was run. This attempt is interrupted for integrity, not a valid
negative generator terminal, and must not be resumed silently.

The operator reason and events are retained under
`.generated/navigation-development-staging.m6MDz1/`. The external qualification
roots remain untouched. The full scientific goal is not achieved.

## Development correction

`lewm/safety/contact_attribution.py` explicitly selects a batched environment,
requires and applies validity masks, preserves per-contact force vectors and
link/object identity, and uses the existing contact ontology. Missing force is
reported as unavailable; a conservative label with missing force is not
presented as measured collision evidence. Invalid rows are ignored, malformed
valid evidence is rejected, and object identities are supplied rather than
guessed. The frozen detector is unchanged.

The causal sensor-history and directed traversal modules prepared during the
run have also been integrated as development components. They are not wired
into the frozen experiments or claimed to improve navigation. The explicit
combined source/synthetic suite passes 194 tests, including five exploratory
trace-analysis tests retained in the staging directory.

## Fresh native contact-packet assay: specified before execution

Purpose: verify the corrected adapter against native one-environment Genesis
packets, including measured ground support and wall contact. This is an
independent primitive diagnostic, not Go2, controller or maze qualification.
No prior candidate, checkpoint, image, dataset or model is loaded.

Use Genesis CPU, seed 20260905, `n_envs=1`, dt 0.002 seconds, and exactly
100 steps per case, with no renderer or camera:

1. `ground_support`: a 0.2 m cube at (0,0,0.1), gravity (0,0,-9.81), plane z=0.
2. `wall_impact`: cube at (0,0,0.5), gravity zero, initial x velocity 1 m/s;
   fixed wall centred at (0.3,0,0.5), size (0.2,1,1); plane retained at z=0.
3. `mixed_touch`: cube at (0,0,0.1), gravity (0,0,-9.81), initially stationary;
   fixed wall centred at (0.2,0,0.5), size (0.2,1,1), touching the cube's x face.

The cube's sole link is declared a support link for this diagnostic only.
Persist native public contact fields, per-contact attribution and old/new
boolean comparisons at every step. Fix output identity and source hashes before
stepping; refuse an existing output directory and preserve failures.

Acceptance: native arrays retain an explicit one-environment axis and boolean
validity masks; attributed force equals the corresponding native per-contact
force norm to absolute tolerance 1e-9 N; ground-support case has measured
nonzero support contact and no disallowed contact; wall-impact case has measured
nonzero wall contact classified disallowed. Every valid force/position must be
finite. The mixed-touch case is diagnostic only: whether it produces an actual
old/new disagreement is reported without moving geometry or retuning until a
desired result appears. Passing does not repair old material or authorize a
silent restart of the interrupted attempt.

After this assay, specify a fresh, bounded Go2 contact-attributed execution
experiment, retaining requested/applied commands, measured motion, first-contact
link/object and all failures. Only then revisit teacher feasibility and action
selection. Hold JEPA comparisons until local execution has trustworthy endpoints.

### Assay attempt 001 and reference-calculation correction

Attempt 001 completed all three cases and retained all raw packets, but returned
`CHECK_FAILURE`: its comparison used a float32 native-vector norm against the
adapter's float64 norm at a 1e-9 N tolerance. The scalar reference is corrected to
`math.hypot` over the unchanged native components converted individually to
Python float. A focused counterexample reproduces the float32 rounding error.
No geometry, dynamics, seed, force floor, acceptance tolerance, or adapter
behavior is changed. A fresh attempt 002 will repeat the same cases; attempt
001 and its failed report remain intact.

The native wall-impact packets already show ten steps (60–69) where the old
detector is true and the corrected detector is false. For example, at steps
60–61 the API reports five and four wall contact points respectively, all with zero force magnitude;
the old batched loop still declares disallowed contact. This confirms a native
measurement discrepancy, not merely a synthetic possibility. It still does
not reveal which historical Go2 flags were affected.

### Native assay result

Attempt 002 completed and passed all six predefined native checks. Ground
support produced 396 nonzero contact-point samples and no disallowed labels;
wall impact produced 35 nonzero wall contact-point samples, all classified
disallowed. These are contact-point samples, not independent trials. The same
ten old/new disagreement steps were reproduced. The three native-packet files
and three attribution files are byte-identical between attempts 001 and 002:
only the reference calculation and its report changed, not measured physics.

Results remain under `.generated/contact_packet_semantics_development_v1_attempt_001/`
and `..._attempt_002/`. Their result-file SHA-256 values are, respectively,
`28bbb36bd096d9b8dd63bb217df380c8890073a1060f3d293c6bf3ffba76b9e5` and
`930b2ef8c919e7d655189f4b194b1a6071e68d73a0c504391932555aee88b68c`.
Neither result repairs the interrupted Go2 evidence or proves maze navigation.
The next work is fresh contact-attributed Go2 execution, not continuation of
the old stream.
