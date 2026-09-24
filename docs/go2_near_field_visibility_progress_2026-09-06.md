# Near-wall visibility defect confirmed and corrected in a native camera bench

The previous status turn produced actionable evidence: the camera saw through a
wall inside its5cm renderer near plane. This turn reproduces the defect, tests a
distinct5mm-near renderer and integrates physical-first-surface checking into
the independent-layout collector. No JEPA/navigation success follows from this
sensor correction. The full scientific objective remains active and unachieved.

## Completed native comparison

The [fixed camera bench](go2_near_field_visibility_probe_v1_2026-09-06.md) completed
all18RGB/native-depth captures with zero physics steps, using two otherwise
matched scenes and nine fixed wall distances per near-plane arm. Native camera
pose, intrinsics, clip planes, single-sample framebuffer and visual/collision
geometry were checked. Every declared positive and negative outcome occurred.

| Front-wall distance | Legacy5cm near | Corrected5mm near |
| --- | --- | --- |
|4mm|See-through, correctly rejected by new audit|See-through, correctly rejected|
|6mm,20mm,49mm|See-through, rejected|Correct nearest wall, passes|
|51mm,199mm,201mm,1m,4.9m|Passes|Passes|

At20mm, the old native depth is1.219996m; the corrected native depth is0.01999993m.
At4.9m, the corrected maximum full-image error is0.0000711441m (0.0711mm), below
the unchanged1mm gate. Saved measurements were independently compared with the
analytic whole-image planar oracle; all source/artifact bindings were verified.
The deliberately below-near4mm case remains invalid; the correction is not a
claim that clipping is impossible. General-scene audit uses a declared stride8
sample, not a complete-image proof. Robot self-occlusion remains absent.

The new evaluator finds the nearest positive opaque surface without inheriting
the render near cutoff. It rejects cameras in/on solids and on/below the opaque
floor. It reports clipped surfaces instead of treating the background as free.
No analytic geometry enters policy pixels. Public depth remains0.2–5m with
unknown pixels below that range; mount and intrinsic calibration are unchanged.

Bench output is the owned external artifact root
`go2_near_field_visibility_probe_v1_attempt_001`; collector91270 terminal exit0.
Launch SHA256: `429c73911834c7c0243d9b68aa9297848ee248afa7f19ca5965b7c35e88c90e6`.
Result SHA256: `6eba751f612df0ad40d56b26fb4a62923528adfbfb3dd1fa897414641371504b`.
Its699bound sources, including the new reference/builder/probe/tests, are frozen.

## Scope of the defect in the existing pilot

All390recorded pilot frames were examined. Twenty-one frames fail the physical
visibility check: nominal_action_1 indices13through33 inclusive. Each has4,800
sampled clipped-wall rays with falsely public-valid background depth, totaling
100,800rays. The other369frames pass this sampled check. All11other episodes
have no detected visibility failures. This is not permission to silently salvage
the pilot, repair its pixels or relabel it as independent evaluation.

The all-frame audit program54333 reached its terminal metadata-budget check
after writing all12reports and rechecking sources/inputs, then failed: its
inherited launch alone is9,819,679bytes, exceeding the fixed8MiB metadata cap.
The failure is preserved, not converted to a successful run or resumed with a
larger budget. A separate posthoc read-only check68356 exits0: it independently
recomputes all390row values from the bound native recordings, checks every saved
row and all source/input/output identities, and confirms the counts above.
Total persisted audit metadata including the failure is10,019,943bytes.
Future metadata preflights must count serialized inherited bindings before
exclusive output creation; increasing budgets after a failure is not a fix.

Audit output: `go2_independent_pulse_context_physical_visibility_audit_v1_attempt_001`.
Launch SHA256: `092508fa7a0e0ef2a58ad0ccb6e9ff3b75f1642b317a6fb9b5100e06eea39f5f`.
Failure SHA256: `6673a4c0f4da975da41f88a6bd4f8832de2a9c048eca2f0f5707bf0e0ac62ba4`.
Bad-case report SHA256: `189f7287cc6dcde428ed967834b5a1f60d61c21ec51d65a6c8e3a60e3b110383`.

## Collection integration and verification

Distinct capture and raw-audit modules preserve the old frozen implementations.
The previously unlaunched inventory adapter now explicitly binds5mm render near
and the unchanged public range. The native initializer uses the corrected scene
builder; the session uses the distinct rigid-mounted capture method. The raw
audit preserves all command/contact/body/setup/stop checks and adds physical
visibility. A failed visibility report is saved before collection terminates;
the terminal auditor retains physical labels and attempt counts but excludes
that episode from dataset materialization and lists it explicitly.

Seventy-five focused tests pass (4053,9.84s). An initial new test double used
optical instead of native OpenGL camera coordinates; correcting the test double
made it exercise the actual pose check. No native threshold/source was relaxed.
The full explicit219-file regression passes2,795tests in218.02s (63001,exit0).
The earlier217-file regression handle28322 is missing and its terminal output
was unavailable; no unsupported success claim is made for that earlier run.

Read-only l00 preflight71720 exits0:719source bindings,120exact prospective
episodes, corrected camera and the completed bench's exact identities. The first
episode is l00_open_passage_quiet_nominal_a0. The collector was then launched as
63782 with the exact frozen120episode batch,40GiB free reserve,8GiB batch cap and
256MiB per-episode allowance. Keep its launched sources frozen and poll that
same handle, never restart because observation timed out. Live batch state is
tracked in the autonomous checkpoint; this launch is not a completed dataset.

First-episode progress is now independently verified (38356,exit0): all2,250
native samples,31RGB-D frames and30commands recorded; setup/departure/schedule
complete; physical visibility passes with maximum depth error0.000291127m.
Shadow tracking has a pose at31/31decisions, versus the old pilot's largely
unavailable tracking; this is a different scene/history, not a matched tracking
improvement claim. Five motion/future-image targets and zero positive contact
targets are available. Its58,665,673artifact bytes and all source/artifact hashes
were checked. Three episodes have passed raw prechecks and a fourth artifact
commit is observed while the batch remains live; no terminal audit yet.
Batch launch SHA256: `1d46515179b91d7134c83b6247f0cc22db1de786e3a99a1db9bde82ca85fb218`.
First commit SHA256: `d2a0150ca7552f228f5c3a0d75f48eb43cdaa62e04d57350f9b40c6bf1eab25b`.
First precheck SHA256: `e3ec7e15914dafe8fcfc22b6cc19457d611dd7dcd4e47dabb1fb44a210cbb594`.

## Next scientific work

Complete and audit l00, including every failed or absent planned attempt. Inspect
actual setup/common-prefix survival, visibility, contacts and support/history/
action coverage before launching another declared layout. Do not change roles
or select only successful contexts. No further one-room fitting is justified.
Then complete adequate independent data and matched action/history/RGB and
direct/supervised-rollout/JEPA comparisons. Reliable physical local execution,
online planning, memory/backtracking, independent novel-maze outcomes, realistic
sensors/deadlines and bounded hardware evidence remain necessary. Previous
learning still loses the empirical motion baseline; latest room-return remains
0/3. The learned gait and fitted predictors are not a demonstrated learned
maze-navigation policy.
