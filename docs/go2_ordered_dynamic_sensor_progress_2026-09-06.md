# Moving sensor pilot: implemented, native query failure preserved and corrected

The full scientific objective remains active: independent-layout RGB-plus-sensor
JEPA prediction and useful rollout/memory for reliable novel-maze Go2 navigation,
with deployment-valid sensing, real-time execution and hardware evidence. This
stage tests moving sensor repeatability, not navigation or learned-model utility.

## Original bounded pilot

Implemented a new union-scene initializer, mounted capture wrapper, explicit
eight-run inventory mapping/artifact roster, raw prechecks, complete-stream
witnesses and terminal auditor. The fixed population includes two repeats of
junction/recent-forward/nominal actions 0 and 3, and near-wall/recent-forward/
lower-friction actions 0 and 1. All original inventory roles and specifications
remain unchanged. Strict depth scores and independent pixel-footprint diagnostics
are recorded separately, without training eligibility or pixel repair.

Focused tests passed 83 cases in 10.07s (18461). The full 225-file regression
passed 2,886 tests in 220.10s (51601). Preflight 48066 passed with 738 source
bindings. An initial test caught a stray empty copied function stub before
preflight/native execution; it was removed before those final passes.

The original collector 2886 is terminal exit 1. Its first precision readback
raised OpenGL invalid enumerant before any episode commit. Partial artifacts
remain, including the first attempted scene's observations; they are not audited
complete episodes. The frozen terminal auditor 84275 completes exit 0 and reports
one attempted/uncommitted run, seven unattempted, zero committed/audited runs and
four unavailable comparisons in each comparison family. No restart occurred.

Original external root child:
`go2_ordered_union_dynamic_sensor_pilot_v1_attempt_001`.
Launch SHA-256:
`ac2542dfab74b5c4df0f7ad4797999e400c0a6d58b2e11d52db87e777e16d626`.
Failure SHA-256:
`ace19931d193ad78054090b0f580e029a4005bb9481fb64ed6bca3961c9dce96`.
Terminal audit SHA-256:
`404a10fdcb7dfdb16553c6f5a4f4e4efa812af3edf706334a014e5457041fd94`.

## Native failure mechanism established

The [core-query probe](go2_core_raster_precision_native_probe_v1_2026-09-06.md)
completed exit 0 (79624), using the same native renderer's framebuffer allocation
in software EGL, with zero scenes, render calls or physics steps. On Mesa4.5
Core Profile/llvmpipe it reproduces GL_INVALID_ENUM specifically for
glGetIntegerv(GL_DEPTH_BITS). The corrected framebuffer attachment query succeeds:
8 subpixel bits, 24 depth bits and four RGB sample positions
(0.375,0.125), (0.875,0.375), (0.125,0.625), (0.625,0.875).
The depth framebuffer is restored and GL error state is clean.

The [Khronos attachment-query reference](https://wikis.khronos.org/opengl/GLAPI/glGetFramebufferAttachmentParameter)
supports querying attachment depth-component bit counts this way. The native
probe, not just that documentation or a mocked test, establishes the local fix.
These bit counts do not alone prove a world-space error bound near silhouettes.

Probe launch SHA-256:
`8f30e3f9e4e9db5fb3025b1ce8753c5438454b895578f0f52739c5e49f93a813`.
Probe result SHA-256:
`fc2d51f3011294573247cfb1782f9c0631dca8a1daa1a8ae390af2e70db60819`.

## Distinct corrected pilot

The [new protocol](go2_core_ordered_union_dynamic_sensor_pilot_v1_2026-09-06.md)
preserves the same eight cases, scene/gait/sensors, commands, scoring, stops and
budgets. Only the precision metadata query changes. New collector/capture/auditor
paths and an exclusive output prevent overwriting or resuming the failed run.
The collector body is AST-identical except its session class symbol; terminal
audit computations are AST-identical. The original raw capture, initializer and
motion/sampling guards remain inherited unchanged.

Combined corrected-pilot, core-query and predecessor-pilot tests pass 26 cases
in 1.85s (80088). The corrected preflight passes with 747 source bindings and
eight exact definitions (99208). The full 227-file regression passes 2,896 tests
in 223.21s (86569, exit 0). Corrected collection 51519 is launched under the frozen
747-source definition and subsequently completes all eight runs (51519, exit 0).

## Completed native results and terminal audit

The frozen terminal auditor completes all eight runs (37417, exit 0), including
raw reconstruction and equality with saved prechecks. Total: 16,436 physics
samples, 214 RGB-D frames, eight valid setups and eight action departures.
Collection commits plus launch/precheck metadata occupy 273,057,887 bytes.
No acquisition failure, model fit, hardware action or navigation promotion occurred.

All four across-action full native/contact/sensor prefixes match exactly. All
four pairs of repeated recorded streams are bit-identical, including native
depth. Under the frozen classification, the two junction pairs are
EXACT_COMPLETE_REPLAY; the two near-wall pairs are EXACT_PARTIAL_REPLAY because
native contact stops their command schedules. Therefore the protocol's four-
complete-repeat criterion is **not met**. Matching streams through contact are
useful evidence, but are not relabelled as full command completion.

Both near-wall conditions reach measured base-to-wall contact during the zero-
command braking phase. Short-pulse contact occurs at 3.838s (1.538s after action
departure); long-pulse contact at 3.298s (0.998s after departure). Each occurs at
the same native sample in its repeat. These four episodes supply 12 positive
cumulative-horizon labels: two for each short-pulse run and four for each long-
pulse run. They represent four recorded contact events in two repeated conditions,
not 12 independent collisions or evidence across independent layouts. Motion and
future-image targets remain censored after termination; no trajectory is filled in.

Both junction-action-3 repeats retain their original strict frame-11 visibility
failure. Across all 214 frames the separately reported stable-interior metric
passes and there are no near-occlusion failures; the two large residuals are in
the boundary-ambiguous sets. This remains explanatory coverage accounting, not
boundary-pixel certification or permission to repair pixels from geometry.

Read-only verification 50064 independently confirms the first corrected run's
initial 750 native/contact samples and first RGB/native-depth arrays are exactly
equal to the failed predecessor's preserved pre-error evidence. The correction
did not alter the initial physical/sensor trajectory. That comparison neither
completes nor rescues the predecessor episode. Read-only check 68037 verifies all
saved precheck bindings and reproduces the four prefix matches, two complete/two
partial repeat classifications and per-frame footprint/contact summaries.

Corrected launch SHA-256:
`8e6162bcd8bfb467a54185489f8d7e1eb770f102a15d8fece1a2f413d81226a5`.
Collection result SHA-256:
`8cf589de36e00b65288bbf0abe322c65ff40d14b286acd7db7d0c9d2f923d6af`.
Terminal audit SHA-256:
`10d72ad0817a4a807f018ebfb5fbc5974100b38848346892c1feae977c4d9c5b`.

## Next scientific decisions

The [next acquisition/measurement plan](go2_audited_terminal_event_collection_next_steps_2026-09-06.md)
must separate three questions: faithful observation through a real terminal event,
uninterrupted command completion, and deployable navigation success. Requiring
all command schedules to finish is incompatible with deliberately collecting
collision-ended examples. The failed frozen criterion is preserved; a future
contract must count terminal-event evidence honestly without calling it navigation
success, and must retain missing/invalid trajectories and sensor uncertainty.

The other prerequisite is a prospectively defined finite-pixel measurement and
policy-side uncertainty contract. The current pilot grants no training eligibility.
After these are resolved, acquire the predetermined independent train/selection/
development-evaluation layouts and run the matched prediction study, followed by
rollout and memory studies with dependable physical execution. Latest learned
heads still lose the action/time baseline; reliable room return remains unachieved.
