# Consistent candidate selection for the existing gyro-height fallback

Close-wall failures can leave a narrow floor patch that cannot estimate a
full normal. The existing partial-height observer uses the public gyro prior
for that case. However, its raw candidates were pruned against a separately
fitted normal; some of those candidates then exceed the original 3-mm
residual limit against the gyro normal. This is a mismatch between candidate
selection and its consumer, not a reason to enlarge the residual threshold.

The isolated candidate prototype first runs the original selection and full
plane fit. It preserves fully accepted planes and every rejection except
missing two-axis extent. For that case only, it removes original candidates
whose residual to the existing gyro normal exceeds 3 mm, recomputing the
height offset within the original eight-step bound. It still requires at
least 100 points and at least one quarter of the original pool. It never
adds a point or fills a missing ray. It supplies a scalar-height hypothesis,
not a new measured full normal or established floor identity. If pruning
cannot meet these conditions, it retains the original rejected candidate set.

Source: `lewm/gyro_conditioned_partial_floor_candidates_development.py`.
Four tests passed: fully observable planes unchanged, already-consistent weak
patches unchanged, bounded gyro-residual pruning, and missing support unchanged.

## Exact recorded-frame probes

`scripts/probe_go2_gyro_conditioned_partial_floor_development.py` reconstructs
the delivered noisy depth and compares both selectors under the recorded
current up direction. All 13 original outcomes match their saved receipts
exactly. No native pose is loaded. Four unavailable patches become eligible
for the existing scalar-height rule:

| Recording/frame | Candidates before/after | Maximum gyro residual before/after |
| --- | ---: | ---: |
| Original supervised 1402, 567 | 2442 / 2440 | 3.007 / 2.996 mm |
| Original supervised 1402, 806 | 2615 / 2610 | 3.022 / 2.996 mm |
| Reactive auxiliary-turn follow-up, 4631 | 2176 / 2170 | 3.048 / 2.990 mm |
| Reactive auxiliary-turn follow-up, 4804 | 2008 / 1993 | 3.103 / 2.998 mm |

The other nine outcomes remain unchanged, including all four probed frames
of the original reactive failure. No previously available patch was lost.
Each of the three failure roots contains `gyro_conditioned_partial_floor_probe_v1/result.json`.
These probes do not establish registration recovery or physical navigation.

## Sequential replay complete: obstacle-only gain cannot restore execution

`lewm/gyro_conditioned_partial_floor_consumers_development.py` supplies the
selector to isolated registration and auxiliary-turn obstacle consumers.
The original floor-reacquisition behavior, original raw tracker and all
downstream acceptance thresholds remain. No native launcher uses these classes.

An 820-frame sequential replay of the original supervised-1402 failure
completed in session 91366 after 110.80 s, with no failure. It compared original and new registration/obstacle
consumers on one unchanged raw-tracker stream, verifies old raw/registered
poses against the recording, and validates new available pose evidence through
the existing reader. Both obstacle consumers permit the existing auxiliary-only
turn evidence, so their comparison isolates the candidate change.

Script: `scripts/replay_go2_gyro_conditioned_partial_floor_consumers_development.py`.
Output within that failed root: `gyro_conditioned_partial_floor_consumers_replay_820_v1/`.
It recovered obstacle observations on 224 frames and lost none, but changed
no registration availability and no shared registered pose. Every additional
obstacle frame still lacked an accepted registered pose; none supplied the
required four-frame pose streak. `effective_availability_diagnostic_v1.json`
records this result. This version cannot restore execution on those frames,
so no native navigation test of it is warranted yet.

The registration consumer checks residuals against the transported initial
floor-reference normal, which differs from the gravity-derived `up_body` used
for candidate selection. Its exact formula is already in
`extended_return_budget_transport_development.composition`: form the anchor's
registered/raw rotation correction, apply it to the current raw rotation, and
transport the accepted reference-plane normal into the current body frame.
The next hypothesis is to use that exact existing normal for weak-extent
candidate pruning, while preserving the original up direction for pool
selection and plane metadata. The existing height-composition and residual
validators must still validate the resulting evidence. No such registration
change was implemented during that initial replay; all original attempts and
that negative replay remain preserved. The subsequent transported-normal
implementation restored all 820 registered poses in exact sequential replay
and is now in a fixed native test. See
`docs/go2_transport_conditioned_floor_recovery_2026-09-15.md`.
