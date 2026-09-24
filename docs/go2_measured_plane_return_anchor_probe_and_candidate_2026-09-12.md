# Measured-plane return-turn anchor diagnosis and combined candidate

Development progress on 2026-09-12 Europe/London. The previous goal turn made
progress by implementing the independent-study pipeline and locating the new
return failure. This turn completed a recorded-data pair experiment and tested
a separate observer/controller candidate. The overall goal remains incomplete.

## Fixed recorded-data experiment

Executed `scripts/probe_go2_measured_plane_return_anchor_pairs_v1.py`, session
95017, exit 0, 9.34 seconds. Its 18 focused tests passed before execution
(session 11453, 2.17 seconds). Probe source and tests are now bound to this
completed experiment and must not be edited for follow-up experiments.

Artifact root under the existing development artifact volume:
`go2_measured_plane_return_anchor_pair_probe_v1_attempt_001`.

- Launch SHA-256:
  `1abd1409776931187bdc2ce4959d57e3d47146c4350222f71f06f2377bcfc267`.
- Result SHA-256:
  `d3a01cc5b667b7038964adf04ee135179af8a35c7d8bf0ed875b5baae5831c6e`.
- 2,564 source bindings and 95 recorded-input bindings, checked before and
  after execution. No model, observer-history replay or physical scene ran.
- The input is the closed collection of measured-plane dispatch recovery,
  launch `93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb`,
  collection result
  `922cf4a1134eb5458010e12f89ba65fe427b839476837ab035d8ceace4181680`.
  Its final native audit is still pending; this probe does not replace it.

The fixed experiment covered every recorded retained reference
(3100, 3099, 3098, 3096, 3095, 3094, 3093, 3092), both cameras, two targets,
and two existing association methods: 64 pair/method results total. Every
negative result was retained. Twenty-two paired public observations from
3092 through 3113 reconstructed the features and relative gyro consistency
monitor. The chained method was repeated byte-for-byte. No association or
registration threshold was tuned.

The complete stream line count was 3,124. Selected boundary rows confirmed
frame 3102 was anchored, 3103 began the final gap with bridge count one,
3112 reached count ten, and 3113 exhausted the bridge in both cameras.
This is the start of the **final** tracking gap, not necessarily the first
bridge anywhere in the episode.

| Target | Camera | Original descriptor fits | Chained endpoint fits |
| --- | --- | ---: | ---: |
| 3103 | Primary | 0 / 8 | 2 / 8 |
| 3103 | Auxiliary | 0 / 8 | 1 / 8 |
| 3113 | Primary | 0 / 8 | 0 / 8 |
| 3113 | Auxiliary | 0 / 8 | 0 / 8 |

The qualifying frame-3103 chained fits were:

| Reference | Camera | Inliers | Fit RMS | Gyro disagreement |
| --- | --- | ---: | ---: | ---: |
| 3100 | Primary | 19 | 0.336 mm | 0.0000519 rad |
| 3099 | Primary | 17 | 0.540 mm | 0.0002318 rad |
| 3100 | Auxiliary | 37 | 0.672 mm | 0.0003993 rad |

These are endpoint rigid/gyro checks only. The probe did not refine the fits
against measured planes, check global pose envelopes or anchor/increment
conflicts, or admit any pose. It shows that the existing image-chain method
retains potentially useful reference information early in the gap. It does
not prove that an integrated observer will recover, that a changed controller
will select useful commands, or that the robot will return home.

## Separate combined candidate

Implemented `lewm/measured_plane_chained_anchor_development.py`:

- `MeasuredPlaneChainedAnchorPose` composes the existing measured-plane and
  chained-anchor classes. The measured-plane wrapper encloses each image fit
  before temporal admission. Existing chain fallback, original-measurement
  conflict vetoes, reference promotion, ten-frame bridge allowance, and bounded
  image cache remain inherited.
- `MeasuredPlaneChainedAnchorVisualMotion` exposes the existing measured-plane
  evidence plus copied direct-flow/chained fallback receipts when present.
- `MeasuredPlaneChainedAnchorController` replaces only the measured visual
  motion implementation in the existing residual controller. Model, planning,
  memory and mission behavior remain inherited.

The final 13-test suite passed (session 41576, 17.53 seconds). It covers actual
synthetic primary and auxiliary image-chain fits followed by measured-plane
refinement and public pose admission; unchanged healthy-anchor evidence;
original and chained plane/image conflict rejection; preservation of ten
bridges and the terminal eleventh frame when recovery is unavailable; rejected
refinement without publishing an unconstrained anchor; sensor-failure latching;
and complete unchanged image-to-first-forecast controller decisions for no-RGB
direct and full-RGB JEPA fixed-head test models, with unchanged model state.

Initial tests incorrectly expected a bridge at frame one when a retained anchor
was also the immediately previous frame. The existing direct-flow fallback can
validly measure that anchor. The fixtures now check that behavior explicitly,
then exercise all ten bridge intervals after it. Production code was not
changed to force the mistaken expectation.

This candidate is local source preparation, not adopted perception or a frozen
native experiment. It has not processed the full current recorded controller
prefix and has not run in a new scene. The existing independent-study roster
still names the previous measured-plane implementation and is not silently
changed by this candidate.

## Next evidence

Finish and inspect the exact live measured-plane worker audit. Preserve its
negative collection and any distinct audit failure. Keep the already registered
nominal, reactive and full-history timing work tied to their exact owners.
The timing waiter reserves the next full CPU replay slot; this small pair
probe did not occupy that slot.

Prepare an explicit paired controller replay on the completed current-run
inputs, reconstructing the original full sensor-to-command path and the new
candidate from fresh state. Stop at the first changed command; recorded future
observations after that command cannot establish a counterfactual trajectory.
Only that checked boundary can support a new prospective native attempt.
Do not interpret these component and pair results as successful navigation.

## Source identities

| Path | SHA-256 |
| --- | --- |
| `scripts/probe_go2_measured_plane_return_anchor_pairs_v1.py` | `916b46f10dd8c30db7bcf7740bff84becb3d34a22bc6644c66816e94812aa5d0` |
| `lewm/tests/test_measured_plane_return_anchor_pair_probe_development.py` | `1a686994cfa1d3d4dd1479eecd7888feb4a4cff3a620aed8eabaa39f1ee3aeeb` |
| `lewm/measured_plane_chained_anchor_development.py` | `fc9ea2747a27c5d1eecf6e91197623cf3eba89960974e1a326e29812502e2a11` |
| `lewm/tests/test_measured_plane_chained_anchor_development.py` | `f51239caf0d9604c82950c14dc4003be6d70328fab6b6f078ba48f2adcdeafe5` |
