# RGB/body observation evidence and next scientific decisions

The fitted exploration/local-controller bridge, visible-floor baseline and
body-sensor ground hypothesis have now been exercised on actual recorded inputs.
None is a demonstrated RGB place/exit/beacon detector or full-maze navigator.
The combined development suite passes655 tests across52 explicit files
(session17752, exit0,8.24 s). All earlier physical/model results remain intact.

## Fitted bridge replay: exact decision path verified

The [fixed integration replay](go2_exploration_bridge_replay_development_v1_2026-09-05.md)
passed all144 recorded streams and1,114 choices using the original fitted
three-seed ensembles. Current input hashes, RGB provenance, model/member
predictions, selected actions and command tapes match exactly. Reconstructing
the original directional cue from a symbolic bearing changes direction/cost
values by at most8.88e-16, below the fixed1e-12 roundoff allowance.

The replay intentionally leaves zero qualified traversals and one outstanding
symbolic attempt per stream: the old directional task supplies no qualified
place arrival. Supplied replay bearings are not detected exits. This validates
the inference connection without fabricating exploration evidence.

Session11013 terminated exit0. Root:
`.generated/go2_exploration_bridge_replay_development_v1_attempt_001`.
Result SHA `87c558e98e1990b52c37bd1854b4f6b2d14ac692322e22581e232494316f4383`.
Launch SHA `03a2b06e0639599a2e60e84a1852b8edb36059f7b224593ad01f74f0704222d7`.
The memory, bridge, new comparator, protocols and tests bound by this replay
must now remain unchanged; later improvements need separately identified source.

## RGB floor evidence: strong current-palette shortcut, no transfer evidence

Source inspection found that the renderer's plane uses material key `floor`,
whose default palette is(.45,.47,.42), while the old neutral-floor override is
named `NEUTRAL_FLOOR`. The wall material is neutral. A viewed route image indeed
has greenish floor against neutral walls. This makes a simple chroma baseline
possible; it is not evidence of learned scene understanding.

The [fixed RGB-only baseline](go2_rgb_floor_evidence_development_v1_2026-09-05.md)
was evaluated on818 unique current frames (546 train/272 development-validation)
from the existing24 layouts. Privileged camera/box geometry is used only in a
separate visible-surface ray labeler; the runtime receives only causal RGB/body
packets and emits floor pixels and a bottom-connected image envelope.

| Original RGB, layout-macro average | Train16 layouts | Development-validation8 layouts |
|---|---:|---:|
| All-valid sampled-pixel precision | 99.9004% | 99.8874% |
| All-valid sampled-pixel recall | 99.9911% | 99.9907% |
| Fixed interior precision | 99.9998% | 100% |
| Fixed interior recall | 100% | 100% |

Across all valid sampled rays there are1,047,952 true positives,1,037 false
positives,98 false negatives and2,851,810 true negatives. Of3,926,400 planned
rays,25,503 are explicitly excluded for near-clip/wall ambiguity. The separately
reported fixed interior population has one false positive and zero false
negatives; do not substitute that easier population for the all-valid result.
Both grayscale and red/blue-swap controls produce zero positives and zero recall
on every layout; precision is undefined, not100%. No threshold was refitted.

These results show palette dependence and agreement with simple flat-scene
visibility, not robustness, metric clearance or exit identity. Floor visible
through a narrow gap need not admit the quadruped; an obstacle above the floor,
the camera's blind near field or an unobserved direction remains relevant.
Negative pixels are unknown, not a reliable wall label. No runtime map was read.

Session63372 terminated exit0. Root:
`.generated/go2_rgb_floor_evidence_development_v1_attempt_001`.
Result SHA `bf822b9eecbf711a634330dc4fa27ce33910e39f8860e8691d9988f6a2cc1a74`.
Launch SHA `8e1001d73d8ec141280fc79e7786963c63c8c01bf15010624d61f92fee46974c`.
Separate evaluation-label SHA:
`fad18a7a1403e69986b701123e9ff09c14ee1915c0f133987e8ed09ab9a646b4`.
No model was fitted. Source/protocol/test bytes are frozen by this result.

## Ground hypothesis: usable body information, uncalibrated distance errors

The [fixed body-sensor replay](go2_ground_plane_development_v1_2026-09-05.md)
uses initial zero-command specific force to hypothesize gravity, gyro history to
propagate it, and current joint angles plus exact Go2 geometry to hypothesize
height from the lowest foot sphere. No world pose or actual velocity enters the
runtime. Independent URDF chain composition agrees with closed-form foot centres
to1.67e-16 m at all2,852 packets. All26 streams initialized and replayed without
sensor rejection. Initial gravity and flat-contact assumptions remain assumptions.

| Evaluation | Eight continuous routes | Eighteen turns |
|---|---:|---:|
| Packets | 1,498 | 1,354 |
| Body-height mean absolute error | 0.562 cm | 0.244 cm |
| Worst body-height error | 2.541 cm | 0.509 cm |
| Mean up-direction error | 0.02405 rad | 0.01022 rad |
| Worst up-direction error | 0.08523 rad | 0.02305 rad |
| Worst camera-height error | 5.124 cm | 0.557 cm |

The camera lever arm makes attitude error affect camera height. Near-horizon
pixel-to-plane distance can amplify these errors, so low mean body-height error
is not a clearance certificate. Measure actual projected-point errors before
using this hypothesis to place an exit or merge places. The runtime explicitly
keeps `ground_plane_qualified: false`. Flight, uneven support, sensor bias and
hardware timing/calibration have not been tested.

Session14179 terminated exit0. Root:
`.generated/go2_ground_plane_development_v1_attempt_001`.
Result SHA `b798b0d9afd7764f37cbcdcafb14400e870138b8e13144d34c80227e0f1a1fa7`.
Launch SHA `aa47f0d078474668279a63b6a62ec93b5da27c788c5272962d6457de2c57ecc1`.
The exact installed URDF was resolved read-only into RecoveryStorage and bound
by SHA; no source export/copy occurred. Its two pre-launch path checks happened
before output creation or replay. The final executed source is now frozen.

## Current dataset milestone and next actions

The separate moving-prefix physical collector completed all384 trials, including
103 native-contact trials; all384 moving conditioning prefixes were available
and matched. There are100 contacts within the three-second suffix and three
later release contacts, which must not contaminate suffix labels. Recorded
totals are1,710,516 physics samples,171,005 sensor samples and28,883 RGB packets.
The independent full audit13100 completed exit0 and passed all384 trials; the
actual600-cell tensor check35152 also completed exit0 and passed. The physical result SHA is
`c50e6943739b19a834a623b0d8c40f5ef7f9f368d4d340a49e375f8a60dbe2e0`.
Collection1894 is terminal exit0; do not restart it.

1. Collection, full audit and actual600-cell tensor qualification are complete;
   see the [data result](go2_moving_prefix_counterfactual_development_v1_result_2026-09-05.md).
   Do not rerun them. No new fitting has begun.
2. For the next causal-coverage comparison, avoid silently replacing914 old
   windows with600 selected cells. Retaining all914 plus384 new suffixes gives
   1,298 windows (866 train/432 validation). Match the *conditioning-context
   schedule* between coverage-limited and expanded arms: at the four one-second
   moving contexts per layout, the former has only continuation and the latter
   has five observed actions. Keep current-image exposure, updates, seeds and
   model/loss design matched. The augmented loader/sampler now has13 focused
   passing tests and actual866/432-window metadata preflight. The matched
   training runner, its tests and a fixed launch protocol remain to implement.
3. Turn the RGB/body observations into tested local exit hypotheses: measure
   pixel-to-ground projection error, require footprint/near-field visibility,
   and preserve unknown directions. Evaluate learned semantic observation heads
   and appearance variation separately from the action-coverage intervention.
   Do not adopt the palette cue as the final real-robot solution.
4. Add actual RGB place/beacon association and physically verified arrivals to
   the bridge. Test aliasing, missed/wrong revisits, turns and directed return.
   The existing graph cannot certify supplied perception identities.
5. Evaluate actual exploration/discovery/return with matched memory and JEPA
   ablations, then locked independent layouts/shifts and supervised hardware
   transfer. The final scientific goal remains active and unachieved.
