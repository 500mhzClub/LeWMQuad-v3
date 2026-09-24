# Release-aware navigation V1: audited result and changed next priority

Collection43946 completed2/2 development missions, exit0. Full audit87286 passed
both, replaying853 controller decisions and864 RGB/depth/relative-state records.
Prelaunch test35979 passed1,135 across103 files in77.31 s. Preflight76934 and
terminal audit checked298 source,204 input,2 gait and10 native bindings; the
post-audit checksum check passed. All launched source/results remain unchanged.

## Whole-task outcome remains0/2

North failed alignment after39.0 s, with path1.748954 m and terminal distance
1.532853 m from home. South completed release-verified alignment and advanced
onto its second traversal, then stopped with a sensor failure after47.2 s,
path4.376683 m and terminal home distance3.149386 m. Neither discovered the
marker, returned home, recorded contact/native body stop or made a false home
claim. Both physical zero-release checks pass. Trusted graph edges remain zero.

The south alignment completed at tick308, after0.5 s of actual zero commands:
heading error0.01708286 rad, projected rate-0.02239503 rad/s, with the required
dwell. It entered its second traversal after the existing hold. This is actual
progress beyond the first local arrival/scan, not a complete mission or robust
alignment demonstration: north still fails the same operator.

## North: heading release still does not reliably settle

The first release began at tick307 with error0.00440346 rad, comfortably inside
the tighter inner target. Under zero requests the error then increased through
0.00694,0.01095,0.01450,0.01767 to0.02080382 rad at0.5 s; measured projected
rate remained around-0.03 rad/s. A second release late in the same12-s budget
also failed. Terminal error0.02006158 rad is still failure, not rounded success.

Thus a smaller pre-release error did not reliably produce a stationary heading.
The synthetic one-off0.008-rad recoil fixture was insufficient to model the
longer actual drift. No further inner-tolerance tweak alone is justified as the
main next scientific step. Consider continuous measured heading stabilization
through operational holds, with any nonzero compensating request explicitly
logged and subjected to the same turn-volume gate. That is a different local
control contract, not a retroactive zero-command success; whole-task terminal
zero-release scoring and invalid-sensor zero commands must remain unchanged.

## South: target validity is a prerequisite to estimator recovery

The second local target at observation327 was4.13436 m forward, derived from
an earlier near-boundary corner hypothesis. In that very observation, measured
front-wall segments have normal approximately[0.99704,0.07684] and offset
3.53494 m. Their adjacent supports reach body lateral coordinates+0.00410 and
-0.01229 m around the forward ray, placing the observed blocking surface at
about3.545 m forward. The chosen target is beyond that measured wall. This is
sensor-only evidence of a planning contradiction; no privileged map is needed
to reject it. A corner hypothesis whose opening extent is explicitly unverified
must not override a directly observed occluder.

At observation467, after further forward motion, the measured front-wall offset
is0.93637 m. The translation normal spectrum falls to
[0.00016147,0.21013205,0.78970649]; its weak direction is approximately
[-0.01339,-0.04975,0.99867], predominantly vertical. The fitted residual is only
0.05633 mm, but vertical displacement is not observed. Cumulative position is
invalidated and the controller stops. Do not replace the missing vertical
component by zero or weaken the rank test.

Read-only diagnostic c8ab13 counted actual retained ground-like motion points
(normal aligned within the existing0.97 criterion to the observed gravity
direction, points below the body). Counts were1,281 at327,593 at450,351 at460,
70 at465,103 at466, and zero at467/468. The loss of floor support explains the
missing vertical constraint. Restoring motion estimation alone would not repair
the contradictory target and could permit continued motion toward the wall.

## Measurement scope and negative results

North motion estimates are full on390/390 intervals, maximum step error0.77640 mm,
final position error0.52042 mm; all391 depth checks pass. South has466/472 full
intervals, maximum full-step error0.95351 mm and no final cumulative position,
so its motion check fails. Depth checks pass472/473; frame350 retains a38.82749-mm
maximum error requiring exact-ray diagnosis. Do not assume its cause from earlier
wall-edge cases. Full audit PASS means faithful replay/scoring, not that these
scientific checks or the task passed.

## Next implementation plan

1. Validate every candidate approach against current observed blocking surfaces
   and bounded ray support; do not allow a near-corner target to supersede a
   nearer observed wall. Preserve unknown opening extent and current lateral/
   height coverage. Add the actual second-traversal contradiction as a replay
   regression, plus synthetic concave corners, partial supports and occluders.
   Recheck stopping support during motion, not only at the target.
2. Make stopping targets compatible with the available sensor view, or obtain
   genuinely independent vertical-motion evidence before entering a floor-blind
   approach. Evaluate causal IMU/contact-kinematic fusion or a separately declared
   downward range observation with a deployment counterpart and uncertainty.
   Do not infer fixed height, zero slip or a missing translation component from
   command intent. No new sensing configuration is implemented or authorized by
   this result document alone.
3. Replace the unreliable open-loop heading release/hold assumption with measured
   stabilization and explicit post-transition heading checks. Preserve actual
   motion/rate/clearance constraints, invalid-input stopping and full-mission
   metrics; preregister any changed local holding contract as a new intervention.
4. Freeze the resulting controller/protocol and test full discovery/return again
   in fresh named runs, preserving all predecessors. Once execution is reliable,
   perform matched memory, supervised/JEPA predictive-training and genuinely
   multi-step planning comparisons, then independent layouts/seeds, sensor
   robustness and bounded real Go2 work when hardware is available.

The ultimate scientific goal remains active and unachieved. This work remains a
geometric development control baseline over learned locomotion, not a learned
JEPA navigation policy or evidence of JEPA's contribution.

## Exact identities

Output: `.generated/go2_release_aware_navigation_development_v1_attempt_001`.

- launch.json: `76157aa394eebac48ff2feff2c3879bb61cb5471fa418a6aa22d44dd3ae34597`
- result.json: `b3d2e6ee9f15761071a76367a4f5cb31bfe78e9fb59d3c58757ea45e868d7d78`
- raw_artifact_audit.json: `e4db58b445c28945dc21ce028db4aec96981b42c60d96f34854dfcf20697cbb3`
