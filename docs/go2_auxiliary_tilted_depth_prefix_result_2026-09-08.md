# Robot-visible auxiliary depth prefix result

Actual auxiliary depth establishes complete measured-floor coverage for all six
previously uncovered terminal-foot regions at frame 17, before the recorded
first translating command at tick 19. This resolves the identified observation
gap in this sensor characterization. It is not a prospective navigation result.

The completed integrity-V2 attempt captured 20 primary/auxiliary frame pairs,
replaying the original 19 zero/turn commands in one fresh robot-visible CPU
scene. All 1,700 physical samples, body sensor samples, fast-gyro samples and
policy/gyro histories match the original corrected JEPA prefix exactly.
Settling/setup and the original native contact/speed/domain stops passed. All
raw auxiliary depth masks were reconstructed exactly from the native finite
0.2–5 m depth criterion. All sources and input/output bindings reverified.

All six 44-mm foot squares first pass the original single-frame measured-pixel
coverage test at frame 17 (3.2 s physical clock); they remain covered through
frame 19. No subdivision, interpolation, footprint shrinkage or unknown-area
waiver is used. The shared auxiliary frame's native depth SHA256 is
`d16fc25809605842be09a8c8638a383d2bc7f367ed78cede476429e478fb3a11`.
The audit uses the original run's observed poses and the retrospective terminal
foot regions; neither is a newly tested online controller input path.

The renderer includes all 33 robot visual geometries and keeps the original
physical geometry. Native link segmentation identifies the robot roster but
contains zero robot pixels in all 20 auxiliary images. This reports no robot
self-occlusion in these sampled auxiliary views; it does not validate every
posture, camera housing, mount clearance or real hardware. Segmentation is
evaluator-only and never supplies the floor classification.

Primary RGB compatibility is not exact. All 20 primary native-depth arrays
remain bit-identical, while 1,268–11,135 of 307,200 RGB pixels change per frame
(about 0.41–3.62%). Maximum absolute channel differences are 13–25 on the
0–255 scale, and per-frame mean absolute channel differences range from
0.005032552 to 0.122473958. Visual inspection of frame 0 shows the same scene,
but that does not establish model or feature-observer equivalence. The source
of these RGB changes has not been isolated, and no model consumed these images.
Preserving camera extrinsics is not a substitute for checking image compatibility.

The diagnostic auxiliary capture costs 66.98–94.42 ms per frame, median
78.5748515 ms, including auxiliary RGB, depth, segmentation and file output.
Sequential rendering emulates two sensors at one paused physical sample. These
measurements do not establish simultaneous sensor operation, deployment latency
or real-time navigation. The complete collection and raw audit took
44.380167519906536 s after admission. The launch observed 76.82 GiB available RAM
and 63.85 GiB free artifact storage, above its 32-GiB/1-GiB-plus-40-GiB-reserve
requirements. One causal scene offered no useful independent-scene parallelism.

Two prior acquisition failures are preserved:

1. Original V1 failed on native segmentation background `0: -1` before any
   prefix command. The integer sentinel was incorrectly treated as a tuple.
2. Integrity V1 saved one auxiliary frame and executed one zero command, then
   failed because the collector checked for the next primary observation before
   invoking lazy current-frame acquisition.

Integrity V2 preserves scientific settings and corrects only these acquisition
issues. The two original scope tests, nine segmentation/scope checks and two
lazy-acquisition tests passed before the relevant launches. The new lazy test
exercises all 20 paired-frame requests and rejects wrong indices/clocks. The
separate geometry and visibility helpers also each passed two focused tests.

Completed artifacts under the fixed navigation development artifact root:

- Capture `go2_auxiliary_tilted_depth_prefix_integrity_v2_attempt_001`:
  launch `a01f0975b8d80cbd9633672db9b0cb397d08372bccb5a89cf222ee651308fcf7`;
  raw audit `8693b8c94cf5a570e9a33205a5d241bf3b67024473bfe8f3b7efa10bed5b463e`;
  result `e8de0873c71f50c5a01793c0464ba2014c1a7fe72f77daf47204eaeee1c8c38a`.
  All 152 bound capture artifacts are retained.
- Readout `go2_auxiliary_tilted_depth_prefix_readout_v1_attempt_001`:
  launch `3fad14a596b09d5e1d641befbafc5f588dc0a560bcd4914a04d77a9b03fe3f22`;
  result `f381538cf2f67afa6e654494027409178905ce625ef48f69a35366d26c912cb1`.
  Its 1,215-source closure includes both acquisition failures and fixes.

Next work, in order:

1. Replay the new primary RGB/depth/body prefix through the unchanged visual
   observer and fixed corrected model assignments. Compare sensor-derived pose
   gates and selected commands over the causal prefix. Do not equate small RGB
   errors with model compatibility or infer unexecuted future outcomes.
2. Implement an explicit auxiliary depth packet with its own calibration and
   acquisition identity. Keep native pose, scene labels, segmentation and the
   retrospective target foot coordinates outside online inputs. Register actual
   auxiliary depth using the currently observed body pose and tested extrinsics.
   Retain all auxiliary obstacles/unknowns as well as floor evidence; adding only
   favorable floor pixels is not an adequate integration.
3. Add the measured auxiliary floor evidence to a distinct online controller and
   raw auditor, preserving arrival, collision, timing and resource gates. Run new
   prospective closed-loop cases with the preselected models. The original
   corrected JEPA/direct failures remain unchanged. Share any adopted sensor
   stack across subsequent matched reactive/nonpredictive and memory baselines.
4. Establish independent-maze exploration, useful memory, physical backtracking,
   matched learning/planning/memory comparisons and realistic timing/hardware
   evidence before any overall-goal completion claim.

There are still zero verified navigation arrivals and zero independent novel-maze
evaluations. This fixed-prefix capture and its retrospective sensor readout do
not change those counts. No navigation or hardware qualification is claimed.
