# Fresh RGB-D/inertial shadow-motion result

One fresh development layout, three matched appearances, actual Go2 simulation
physics. All three completed the frozen 50-command tape and three zero-command
tail ticks. The estimator did not choose commands; native simulator guards
supervised collection. This is not a completed navigation mission.

| Appearance | Accepted RGB-D pairs | Admitted frames | Maximum position error | Outcome |
|---|---:|---:|---:|---|
| Neutral | 0 | 28/54 | 4.888 mm before failure | Uncertainty-budget failure at 4.3 s |
| Repeated | 53 | 54/54 | 2.441 mm | Shadow tracking completes |
| Distinctive | 53 | 54/54 | 2.734 mm | Shadow tracking completes |

On the common first 28 frames the maximum errors are respectively 4.888, 1.015
and 1.136 mm. Both textured conditions fill 18 weak-plane intervals with accepted
point constraints; the original depth rank remains unchanged. Their final
position-error scales are 46.920 mm, including 36 mm of linearly accumulated
point-error hypotheses. These scales are uncalibrated, not measured confidence
bounds. The neutral condition reaches 80.194 mm against the unchanged 80 mm
budget; its failed observation is never admitted and its estimator never resumes.

The independent saved-artifact audit passes: all 582 bound artifacts, raw sensor
reconstruction, sensor-only estimator replay, geometry/material/actuator identity,
camera/depth checks and native collection guards. All raw physics arrays are
bit-identical across appearances. There are 3,400 physics samples and 54 RGB-D
frames per arm; no native guard violation. Maximum body speed is 0.06721 m/s,
net post-settle displacement 25.411 mm, and zero-tail displacement 3.863 mm.
This is small local motion, not corridor traversal. Final quiet-window maximum
linear/angular speeds are 0.01009 m/s and 0.01402 rad/s, not exact rest.

Every measured outer tick exceeds 100 ms. Textured median times are 273.5 and
260.3 ms, including physics/capture/estimation and heavy diagnostic checks under
concurrent regression load. This neither establishes deployment latency nor
meets a demonstrated 10 Hz full-loop requirement. The regression run passed
1,909 tests across 154 explicitly selected files; raw audit also completed.

Output: `.generated/go2_rgbd_shadow_motion_development_v1_attempt_001`.

- `launch.json`: `628168df54a4e8d96575f939609057f610de35d60fd579cd58d76ca58dd964dc`
- `result.json`: `4a5b97bb50f28ca631bdb1bb1e3ee953658105613742333c638429b0c482bfd1`
- `raw_artifact_audit.json`: `dffa2ff61d64d32d54de92b9d445b6cc21e501649d368b657987a0459bfad42f`

The launched 452-source/4,804-input closure and all predecessor results remain
unchanged. No sealed material, hardware execution or new JEPA training was used.
Ideal simulated depth/IMU, one layout and small motion do not establish noisy
sensor robustness, long-duration localisation, uncertainty calibration, learned
navigation or JEPA advantage. Whole-maze completion remains 0/2.

Next: feed this one fused state into discovery/return control, preserve raw
depth provenance and stop conditions, test full mission transitions and actual
stopping-tail ingestion, then run independently specified complete missions.
Do not substitute a short tracking result for that outcome. Long-duration
relative-pose error transport, sensor robustness, matched JEPA/multistep/memory
comparisons and bounded hardware evaluation remain required.
