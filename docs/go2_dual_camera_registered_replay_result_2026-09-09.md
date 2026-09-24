# Dual-camera motion admission and floor registration completed

Result: `2310908d24aee137884d6e4fcfded92a2b31f03b93edfae7608ba48f2a0b4190`.
Launch: `85842f515db20635c0f7df9807c3d10030510e3b63bdca85561cbc4b7d604c24`.
Stream: `d35e8b9fade8ebb987581031becdbd7c3c189463c0e3a946d0db5e3d9532c303`.
Postfit errors: `cd87c21c947be50c9768679f297d00cfd746fbd288701766361b87dc3c5f98f1`.
Root: `go2_dual_camera_registered_replay_v1_attempt_001` in the development
navigation artifact store.1566 sources and final input/source checks pass;
wall343.070163273s.

All1881 raw poses pass the dual-camera motion accessor and reproduce the
completed standalone observer's pose, continuity, camera selection, reference
selection and overlap receipt. All1881 poses also pass unchanged joint-floor
registration and its existing witness accessor. Frames0–1869 reproduce the
original registered poses and complete floor-registration receipts exactly.
No failure or reinitialization occurred.

| Registered postfit error | Mean | Maximum |
| --- | ---: | ---: |
| 3D translation | 4.229 mm | 8.582 mm |
| Horizontal position | 4.227 mm | 8.578 mm |
| Rotation | 0.004236 rad | 0.010556 rad |

Native state entered only after observation processing. Raw-pose errors
reproduce the standalone observer exactly. These are measured retrospective
errors, not calibrated uncertainty bounds. Combined wrapper/admission/floor
processing median140.790ms, maximum302.732ms; all1881 frames exceed100ms.
This timing includes the replay's extra admission/comparison calls and excludes
decoding, mapping, planning and physics, so it is not a native-loop timing.

This closes full recorded motion-witness/floor integration. The prepared
controller still needs complete decision-prefix validation, and a new
prospective episode must execute the resulting commands. The old recorded
stop/drain trajectory does not demonstrate physical return. Native audit,
independent layouts, matched contributions, sensing/timing, uncertainty and
hardware requirements remain separate and incomplete.
