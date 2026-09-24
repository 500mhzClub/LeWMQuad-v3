# Continuous tracking recovered the ninth recorded failure

Result: `540f63243f74b63e90dadeaa8e7aebde936bb7880d49f0c77ffc5e8affc59eea`.
Launch: `f0ea9c76b46e2c0724159deed6d97f821827a27415d91b007dca99b06f1a56d0`.
Stream: `80debed2accb3361e445a9ca44896a591703455bbdb8b944e2c393f9682562b5`.
Postfit errors: `59ca2616cc6f23465c439eefceed9f5fcf36682f3bb57669195132a328907551`.
Root: `go2_dual_camera_observer_replay_v1_attempt_001` in the development
navigation artifact store. 1558 bound sources; final source/input checks pass.

One continuous observer accepted all 1881 frames without terminal failure or
reinitialization. Frames 0–1869 reproduce the original primary pose fields,
continuity, reference selection and overlap receipts exactly. Auxiliary RGB
was first attempted at 1870, the original tracking failure. It supplied three
accepted anchor measurements:

| Frame | Reference | Inliers | Promotion |
| --- | ---: | ---: | --- |
| 1870 | 1854 | 17 | Qualified recent reference |
| 1871 | 1870 | 27 | Accepted half-feature overlap |
| 1876 | 1875 | 28 | Accepted half-feature overlap |

Each followed primary NO_CURRENT_MEASURED_TRANSLATION. There were 1877
primary-selected frames plus the initial paired reference. Eight references
remained retained, with 331 total keyframes and one measured bridge frame.

| Postfit error | Mean | Maximum |
| --- | ---: | ---: |
| 3D translation | 7.552 mm | 15.468 mm |
| Horizontal position | 4.379 mm | 8.986 mm |
| Rotation | 0.005182 rad | 0.011267 rad |

Native poses were opened only after the entire observer run. These are
observed errors, not uncertainty bounds. Observer-only median processing was
54.526 ms, maximum 200.111 ms, with 18 of 1881 frames over 100 ms. This excludes
sensing, decoding, mapping, planning and physics. Total replay wall time was
187.523494 s on a shared machine.

This closes continuous observer coverage on the ninth recorded episode. It
does not produce new commands or alter the original failed navigation result.
The recovered frames include the old stop/drain trajectory, not a newly
executed return. Next requirements are motion-witness/floor-registration
integration, complete controller prefix validation on the latest completed
episode, and prospective native execution. No independent-layout, calibrated
uncertainty, timing, hardware or navigation qualification is claimed.
