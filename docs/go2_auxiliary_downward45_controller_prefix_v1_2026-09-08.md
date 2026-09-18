# Explicit 45-degree auxiliary controller prefix V1

Authenticate the passing 18-frame native 45-degree capture and the completed
two-model 30-degree reobservation mission/readout. Replay both fixed corrected
seed-2026091001 models with the explicitly bound 45-degree public packet and map.
The primary observer, model input, weights, correction, selector, collision
checks, bounded reobservation and goal/execution state machine remain inherited.

Reject cross-calibration packets in both directions. Admit only the actual depth,
mask and redacted acquisition identity; no native pose or segmentation enters
the controller. Use the correct 45-degree optical/body transform for every
sample and retained floor patch. Preserve all floor and other/unknown returns,
occupied cells, full-foot coverage rules and original constraints.

Run two fresh model/controller replays per method, requiring exact decisions
and unchanged model state. Compare with each method's 30-degree predecessor
only through the observation before the first command or terminal difference.
Check the fixed recorded native prefix explicitly. Later replay rows are shadow
decisions on that fixed trajectory. The last observation's proposal was not
executed. Infer no counterfactual native result or goal arrival.

Freeze complete source and input bindings before exclusive root
`go2_auxiliary_downward45_controller_prefix_v1_attempt_001`. Use one CPU process
and one numerical thread, with 8 GiB available RAM and 256 MiB output allowance
above the 40-GiB reserve. Record hardware and reverify all source, sensor, model,
correction and outcome bindings afterward. This integration result alone grants
no navigation, real-time, independent-maze or hardware qualification.
