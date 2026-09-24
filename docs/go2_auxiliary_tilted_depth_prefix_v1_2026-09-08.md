# Robot-visible auxiliary depth prefix characterization V1

Use one fresh CPU scene on the reused family_episode_039 development layout to
replay exactly commands 0–18 from the corrected JEPA run, with no learned model
or high-level navigation controller. These commands contain settling-history
zeros and turns only; capture observations 0–19, stopping before its first
translating command. Keep all physical geometry, friction, gains, gait, low-level
timing, command slew, setup and native contact/speed/domain stops unchanged.

The distinct scene must enable actual robot URDF visual meshes while retaining
its collision geometry. Keep the existing opaque floor/wall order and append
robot visual nodes in their native geometry order. Verify the actual draw-node
population and order. This is an instance-local renderer ordering hook; do not
change a predecessor module or renderer source. Primary camera calibration
remains unchanged, but report whether making the robot visible changes its RGB
pixels. No model may consume these new images in this characterization.

At each physical sample, acquire the fixed auxiliary depth candidate at body
[0.35, 0, 0.08] m and 30-degree downward pitch, with unchanged intrinsics and
0.2–5 m sensing range. Use the existing single-camera rasterizer sequentially
to emulate two rigid cameras at the same paused physics sample; restore the
primary transform exactly afterward. This is an ideal acquisition model, not
a simultaneous-camera or latency validation. Preserve raw RGB/depth and
evaluator-only link segmentation with its explicit native index map, including
robot pixels. Segmentation and native poses never enter floor coverage.

Authenticate native depth arrays and reproduce the finite/range validity mask.
Verify the complete physical, body, fast-gyro and policy-history prefix exactly
against the existing run. Reconstruct auxiliary retained floor patches using
the original run's recorded observed poses, its fixed measured floor height
and the tested auxiliary extrinsic adapter. This is retrospective sensor
characterization: neither a new observer nor online policy is being tested.
Evaluate all six full recorded terminal-foot regions at every captured prefix;
these future diagnostic targets may never be supplied to an online controller.
Preserve all pixel failures and unknowns. Report actual robot occlusion counts
and whether new floor coverage is established; do not infer navigation success.

Freeze all source/tests, original probe and geometric-characterization bindings
before `go2_auxiliary_tilted_depth_prefix_v1_attempt_001`. Require 32 GiB available
RAM and a 1-GiB output allowance above the 40-GiB reserve. Use one scene and one
numerical thread: the two views share one causal physical prefix, so independent
scene parallelism adds no useful comparison here. Retain the original resource
and physical stops, all partial artifacts and any terminal failure; no retries
or resumption. Reverify source/input/output identities after the raw audit.

This is simulation sensor development with robot visual meshes. Hardware mount
and housing feasibility, sensor calibration/noise/latency, new-observer replay,
all observed obstacle retention and prospective navigation remain unvalidated.
Any sensor adoption must be shared by matched baselines. No independent-maze,
hardware, real-time or overall-goal qualification follows from this capture.
