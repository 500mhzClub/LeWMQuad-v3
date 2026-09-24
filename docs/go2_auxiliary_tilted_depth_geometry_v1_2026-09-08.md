# Separate tilted depth camera geometry candidate V1

Characterize one fixed additional camera: optical centre [0.35, 0, 0.08] m in
the robot body frame, downward pitch 30 degrees, and the existing depth
intrinsics/resolution and 0.2–5 m sensing range. Preserve the original front
RGB stream and all trained models. This is a proposed extrinsic geometry,
not a validated hardware mount, rendered sensor or observation source.

Use only the corrected JEPA run's recorded observed poses and fixed measured
floor height to project the six full 44-mm terminal foot squares into this
hypothetical camera at every observation up to terminal. Report every result
and the earliest complete frustum view, including which occur no later than
the observation before the first translating command. These terminal regions
are retrospective diagnostic inputs; they may never be supplied to an online
view policy. Test the full extrinsic coordinate adapter independently.

Do not treat frustum containment as measured floor coverage. No depth pixels,
occlusion checks, robot visibility, mounting proof, training, command selection,
native scene or counterfactual navigation outcome is generated. A useful result
would justify a subsequent bounded raw sensor capture and self-occlusion audit,
with physical/model/sensor identities preserved and all observed obstacles
retained before any navigation adoption.

Freeze the source, tests, visibility result report and predecessor bindings
before `go2_auxiliary_tilted_depth_geometry_v1_attempt_001`. Use one CPU process
and numerical thread; this small serial array calculation has no useful
parallel scene workload. Require 2 GiB available RAM and a 64-MiB output
allowance above the 40-GiB reserve. Reverify all source/input identities afterward.
