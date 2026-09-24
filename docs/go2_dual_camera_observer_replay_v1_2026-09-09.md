# Continuous dual-camera observer replay from the ninth episode start

Run one continuous DualCameraAnchorPose instance from frame 0 over all 1881
recorded observations of full_jepa_novel_maze_00 in the ninth completed native
episode. No commands, controller, model training or physics are executed.
Bind its result3745c0b7c45265a2fcc38caf732b6f3f4b228487f344d23502dc7395077d5755
and all input artifacts, plus completed auxiliary packet audit
122da7713c522ca06e8c6881597d42cc2b6f49a35bae691703aaf94f20a2b5c0.

Both simultaneously captured views are retained at each accepted common body
pose. Apply the original front-camera anchor/increment selection first. Try
the auxiliary camera only for NO_CURRENT_MEASURED_TRANSLATION or exhausted
measured bridge. Qualified primary conflicts remain terminal. Auxiliary
descriptors match only auxiliary descriptors, with fixed extrinsic/lever-arm
conversion of gyro, poses and point witnesses. Keep all fitting, displacement,
reference-selection and anchor/increment agreement rules. Eight paired anchors
and one previous paired view share the original ten-frame bridge budget.
Bridge-only poses cannot promote anchors. If a primary increment qualified
before exhausting its bridge budget, compare it against the auxiliary pose
with the same 2 cm / 0.10 rad disagreement limits; conflict remains terminal.
Retain half-overlap frames using the selected camera's feature population.
No pose reset, command integration, native pose input or uncertainty claim.

Before any auxiliary attempt, require the accepted pose fields, continuity,
reference selection and overlap receipt to equal the original saved observer
evidence exactly. Save all subsequent camera choices, failures, accepted pose
and continuity witnesses. A terminal observer never restarts. Open native
poses only after all observer decisions finish, then evaluate position and
rotation errors for every accepted frame. Report first fallback, first failure,
coverage, promotions, bridge use, observed errors and processing time. These
are retrospective development diagnostics, not an independent layout, online
controller recovery, calibrated error bound or navigation success.

Source tests cover exact primary-path behavior, real synthetic feature loss
and camera return, qualified conflicts, shared bridge exhaustion and latched
invalid input. They do not replace this continuous actual-recording replay.

Exclusive output go2_dual_camera_observer_replay_v1_attempt_001; preflight via
scripts/replay_go2_dual_camera_observer_v1.py --preflight-only. Record hardware
and competition before launch. One CPU process/numerical thread, 4 GiB
available RAM and 1 GiB output above the 40 GiB reserve, no new native scene.
Launch after the paired timing benchmark finishes, beside the existing tenth
episode audit if it is still live. Preserve all frozen attempts and outcomes.
