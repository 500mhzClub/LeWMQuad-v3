# Physical first-surface visibility correction: fixed native camera bench

The recorded connected-layout pilot contains a confirmed see-through artifact.
In nominal_action_1 frame33 the optical-axis wall entry is0.02007389554m but the
native centre depth is1.22021878m and publicly valid. The renderer near plane
is0.05m. The old analytic reference discards box entries at/below that plane,
so agreement with that reference is not proof of physical visibility. Preserve
the original pilot, failed/corrected command audits and all negative outcomes.
Do not use its near-wall images as physically validated sensing evidence.

This distinct development experiment changes only the render near plane to
0.005m. It does not retract the camera, change the0.2–5m public depth interval,
fill invalid rays using geometry or change the camera's intrinsic calibration.
The new evaluator intersects the nearest positive opaque surface independently
of clipping and rejects a camera in/on a solid or below/on the opaque floor.
Its stride8 comparison is explicitly sampled, not a full-image visibility proof.
Render-near is not a claim of a real sensor's minimum measurable depth.

## Frozen bench before execution

Use one CPU/offscreen native scene per near-plane arm0.05and0.005m, the same
seed2026090901 and appearance seed2026090902. Retain native Go2 geometry but
execute zero physics steps and no gait/control. This is a camera bench: move
the camera explicitly, not the robot, and make no rigid-mounted-motion claim.
Two opaque12m-wide/12m-high walls have front faces x0.56and1.76m. Look along+x,
camera y0,z6m. Freeze front-wall distances0.004,0.006,0.020,0.049,0.051,0.199,
0.201,1.0and4.9m, nine per arm. Save all18RGB/depth pairs and native identities.
Same native pose/physics epoch must hold between separate RGB and depth renders;
verify native intrinsics, clip planes, optical pose and single-sample framebuffer.

The0.004m case deliberately lies below both render near planes. It must fail
the physical visibility check in both arms; this is evidence the new check
detects residual clipping, not a case to remove. The legacy0.006/0.020/0.049m
cases should also fail; all remaining legacy cases and the eight corrected
cases should pass. No threshold search or another near-plane arm is authorized
by this fixed bench. Unexpected results remain results and require diagnosis.

Pass requires at least1000interior sampled rays, maximum optical depth error
at most1mm, no sampled opaque surface at/below the native near plane, and no
publicly valid depth where the first surface is below0.199m. The1mm gate also
checks the loss of far-depth precision from reducing the near plane. Public
range is unchanged; the1mm-wide margin at0.2m avoids classifying quantization
at the exact sensor boundary as see-through. Compare saved values independently
after collection and verify artifact/source bindings before and after.

Output is the exclusive owned root
`go2_near_field_visibility_probe_v1_attempt_001` under the existing external
navigation-development artifact root. Reserve40GiB and cap this bench at256MiB.
Write launch/source/native bindings first; retain failure, never retry/resume or
overwrite this attempt. Do not launch l00 until this bench is reviewed and the
new capture/physical-first-surface audit is integrated and tested. This bench
does not establish robot self-occlusion, real sensing, successful navigation,
JEPA utility, collision-risk calibration or final-goal completion.
