# Fixed native foreground-edge crossing assay V1

Nine static camera poses around the corrected pilot's junction-action-3 frame11
test both sides of the known foreground silhouette. This is a development
mechanism assay, not independent-layout validation or a rescoring of the pilot.

Use its exact inventory scene, appearance, union surfaces and floor-first draw
order. Keep the native .005m near plane, single-sample depth target, 640x480
camera and core-profile precision readback. The base optical pose comes from
the exact committed `repeat_0_l00_junction_recent_forward_nominal_a3` camera audit.
Rotate about optical y by atan(offset/focal), at fixed optical position, for
principal-axis offsets -1,-.25,-1/256,-1/4096,0,1/4096,1/256,.25,1 pixels.
These offsets are fixed before rendering; do not search for passing poses.

Evaluate all stride8 centre rays with the unchanged strict 1mm score and separate
full-pixel-footprint boundary accounting. Record coverage, stable-interior errors,
boundary errors and below-near/false-valid occlusion failures at every pose.
At every pose choose the first stable interior reference ray with expected depth
in (.2,4.9)m, independently of measured error; perturb only a temporary analysis
copy to expected+1cm. The stable-interior check must reject this negative control.
Never modify raw arrays or policy packets. The native base-pose depth must be
compared bit-for-bit to the committed pilot frame; mismatch remains a result.

The bounded assay records all nine frames, two visual meshes, native identity
and order, precision/sample readbacks and scores. It executes zero physics steps,
one static build, exactly nine RGB and nine depth render calls. Output is the
exclusive child `go2_native_raster_edge_crossing_v1_attempt_001` of the existing
development artifact root, capped at128MiB including launch/result metadata with
40GiB free reserve. Bind exact source and input hashes before execution; preserve
any partial output and terminal failure, with no rerun or replacement within V1.

An accounting-mechanism pass requires all nine stable-interior scores and all
nine negative-control rejections, no near-occlusion failures, and exact base-pose
depth identity. Report every strict failure separately: the assay does not
require them to vanish. Passing explains a bounded sampled measurement mechanism;
it does not certify arbitrary silhouette depth values, all rays, thin obstacles,
policy-side free space, articulated collision safety, or deployment sensing.
Analytic counterexamples and an explicit conservative policy-side uncertainty
contract remain prerequisites before depth-based navigation qualification.

This does not grant pilot training eligibility or independent-layout collection
promotion. The full JEPA/predictive-rollout/memory/novel-maze/hardware objective
and all earlier negative results remain unchanged.
