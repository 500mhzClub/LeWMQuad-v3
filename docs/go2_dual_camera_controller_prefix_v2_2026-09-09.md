# Same controller prefix with explicit JSON identity restoration

V1 stopped at frame0 before any controller-prefix comparison was admitted:
the serialized decision has JSON-list episode identities, while the unchanged
pose accessor requires tuples. Preserve launch
8b01122596274745c0d2e0de89cf8f51956ebafc316b58977d7339223f571efe,
failure25421b3950be88c00340b586399ec151324bb771509782de6c809c7567637ca2
and mismatchf3e1d3d3894547374ff4dfb4aa4fb4f16d219cca707c9e4cffb2202a7488f79a.

The separately named V2 restores only the three explicit visual/registered
episode identity paths in each decoded original/candidate decision using the
existing strict json_identity adapter. It rejects malformed integers, wrong
episodes and behavior/witness differences. Delegate all remaining comparison
to the unchanged V1 comparator. No controller, sensor, model, threshold,
collection binding, first-intervention stopping rule or qualification changes.
Inherit and revalidate the failed V1 source and artifact witnesses.

All scientific/resource rules in go2_dual_camera_controller_prefix_v1_2026-09-09.md
remain in force. Use exclusive go2_dual_camera_controller_prefix_v2_attempt_001
and scripts/replay_go2_dual_camera_controller_prefix_v2.py with the same actual
registered result hash. Run the serialized-decision regression tests and
--preflight-only before execution. No new native run until the tenth full
audit and final collection binding match. Native61184 remains untouched.
