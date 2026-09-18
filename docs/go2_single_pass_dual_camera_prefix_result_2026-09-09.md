# Exact optimized-map dual-camera prefix completed

Root go2_single_pass_dual_camera_prefix_v1_attempt_001, result
18b79c77516ef738366298ee3e093952a250cfd083938818004d232e71a7ca96.
Launch1ffeb5d5028a5092a7ed5f58da06556e860ec6231cd1ae102e25174986e87f97,
decision stream550606ee52254fa3e6a57cafa961eee9aa3d367a59e4552e927ec2d974aff201.
Session2602 CLOSED exit0, status SINGLE_PASS_DUAL_CAMERA_PREFIX_COMPLETE,
wall1841.509078746s,1624 frozen source bindings.

All1873 complete decisions match the completed dual-camera controller prefix,
including the auxiliary-camera intervention1872 and turn[0,0,-0.45]. All1872
earlier commands match the actual tenth native tape. Every input array remains
unchanged, the model stays4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6,
and final source/input checks pass. No following recorded observation was
consumed after the first changed command. The candidate replaces only the
empty-state map indices with the previously tested packed-owned/single-pass
implementation; no complete-decision metadata normalization was used.

This is equivalence on the declared prefix, not a controlled speed comparison
or a full native-loop deadline result. It does not cover the subsequent new
native trajectory or its floor-registration failure at1904. The current native
episode did not adopt it and remains unchanged. Any future implementation use
must preserve the new failure evidence and verify its own actual scope. No
navigation, round-trip, independent-layout or hardware qualification follows.
