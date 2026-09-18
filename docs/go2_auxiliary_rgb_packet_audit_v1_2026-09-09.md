# Causal downward RGB packet integration on the fixed diagnostic segment

Validate a distinct auxiliary RGB packet on all 18 previously diagnosed
frames 1853–1870. Bind the completed pair result
4980170b5c02e9ab8de55f331583f7b0070decdc53aa753dbe9466821cf855da
and its launch/source/input identities. Reconstruct depth through the existing
public adapter. Decode RGB without color conversion and require the recorded
acquisition pixel hash. Pass only frame, time, pixel/depth hashes and fixed
calibration from acquisition metadata; native poses and segmentation are not
packet inputs.

The separate RGB schema requires uint8 480×640×3 pixels, fixed downward
calibration, exact public depth/primary RGB/episode bindings, equal acquisition
time and availability no later than the decision. The replay assumption is
ideal zero latency, stated explicitly; no hardware timing is inferred. The
constructor owns copied RGB pixels; depth invalid rays remain unknown.

Require every feature witness and public pixel/depth digest to match the
completed diagnosis. Verify source and artifact identities before and after.
This tests packet integration, not refitting or continuous pose estimation.
The observer/controller and running native experiment remain unchanged.

Run scripts/audit_go2_auxiliary_rgb_packet_v1.py --preflight-only before the
exclusive attempt go2_auxiliary_rgb_packet_audit_v1_attempt_001. One CPU process,
one numerical thread, no new native scene, 2 GiB available RAM and 64 MiB output
above the 40 GiB storage reserve. Record current CPU/GPU/RAM/storage and
competition beside the existing native scene and controller replay. No
training, navigation, uncertainty, real-time or hardware qualification claim.
