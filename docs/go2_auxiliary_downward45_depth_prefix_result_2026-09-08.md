# Native 45-degree depth prefix covers the diagnosed full-foot squares

The fixed 18-frame primary/auxiliary capture completed and passed all original
primary and explicitly calibrated auxiliary raw, metric-depth and visibility
checks. All 1,600 native physics samples and the recorded ideal sensor, gyro
and policy prefixes matched the preceding JEPA mission exactly. All 18 primary
RGB frames also matched exactly. Every auxiliary frame had zero robot pixels,
while the complete 33-node robot visual geometry remained enabled in rendering.

Actual new auxiliary depth established complete retained measured floor patches
for four of the twelve diagnosed full 44-mm front-left-foot squares at frame 13,
eleven at frame 14 and all twelve at frame 15. All twelve remained covered through
frame 17, the observation before the first translating command in the recorded
mission. Each positive witness passes the original whole-square projected
floor-pixel check; no subdivision, footprint reduction, synthetic free space or
segmentation-based floor classification was used.

The experiment executed only the fixed first 17 zero/turn commands. Foot targets
from later stopping events and original primary-observed poses were used
retrospectively for this coverage diagnosis. No new observer, learned controller
or navigation policy used the 45-degree camera. Hardware mount calibration,
realistic delivery latency and prospective navigation remain unverified.

The capture took 44.032079 seconds, bound 1,278 sources and 140 artifacts, and
preserved all previous attempts. The three geometry and two capture-scope tests
passed before their corresponding launches. Hardware was checked before the
single CPU scene; no competing substantive Python job was present.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | ec4b4005a0e09edcfe70ddd5abc31964c188b2f2d5ef7c85b3f9625ee898c25b |
| raw_audit.json | 17f6e3fecc763c1a558632b6af5a3ec945711ac704cc15f646fcd049b55c34e8 |
| result.json | 92f32aad3d2bde466a073df6410bf7060dfc9a1a1cdf5b9520fabcf8eb4268e4 |

Root: `go2_auxiliary_downward45_depth_prefix_v1_attempt_001` under the guarded
development base. Next implement an explicit 45-degree public packet, paired
replay and map integration while retaining all obstacle/unknown returns and
the fixed corrected models. Verify deterministic controller replay, then run a
fresh native probe. This completed sensor test is not a verified arrival or an
independent-maze result; the full goal remains active.
