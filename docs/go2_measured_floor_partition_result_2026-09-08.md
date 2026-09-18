# Complete floor partition: recorded evidence for a scoped contact rule

The direct-039 sample-bound recording was reconstructed with all 219 available
floor-map receipts and every candidate surface check at 44 forecast contexts
exactly matching the original. Ten unavailable/drain observations remain
unavailable. All 4,167,020 returns were classified individually: 997,824 passed
the measured-floor patch test and 3,169,196 remained non-floor or unknown.
There are 9,830 floor keys, 16,666 other keys and 353 mixed keys. Mixed keys
retain both populations; no first witness determines later classifications.

Four new partition/projection tests and fourteen existing geometry/bound tests
passed (18 total, 0.49 seconds). The diagnostic then took 50.911 seconds after
launch using one chronological CPU worker. Preflight recorded 82.44 GB available
RAM, 86.75 GB artifact free space, 0.3% CPU activity and idle GPUs. The result
binds 986 source files. No simulator, training or contact-policy change occurred.

All observed foot intersections in these recorded candidate checks contain
measured-floor returns; none intersects a non-floor/unknown partition bound.
This does not suffice for an exemption: the proposed rule also requires every
grid cell touched by the foot's entire nominal projected disk to be measured
floor. With that requirement, the original 110 conflicting candidates become
74 conditional conflicts; 36 would cease to be vetoed. This is recorded-state
analysis, not executed improvement or a support certificate.

At tick 83, forward and right arc become conditionally admissible. Left arc and
left turn retain foot conflicts because complete projected floor coverage is
missing; hold and right turn were already without a conflict. At the terminal
tick 218, five of six candidates become conditionally admissible; left turn
remains vetoed because its left-front foot projection is not fully covered.
Thus the rule retains a concrete unknown-ground restriction even though the
intersecting measured samples themselves pass the floor-patch classifier.

The justified next prospective experiment explicitly permits measured-floor
contact for the four foot spheres only under this coverage condition, while
retaining non-floor/unknown foot conflicts and every non-foot collision check.
It grants no terrain, friction, pose, prediction, penetration or hardware safety
calibration. Earlier attempts and their zero-goal outcomes stay unchanged.

Artifacts in `go2_measured_floor_partition_v1_attempt_001`:

| Artifact | SHA-256 |
|---|---|
| `launch.json` | `8d441956bb21fe8cf4a21245be03ee3fb15b074e92560e168a51ab3f16e18147` |
| `classification.json` | `5d9e766c355598959e920adade9859dc5f439af947f1a9a23ddd2cf7fa01d8d0` |
| `result.json` | `f9fc578d87b702b02bead31e186493e0904d82ea48dcbc5413fffe68e3acc406` |

The full goal remains active and unfulfilled.
