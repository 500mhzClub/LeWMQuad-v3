# Spatial SIFT quotas rejected by complete-trace replay

The candidate is **ineligible for a native probe**. It reproduces the original
left-case loss at frame5 and introduces a right-case loss at frame56. No new
native execution or commands occurred. Both original mission failures remain
unchanged, and the navigation goal remains active.

| Trace | Original current poses | Candidate current poses | Candidate first failure |
| --- | --- | --- | --- |
| family_episode_052,16 frames | 5 | 5 | frame5: no current measured pose |
| family_episode_039,254 frames | 254 | 56 | frame56: measured bridge budget exhausted |

Every frame was retained, including both terminal drains. Failures remained
latched. Original left-frame5 failure reproduced exactly. Maximum candidate
XY/rotation errors on accepted frames were0.777mm/0.000250rad and
3.931mm/0.002832rad. Conditional accuracy does not compensate for missingness.
Original right maximum errors were5.703mm/0.003670rad across all254 frames.

The candidate detects default SIFT points in valid public depth, filters with
unchanged lifting and selects up to50 per4x3 image cell, at most600. Matching,
rigid registration, six-cell coverage, gyro consistency, reference retention
and bridge gates are unchanged. Six focused tests passed, including an AST
comparison of the inherited observation method and real rendered RGB-D tests.

The failed left trace had only148 selected/liftable detections at frame0 and
118 at frame4; no cell quota was reached there. The600 global SIFT budget was
therefore not the limiting factor. The actual failed RGB image was inspected:
a nearby panel covers most of the image with large, sparsely textured blocks.
The selection quotas cannot create matches which the detector does not find.
In the right case, quotas did discard features as its view widened; the new
observer later exhausted its unchanged measured-bridge allowance. This is a
regression, not evidence for adopting spatial quotas.

Median active observer times, excluding packet loading, were34.410ms original
versus47.026ms candidate in the short trace, and57.646ms original versus73.263ms
candidate in the long trace. Candidate timing covers only its pre-failure
prefix. This is not a comparison of equal complete successful trajectories.

Hardware was assessed before replay:32 logical/16 physical CPUs,82.39GB
available RAM,96.58GB free artifact storage,0.3% CPU activity, idle GPUs and no
substantial competing Python task. Arms ran serially with one computational
thread. No model training, GPU workload, source export or hardware motion.

Root, under the established navigation artifact base:
`go2_spatial_support_observer_replay_v1_attempt_001`.

| Receipt | SHA-256 |
| --- | --- |
| launch.json | 4c227743004373f9f02ec9edf2a99ab3244cf130417cce9146ab81a10edd3465 |
| result.json | 1a8097c3ba3340ce00f3c4b20d995471b67f25a3743c9d8671f71f2e6ab027d9 |

The result binds six artifacts totaling17,813,305 bytes;866 sources were frozen.
It reports `candidate_eligible_for_separate_native_probe=false`. Preserve all
frozen sources and this completed failure. The replay process exited0; there
is no live job associated with this attempt.

The next visual-support candidate should change how features are detected,
not repeat quotas or lower the six-cell gate. Corners of the large texture
blocks are a concrete sensor feature to test, retaining descriptor ambiguity,
LK and geometric rejection. The separate turn-only policy still needs a
prospectively defined learning/planning change; an observer fix alone cannot
establish navigation success.
