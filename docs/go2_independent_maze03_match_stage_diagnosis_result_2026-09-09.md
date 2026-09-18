# Maze3 tracking failure: exact matching-stage diagnosis

The saved observation264 failure is reproduced on both cameras against all
eight retained references256–263. Every pair has fewer than the required12
lifted correspondences, so none enters rigid consensus fitting. The result
identifies correspondence loss; it does not establish that discarded tracks
are correct or authorize a weaker pose gate.

Root: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_independent_maze03_match_stage_diagnosis_v1_attempt_001`.
Session5722 completed normally, exit0. Result SHA-256:
`995d163abbea7bd63ee9607a6d60052421cbe49044095a52a88745709ad90134`.
Launch SHA-256:
`9fea82edfe278fd75c5ea7a2dec6ae8d3dfd266ff664da5f2c781670b6eb7e8d`.
Script `scripts/diagnose_go2_independent_maze03_match_stages_v1.py` SHA-256:
`c8aa94fd7e529cc909db6d7d9fabd18770bdab69a4997714b6b8c4c27f93c5c5`.
The launch binds1660sources and1686input artifacts, checked before and after.
All64output arrays (four per camera/reference pair) exactly match the frozen
matcher in dtype, shape and bytes. Sequential stage counts are nonincreasing.
All16saved rejection reasons agree. No checkpoint, model, simulator, command,
pose admission or threshold change was used. Work time3.866693033836782s.

Final lifted match counts for references256through263:

| Camera | 256 | 257 | 258 | 259 | 260 | 261 | 262 | 263 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary | 6 | 6 | 5 | 6 | 7 | 8 | 9 | 6 |
| Auxiliary | 0 | 0 | 0 | 1 | 1 | 4 | 9 | 11 |

For the immediately preceding263→264 pair:

| Stage | Primary | Auxiliary |
|---|---:|---:|
| Reference selected features | 40 | 147 |
| Current selected features | 48 | 152 |
| Forward descriptor ratio pass | 14 | 50 |
| Backward descriptor ratio pass | 12 | 55 |
| Mutual/distinct locations | 11 | 28 |
| Optical-flow status/finite | 11 | 24 |
| Forward/backward agreement | 8 | 18 |
| Descriptor-location agreement | 6 | 11 |
| Valid paired depth lift | 6 | 11 |

Depth lifting removes no additional matches after preceding gates in any of
the16pairs. Primary matching is already below12before flow filtering in the
latest pair. Auxiliary loses7of18tracks at descriptor-location agreement,
leaving11. The existing0.5pixel forward/backward and1pixel descriptor-location
thresholds are unchanged. Unlike maze1's latest auxiliary pair, which had12
lifted pairs but only11initial rigid inliers, maze3 never reaches rigid fitting.

This supports evaluating a separate correspondence association method on
actual images while retaining geometric validation. It does not support simply
lowering the minimum count or dropping the location check. A successor must
demonstrate valid geometry, temporal continuity and prospective physical
execution. More accepted matches alone do not prove better tracking.

Hardware at admission:16physical/32logical CPUs with all32in affinity,
6.5%CPU busy,78.729GBavailableRAM,90.907GBartifactfree,21.360GBworkspacefree;
both GPUs0%busy. One OpenCV/BLAS thread, alongside the active reactive audit
and ordered feasibility replay. The original scene failure and every input
remain unchanged; this diagnostic is post-hoc development analysis.
