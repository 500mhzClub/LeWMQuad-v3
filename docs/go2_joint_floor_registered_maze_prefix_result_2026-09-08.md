# Common-floor registered maze prefix result

Reviewed 2026-09-09. The prospective replay completed across 960 observations
(frames 0–959). Its first changed requested command is frame 959: hold
`[0, 0, 0]` instead of the predecessor's left turn `[0, 0, 0.45]`.
Neither controller has a terminal policy difference at this intervention.
This establishes a reviewable policy intervention, not improved navigation.
No outcome of the changed command was read or inferred.

Artifact root: `go2_joint_floor_registered_maze_prefix_v1_attempt_001` under
the external navigation development artifact base.

- Result SHA-256: `c1a2c347d5aea7a1e0db352a451daf4add20c9c42905886712778cbdd53166ea`.
- Launch SHA-256: `e42805bdb7edc73939aeef89b00978c60a0d36b5a06e947204ef19e71d24fa02`.
- Decision stream SHA-256: `3fea62a7d96b24e3ea55d357e9efc92bba6588f0ce4e0089bc9a52bc0c9316f3`.
- Bound sources: 1501. Wall time: 932.331636 seconds.
- Unchanged model state: `4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.

The completed predecessor raw audit was reused; its controller was not rerun.
The fresh candidate preserved the original visual evidence exactly and the raw
forecasts wherever both controllers planned. Its declared intervention is the
common-plane pose correction used by map, contact, residual and mission
consumers. The replay stopped before the changed command's unexecuted outcome.
The failed independent-plane prefix and its immutable evidence remain preserved.

At frame 959 the combined plane contains 22,985 measured candidates: 5462
primary and 17,523 auxiliary. The second covariance eigenvalue is
0.06588605085 m², above the unchanged 0.0025 m² admission threshold. Maximum
primary/auxiliary point residuals are 19.5532/11.4261 micrometres, below the
unchanged 3 mm per-point gate. Normal translation correction is -4.36114 mm
and tilt correction 0.002448251 radians. No candidates are trimmed. The static
floor identity, pose uncertainty and ground support remain uncertified.

The candidate selects hold in waypoint mode after the existing surface and
nominal constraints; the replay provides no evidence that this choice advances
the robot. A fresh native execution must establish its actual consequences.
The unchanged prefix includes predecessor strict primary visibility failure
frame 909; this estimator intervention does not repair that measurement result.

Next run the prepared native launcher's preflight with the exact result hash,
inspect current resources and bindings, then execute its one fresh CPU scene.
Preserve numerical settings, the original action set, shared 3000-tick mission
budget, raw audit and strict visibility checks. Seven native maze-0 attempts
remain navigation failures; independent-layout arrival/return, matched baseline,
real-time and hardware evidence remain outstanding.
