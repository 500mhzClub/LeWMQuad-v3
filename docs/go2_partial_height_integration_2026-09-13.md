# Partial floor-height integration

The prior fine-obstacle run failed when a narrow floor strip could no longer
estimate a full plane normal. Current points remained available, but their
maximum residual against the transported floor reference crossed 3 mm.
Both 100- and 150-feature trackers reproduced the failure.

The new registration uses the original fully measured plane where available.
When at least 100 current points lack two-axis extent, it estimates only a
scalar height update while retaining orientation from uninterrupted visual
transport. It checks every current point against the corrected reference using
the original 3 mm residual limit and keeps the total 5 cm correction bound.
Partial observations have a distinct schema and do not replace the last fully
measured floor anchor. They do not claim a newly measured normal.

The independent depth observer separately fits current height with its
gyro-derived normal under the existing flat-floor hypothesis. It preserves the
same point count and residual limits, uses the original full-plane path at
initialization, and explicitly marks its later height-only observations. Its
1 cm obstacle grid, 0.45 m footprint, freshness checks and stopping allowance
remain unchanged.

Two focused tests passed in 2.04 s. The integrated replay of the failed
193-frame trajectory completed registration, current obstacles, mapping and
route queries without failure. It retained the recorded raw poses exactly;
the first 83 fully registered poses also matched exactly. The remaining 110
used the new height schema, and none promoted a new full floor anchor.
Obstacle observations were available for all 193 frames. Post-estimation
native comparison gave median/max position error 9.43/10.49 mm.
Result: `go2_fine_obstacle_round_trip_native_layout00_v1_attempt_001/partial_height_integration_replay.json`.

The prospective full-budget native experiment is launched in
`go2_partial_height_round_trip_native_layout00_v1_attempt_001`, session 75762.
The native collection completed all 1,805 frames and 9,025 policy steps over
181 simulated seconds, with no contacts or registration failure. All 1,805
camera pairs were persisted, with static identity unchanged. There were
393 on-time plans and 57 late plans. The mission exhausted its budget without
an arrival; this is a negative navigation result, not real-time qualification.
Result SHA-256: `0e8327fed7e620068868c4aa11f078ff4ceb4482df59e162d5551d84382661a2`.

The planner chose hold 395 times, left turn 44, forward 10 and right turn once.
Removing contact costs from the recorded candidate scores would still choose
hold in 394 of those 395 cases. Thus reducing the contact penalty would not
resolve this observed stall. At frame 1,000, forward predicts -20.43 mm of
waypoint progress, while hold predicts +1.50 mm. Pure rotation receives almost
no credit from a positional objective, even when needed before useful motion.

The next prospective treatment adds model-predicted reduction of waypoint
bearing error over the actual 300-to-700 ms candidate interval to positional
progress. Its angular scale reuses the existing 0.35 m view-turn scale, capped
by current waypoint distance. The model, contact coefficient, sensors, current
obstacle checks and mission budget remain fixed. Two focused tests passed,
covering useful rotation without translation, exclusion of prefix turn credit,
aligned forward motion and angle wrapping. Native attempt:
`go2_waypoint_alignment_round_trip_native_layout00_v1_attempt_001`.
The alignment attempt stopped after 420 acquired frames and 2,097 requested
policy steps. It selected hold once, left turn 52 times, right turn 41 times,
forward seven times and left arc twice. It did not reach the goal. Alternation
between route-following and `ADDITIONAL_VIEW_REQUIRED` accompanied the repeated
turn reversals. Final observed goal distance was 2.585 m; maximum native XY
displacement was 0.428 m. All 420 camera pairs were persisted. The process
exited with a runtime-stopped exception; no physical contact stop was reported.

Paired public-sensor replay reproduced the 100-feature visual failure at frame
415, after 415 accepted poses. Every replayed 100-feature raw pose matched the
native run's recorded estimator output exactly. Increasing to 150 features
failed earlier, at frame 166. The 100-feature previous-frame attempt reported
`measured-plane pair refinement rejected`; all retained references lacked enough
rigid-pose matches. The 150-feature failure instead rejected rigid consensus,
grid support or displacement. These do not support increasing the budget as a
fix. Post-estimation native comparison gave median/max admitted position errors
11.01/14.71 mm. The failure and paired replay remain in the alignment artifact
directory. Neither the hold reduction nor the replay establishes navigation
success.

An instrumented public-only replay, preserving the original estimator
decisions, isolated the frame-415 rejection to the auxiliary camera's match
against frame 414. The unconstrained image fit had 32 accepted correspondences.
Exact plane refinement lost one original image inlier; the point residual at
that pixel was 0.669 mm, and overall RMS changed from 0.417 to 0.423 mm.
This is not evidence that the original image fit was unavailable. It motivates
testing an explicitly optional refinement that retains a qualified image fit
when refinement cannot preserve its image support, while still rejecting an
actual measured plane/image conflict. No such fallback is implemented yet.
Exact evidence: `refinement_failure_diagnostic.json` in the alignment directory.
