# Translation coverage and measured-view experiment

The preceding exposed failure retained an old wall face but lacked the reverse
face near its end. The map reconstruction and saved footprint probes are in
`docs/go2_observed_wall_face_clearance_diagnosis_2026-09-17.md`.

This experiment adds one coverage-aware translation/view treatment to the
preceding no-early-release runtime. A selected translation is rejected if its
predicted 0.48-m swept disk introduces floor cells absent from both observed
floor and the current/predicted-hold footprint baseline. That baseline is not
declared free or safe. Pure-turn eligibility and all existing obstacle, reserve,
stopping and actual dispatch limits remain unchanged. The first replacement is
an existing eligible hold or turn; the following plan requests a view of a
missing cell using the existing camera-viewpoint planner and known-floor routes.

The view request resolves through an actual floor or obstacle observation.
Alignment alone is not evidence of observation: the existing viewpoint code
can try another known-floor position when a fresh aligned view remains unknown.
Weak visual-support recovery takes priority. An unreachable view remains an
explicit unresolved condition; no missing region is silently marked free.

Six focused tests passed, covering observed translation, preservation of pure
turn eligibility, blocked-turn eligibility, swept-path interior cells, shared
hold motion, visual-recovery priority and observation-based resolution. The
same fixed model, six candidates, sensors, timing, CPU allocation and 4800-tick
budget are retained. The existing arc fallback and disabled early-release rule
remain in both reference and intervention. This is one exposed-layout run;
no fresh-layout, causal whole-trajectory improvement or JEPA advantage is implied.

Launcher: `scripts/run_go2_coverage_translation_view_development.py`.
Plan: `docs/go2_coverage_translation_view_plan_2026-09-17.json`.
Root: `go2_coverage_translation_view_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.

Completed startup-repair and survey-transfer success depth was retired after
reviewing their completed comparisons and superseded debugging role. Reclaimed
4,829,302,784 allocated bytes; 5,368,037,376 bytes free afterward. All non-depth
evidence, the full host-clock and publication-fix success references, and every
failure recording remain. Exact depth replay of the retired successes is no
longer available; their retention inventories record the change.

Completed: owner exited zero; independent evaluation found no arrivals, no
disallowed contacts, and budget exhaustion at 480.90 simulated seconds.
Tracking supplied all 4805 poses (maximum registered position error 2.081 mm).
1179/1200 plans were on time. The intervention did not solve navigation.

The new filter rejected 1174 translations. All requests concerned the same
unknown cell `[8,-7]`; 1172 plans requested travel to a viewpoint, with no
actual coverage resolution. The final chosen viewpoint was `[.925,.025]` in
map coordinates. Travel to obtain its view was itself repeatedly rejected by
the coverage filter. Actions were 1151 holds, 48 pure turns and one right arc.

`scripts/read_go2_coverage_view_stall_development.py` reconstructed the first
101 recorded map updates and checked 74 early request states. Recorded floor
and fine-obstacle counts matched; delivered noisy-depth digests were verified.
In all 74 states, calibrated projection supported viewing the patch by turning
at the actual measured current position, with no stored obstacle blocking the
camera-to-patch ray. This is a visibility hypothesis, not a measurement or an
executed alternative. No native pose or wall geometry entered that diagnosis.
Receipt: `coverage_view_stall_diagnosis_v1.json` in the failure root.

Next experiment prefers that in-place view before requesting translation to a
distant viewpoint. It retains all existing movement and coverage guards and
requires actual mapped evidence to resolve the request. Full failure depth
and all other evidence remain retained.

The saved frame-560 check rejected forward in favor of hold and selected target
cell `[39,-14]`. The existing camera planner found a nearby current-floor
viewpoint with predicted auxiliary-camera coverage. This query used the current
floor cell, not a reconstructed full historical route; it is activation and
projection evidence only. Receipt: `saved_coverage_view_activation_v1.json` in
the preceding failure root.

Launched in session 94471, owner PID 4091385, CPUs 8–15,24–31; archival and
owner exit completed before evaluation.
