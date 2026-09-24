# Reusing optimized floor processing inside visual tracking

A separate controller composition replaces the dense floor candidate extraction
inside the measured-plane visual tracker with the existing tiled/density-routed
implementation already used by floor registration. It retains the original
image fits, plane coherence gates, retained camera references, temporal checks,
map, planner and stopping mission. Existing experiment runners are unchanged.

The original and candidate controllers replayed actual frames 0–12 of the
current stopping-rule trial, alternating execution order. Every complete
decision matched the recorded run, after removing only the candidate's explicit
implementation flag. The assigned model and consumed sensor files stayed
unchanged. No profiler was active.

| Ten active decisions, frames 3–12 | Original | Candidate |
| --- | ---: | ---: |
| Total controller time, s | 5.692897 | 4.989831 |
| Median controller time, ms | 539.458 | 476.858 |

The measured total reduction is **12.3499%**. This is an early-prefix shared-host
measurement, not proof of a population speedup or real-time execution. Warmup
frames 0–2 were replayed and compared but excluded from the timing aggregate.
The comparison completed successfully in session 66886.

The longer tracker-only comparison is now running over observations 0–4739,
through the recorded outbound and return arrival. It compares complete public
tracker results and stops at the first difference or tracking failure. It loads
no learned model and runs no map, planner or simulator. Its purpose is to cover
turns and retained camera-reference behavior before considering adoption; it
cannot prove full-controller equivalence or a new navigation outcome.

At launch it had about 66 GiB of available RAM. It shares the host with the
current raw audit and produces small timing records. Owner: PID 3142817,
creation time 1789258300.55, session 5898. Output root under the existing
development artifact directory:
`go2_tiled_plane_tracker_recorded_comparison_v1_attempt_001`.
The original audit and the queued independent JEPA/reactive cases are unchanged.

Implementation: `lewm/tiled_plane_stop_conditioned_controller_development.py`.
Short comparison: `scripts/compare_tiled_plane_early_decisions_development.py`.
Recorded timings: `go2_tiled_plane_early_comparison_2026-09-13.json`.
Longer comparison: `scripts/compare_tiled_plane_recorded_tracker_development.py`.
