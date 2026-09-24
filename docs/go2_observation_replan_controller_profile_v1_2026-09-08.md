# Observation-replan controller profile V1

Profile the complete recorded first-seed full-direct case from the completed
observation-replan probe, whose result SHA-256 is
`8cdfd800eda961de5b1a58a3b64fe8fd471c0f8c8165aae9c61f699c500010eb`.
This is a diagnostic of the recorded controller's computational cost while
the separate short-horizon fits proceed. It does not choose or change any
model, controller, action, target, gate or scientific outcome.

Verify the completed probe's complete artifact and recursive source bindings,
its stored full-model admission, the assigned final snapshot and URDF. Load a
fresh evaluation-only copy of `seed_2026091001_full_direct`, replay every public
packet of `full_direct_family_episode_039`, and require exact JSON-equivalent
decisions and unchanged model state. Enable cProfile only around each controller
observe call, excluding packet loading, admission and validation. Keep the
original controller and its callees unchanged.

Freeze this script/protocol and inherited sources before the exclusive
`go2_observation_replan_controller_profile_v1_attempt_001` root. Save all function
call counts, self and cumulative times, per-observation profiled durations and
before/after hardware readings. Reverify original inputs and sources afterward.
Preserve any failure. One CPU process and one numerical thread, with at least
8 GiB available RAM and 256 MiB output allowance above the 40 GiB storage reserve.

Instrumented replay under concurrent training is not an uninstrumented latency
benchmark, a speedup measurement or a real-time qualification. Cumulative times
overlap and must not be summed as disjoint components. This diagnostic does not
profile native rendering or physics, launch a native episode, train a model,
alter a completed attempt or provide new maze-navigation evidence.
