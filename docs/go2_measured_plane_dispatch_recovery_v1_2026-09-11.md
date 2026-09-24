# First measured-plane native execution after terminal scheduler failures

The extended-budget native waiter completed with result SHA-256
`1e2da182445446b05de79c09825979a0e4547212d3bd1db5ce999fca4928fc70`.
It verified native result
`c92f0bdf5cc8ebb9e513470492578ec3457d02196dfc7238e2f6b12b9c621b27`:
outbound success, zero recorded contacts, incomplete return after floor
registration failure, and zero verified round trips.

The next sustained-turn waiter stopped before child dispatch because the
existing conservative idle predicate matched the read-only deferred-memo
replay's `run_go2_*.py` name. The replay was still live at that point. The
contact/flow and chained-anchor waiters subsequently stopped for missing
predecessor completions; the measured-plane waiter stopped on the absent
chained-anchor result. This was a scheduling error, not four scientific runs.

`scripts/native_waiter_dispatch_abort_development.py` binds the exact four
launches, terminal failures and closed event streams. It requires the original
owners ended, no child-start event, no native stdout, no success result and no
original child output directory. It retains all four failures and the completed
native predecessor. No old waiter is restarted, overwritten or relabeled
complete. The sustained-turn, contact/flow and chained-anchor physical
diagnostics remain unexecuted and provide no navigation evidence.

The separate `scripts/run_go2_measured_plane_dispatch_recovery_v1.py` prioritizes
the first measured-plane physical episode because the completed 3,838-frame
observer and 123-frame controller comparisons support its prospective
perception intervention. It does not infer outcomes for the three unexecuted
diagnostics. This is an explicit new scheduling path under the standing
navigation goal, with a new root:
`go2_measured_plane_dispatch_recovery_v1_attempt_001`.

Science remains exactly that of
`scripts/run_go2_measured_plane_maze02_pilot_v1.py`: the same measured-plane
controller, no-RGB direct corrected model, maze 02, 4,000 navigation ticks,
public sensors, original physical limits, settling/return mission, strict
visibility gates, raw sensor/model/command/native audit, physical prefix and
readout. The worker, prefix and definition reuse the original function code
with only explicit output/protocol/admission bindings. The optimized combined
controller and comparator methods are not adopted in this native experiment.

Require successful completion and ended owners for both the deferred-memo CPU
replay and its checker before admission. Authenticate the original model and
full raw worker artifact roster; preserve source and binary bindings. Require
32 GiB available RAM and 55 GiB artifact free space. Use one fresh spawned CPU
scene worker, original single-thread deterministic settings and renderer
environment. Recheck admission in parent and worker and after auditing.

Before creating the new output, wait while the existing competitor predicate
reports an occupied slot. An occupied slot does not create a failed native
attempt. Preserve observation errors, do not infer completion from time, and
perform a final idle check before dispatch. No failed original run is resumed.

Retain every physical outcome, including an early negative before frame 122.
If reached, compare the same 123-observation/6,850-sample physical prefix and
require actual completion of the first changed command at frame 122. Evaluate
following navigation from the fresh physical trajectory and full raw audit.
Physics still pauses during controller computation; no real-time or hardware
claim is permitted. This reused development layout establishes no independent
maze reliability. Preserve any new failure without retry or overwrite.
