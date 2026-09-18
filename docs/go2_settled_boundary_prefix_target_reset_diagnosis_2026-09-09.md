# Settling prefix failure: target reset omitted from comparison

The first full settling-controller prefix is terminal, exit1 at frame1866.
Launch cf320b97b50fd5e9d6369608d690147cdb1cce9bb15a19d54cfb89bc126f9f1e;
failure f7dedb15dabba46667624a63434b4ba0bcf01d7b294c501dc19a095bf5f138ef;
mismatch 009585431572ab6742eac10d239a8a580dc0a5817d8fd79fa15b4242b3368731.
The full candidate at the failure and partial compressed decision stream remain
under go2_settled_boundary_controller_prefix_v1_attempt_001. No completed result
or final model/source/input validation was produced. Do not resume this attempt
or relabel it as successful.

Comparing the actual saved original and candidate at1866 found only the declared
settling/mission/controller-label fields plus planner_mode. Original mode NEW
differs from candidate WAYPOINT. MissionTargetWaypointSelector.set_goal resets
the mode when the original mission changes its target to home. The candidate
keeps its outbound goal and prior mode while continuing to hold. Both commands
are exactly zero and neither current decision selects a new action. The
candidate has eight measured quiet intervals and no arrival at that frame.

New lewm/settled_target_reset_prefix_comparison_development.py checks this
effect against the immediately preceding candidate decision. It requires
consecutive timestamps/frames, unchanged candidate outbound goal and mode,
the original confirmed OUTBOUND_TO_RETURN transition to home with NEW mode,
and held zero commands with no selections in both current decisions and the
preceding candidate. All other comparisons delegate to the frozen V1 comparator.
It does not broadly suppress planner-mode differences.22 focused tests passed
in1.80s, covering the actual target setter and rejection of unrelated or
unwitnessed changes. Saved-mismatch check79563 completed: frame1866 admitted
only phase/active-goal behavior differences plus the witnessed target-reset
mode effect; requested_command_changed=False. All1555 V1 source bindings were
rechecked before and after this saved-artifact check.

The controller, observer, mission, map, model, contact checks and numeric settings
are unchanged. A fresh complete replay is running under
go2_settled_boundary_controller_prefix_v2_attempt_001, session17684,
PID2412337. Launch e36fa13e0c1bfb58868e9d3e5d2d5e4bfac2b5cb9a8808fd45023f790488c7d5
binds1559 sources and the original failed attempt's launch, failure, mismatch
and partial stream. Preflight8792 passed. Actual launch:77,569,933,312 bytes
available RAM,118,114,914,304 artifact-free bytes,3.6% CPU utilization. One
CPU replay and one numerical thread beside packed-owned replay94114. GPU/VRAM,
CPU topology/affinity, competition and both volume capacities were recorded.
Capacity admissions remain8GiB RAM and1GiB output above40GiB reserve.

The prepared native V2 launcher uses the same collector, raw audit and worker
body as the unlaunched V1 preparation. Only the predecessor admission and
target-reset comparison are revised, with explicit new result fields/status.
The native comparison also requires the preceding candidate witness and all
physical/public/command/prospective-decision equality through1866.37 tests
passed in2.40s; the newly added readout derivative test passed in0.12s, for38
preparation tests total. Readout computation is unchanged. Source check69911
verified the running1559 bindings and1571 explicitly prepared native/readout
bindings. Native/readout output roots have not been created.

Next: finish the fresh1867-frame replay and final source/input/model checks.
Then invoke scripts/run_go2_settled_boundary_maze_pilot_v2.py with its actual
--prefix-result-sha256 and --preflight-only; launch only if the completed
result and resource checks pass. The old V1 native launcher cannot consume V2.
After native collection and full audit complete, use
scripts/read_go2_settled_boundary_maze_pilot_v2.py with the actual native hash.
The independent packed-owned full replay remains pending and is not adopted
by this settling experiment.

This diagnoses a comparison omission; it does not establish a successful
arrival or return. Nine native outcomes remain preserved, with zero verified
arrivals/round trips. Return tracking, strict visibility, independent layouts,
matched baselines/ablations, real-time execution and hardware remain unfinished.
