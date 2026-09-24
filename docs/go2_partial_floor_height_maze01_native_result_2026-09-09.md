# Partial-floor-height maze 1 native result

The scoped launcher65703 exited0. Result SHA-256:
32edbb748e04e18816e0b0fb265f465706ac8c8684849d73a091321f79a3b07f.
Root: go2_partial_floor_height_maze01_scoped_verification_pilot_v1_attempt_001.
Independent89804 exited0 and rechecked all1,709 source bindings and3,975 output
bindings. Launcher wall time after admission:2,262.9071428021416 seconds.

All505 common observations,504 prior requested commands and501 forecast banks
match the prospective prefix and the original native run. The first25,950
physics samples and all public packets/state are exact. At observation504,
the changed left arc[0.16,0,0.45] was completely dispatched. No physical outcome
after that change was treated as shared predecessor evidence.

Collection saved656 observations and655 completed commands,33,500 physics
samples. Complete raw replay and strict physical visibility pass; no hard
measurement-failed frames, model unchanged. There are no goal arrivals or maze
cell crossings: the robot remains in cell[-1,0]. No return traversal, terminal
quiet pass or verified round trip. The aggregate is now23 completed/raw-audited
native episodes and zero verified round trips. This reused development maze does
not add an independent layout or establish reliable navigation.

## Correction to the interim failure diagnosis

The first terminal decision retains controller tick644, but occurs in saved
observation645. The controller stops before advancing its accepted observation
counter. The earlier interim inspection accidentally read successful row644
and inferred a possible unhandled partial-height evidence consumer. That
inference is withdrawn. It is not justification for changing an evidence gate.

Row644 has no terminal/failure and requests left turn. Its typed partial-height
pose passes its original accessor after restoring JSON-serialized identity
tuples (32895 exit0). A preceding direct check10420 omitted that restoration and
raised an identity-type error; it did not reproduce the live failure.

Actual failing row645 (91145 exit0) has no registered evidence. Its raw visual
evidence is VISUAL_TERMINAL_FAILURE, current_pose=None, reference selection
NO_QUALIFIED_REFERENCE, continuity MEASURED_BRIDGE_BUDGET_EXHAUSTED. The direct
corner-flow fallback records 'bounded measured bridge exhausted without anchor
observation'. The outer error is 'same-episode current visual evidence required'.
Last successful visual frame644 and the terminal's accepted controller tick644
must not be confused with failing observation645.

Next diagnose why a qualified anchor cannot be reacquired before the existing
measured-bridge budget is exhausted. Preserve that budget and this failure;
do not relax it or infer a physical pose/error bound from the successful prefix.
