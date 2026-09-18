# Bounded reobservation replay: one proposed recovery, one bounded failure

Both fixed corrected models reproduced exactly in two fresh replays with
unchanged model state and no controller failure. Every original decision field
except the controller name matched before the terminal-policy intervention.

JEPA first changed terminal policy at tick 38, requested one zero-command wait,
and found an originally admissible right arc at tick 39. That proposed command
[0.16, 0, -0.45] differs from the actual zero tape. Replay stopped immediately
after this 40th observation. The command was not executed; native recovery and
goal-reaching remain unproven.

Direct first changed terminal policy at tick 35. It requested ten zero-command
waits at ticks 35 through 44, then retained the original no-feasible-action
terminal at tick 45. All requested commands matched the available recorded tape.
The final observation's zero proposal had no subsequent recorded command. Replay
therefore reached the 46-frame input bound with no recovered action.

The observer, complete primary/auxiliary map, model input, learned weights,
training-only correction, selector and all surface/nominal constraints were
unchanged. This replay supports a prospective test of bounded reobservation;
it does not establish safe waiting, native recovery, independent navigation,
real-time operation or overall goal completion.

Root: `go2_auxiliary_depth_reobserve_prefix_v1_attempt_001` under the guarded
development base. The run bound 1,252 sources and took 109.140069 seconds after
admission. Six focused state-policy tests passed before launch.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 0d3a13c58b5e93512eaf5c2058f4f11e2241c7a0388b3c2f4feb48d56a4fd070 |
| JEPA decisions | 5f698df7fd974974287af25f0620ab17e3ba3241388aced066a9f6de4028aac8 |
| direct decisions | 8b7744c14d746e9b7b31324bd4f96976709073029a8e65bfc8868ea6ad104a5c |
| result.json | 28ba801b94c8eb7aaec9ac4c596b78ef37bafc350dfaff2e90e082f66fc6c55b |
