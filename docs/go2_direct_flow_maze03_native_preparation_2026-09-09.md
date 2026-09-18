# Fresh maze3 tracking simulation prepared

The completed maze3 replay recovered the original tracking failure at observation
264 with the same DirectFlowFloorTransportController used for the maze1 replay.
All264 preceding decisions/commands and261 raw prediction banks matched. That
evidence ends before executing the new left-turn command; fresh closed-loop
continuation remains necessary. This preparation implements that experiment.

Added a separately named maze3 launcher, collector, physical-prefix comparator,
protocol and focused tests. The collector changes only the experiment's status
labels from the frozen maze1 collector. The original layout-parameterized maze1
raw audit is imported directly and receives layout3. No tracker, model, mapper,
selector, residual, calibration, rigid/temporal gate or evaluation change.

The launcher authenticates completed learned cohort
a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720
and maze3 tracking prefix
104edd70824e7d4e7b909be924eb68a918542efd3464ff3247f9570ddc28669e,
including inherited source/environment bindings and preserved tracking failures.
It admits all265 saved prospective decisions and checks the fixed maze3/model
assignment. Each collection and raw audit receives a fresh identical assigned
model. Actual scientific failures remain results; infrastructure failures preserve
raw artifacts and any audit already completed.

Physical comparison requires13,950 samples before the changed command, all265
paired public observations,264 preceding actual requests and261 raw forecast
banks. Every new decision through264 must exactly equal its saved prospective
decision. Do not compare following old physical outcomes. An incomplete changed
command remains an actual negative result rather than a reason to discard files.

28 tests PASS in2.99s, session56219 exit0. Tests verify unchanged native collection
calculations by AST comparison with the original collector, direct audit reuse,
fixed source/result assignment, fresh model owners and preserved evidence after
audit/prefix/verification failures. Synthetic physical-prefix cases reject altered
physics, public inputs, forecasts, prior commands and insufficient history while
excluding all later decisions/packets and changed-command physical outcomes.

Six reviewed source SHA-256 bindings:

| Path | SHA-256 |
| --- | --- |
| scripts/direct_flow_maze03_episode_development.py | fa92dd286bab520e217d0e68ab4a3c613ff03d64f8a69672033d9b0a3ef8c9c0 |
| scripts/direct_flow_maze03_native_prefix_development.py | 44d0eebf519b42e341d86963b1953ff4b48a0e2351bb058f2e9623ed1f601354 |
| scripts/run_go2_direct_flow_maze03_pilot_v1.py | 92d40f78f6d3d5ee1cba9b76ed4e90d8cea1554451fc1754baf10c98ce4cbb30 |
| lewm/tests/test_direct_flow_maze03_native_development.py | e244d14301861656a70dcbb4a7f26d952f855cc2ed32a364392268126084c038 |
| lewm/tests/test_direct_flow_maze03_native_prefix_development.py | 6962f17d6e99c3a857169974b4cd737004f84ea58b236be5395eda3f2002ed79 |
| docs/go2_direct_flow_maze03_pilot_v1_2026-09-09.md | 36520dae7f6f1e64250571c3274f805043b87a22dea16cc658b20b04dfbf5874 |

Directly reused audit source:
scripts/direct_flow_maze01_audit_development.py,
f2dd53cbaef075b7dc13dc5c8c41ba2c04535f87f80d1e7f8a0ff0c6e3102991.

Hardware check45523 exit0:16physical/32logical CPUs, all32affinity, CPU3.4%,
72,559,284,224bytes available RAM, GPUs idle,78,116,974,592bytes artifact free,
21,359,198,208bytes workspace free. Existing native worker2485788 occupies
10,758,172,672bytes RSS. One CPU-only preflight fits beside this single native
scene. These measurements are capacity evidence, not runtime qualification.

Submitted read-only preflight14145, PID2498090. Last confirmed live at66.34 CPU
seconds and1,216,827,392bytes RSS. Its full result remains pending at this entry;
do not restart or edit its six reviewed sources. It must return before model,
worker, scene or output creation. Exclusive future output is
go2_direct_flow_maze03_pilot_v1_attempt_001. The following command includes
--preflight-only; actual native execution is a separate later step:

```text
scripts/run_go2_direct_flow_maze03_pilot_v1.py --learned-cohort-result-sha256 a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720 --preflight-only
```

Queue remains current residual maze2, tracking maze1, then supervised fixed
layouts1–3 after its storage gate passes; tracking maze3 follows. No native
concurrency increase. At this check residual parent2485335 and worker2485788
remain live, without final result/failure. Supervised all-three admission still
requires78,383,153,152bytes and currently does not fit. No deletion was performed.

Reliable navigation, independent-layout success, physical return, prediction/
JEPA/memory advantages,100ms execution and hardware deployment remain unproven.

## Completed preflight

14145 exited0. All1,685 source/input bindings and the actual completed265-frame
prefix pass. Both memory and storage admission pass; no output or native
execution. Final71,991,451,648bytes available RAM,78,114,082,816bytes artifact
free,21,359,185,920bytes workspace free,CPU3.3%,GPUsidle,all32affinity. The sole
existing residual worker remains live. The six reviewed sources are unchanged.
This experiment is prepared and preflighted, still queued for later execution.
