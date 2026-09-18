# Unchanged tracking implementation on maze3: prefix submitted

Implemented a separately bounded replay for original learned maze3 through its
first visual failure at observation264. It uses the same frozen observer, direct
flow association, rigid/temporal thresholds, reference history, bridge budget,
controller, assigned model and live-contract validator as the completed maze1
tracking prefix. This tests transfer to the other already observed development
failure without tuning the method or following a changed action.

The new comparator is identical to the frozen maze1 source except its fixed
boundary264 instead of214. The replay's executable AST matches maze1 V2 apart
from the scene index, output labels and explicit layout identifier. Tests also
cover complete prior decisions, actual command agreement, early state/input
changes, model integrity, recovered and rejected boundary outcomes, live-contract
failure retention and exclusion of observation265. Twelve tests passed in2.34s,
session91305 exit0.

Submitted session47935, PID2493328. First live inspection:54.51 CPU seconds,
RSS1,330,827,264bytes, launch/result/failure absent during authentication.
Preserve the same handle and four new sources. Output, when admitted, is
go2_direct_flow_maze03_prefix_v1_attempt_001. The positive completed maze1 V2
result and all its artifacts/sources are authenticated, alongside original cohort
inputs and preserved historical maze1 V1 runner-validation failure witnesses.
The old V1 failure is not restarted or reclassified.

Hardware before submission:16physical/32logical CPUs, all32 affinity, CPU6.5%,
71,174,803,456bytes available RAM,79,628,013,568artifact free bytes,
21,359,366,144workspace free bytes, both GPUs idle. Discrete VRAM used
1,398,722,560 of34,208,743,424bytes. The sole residual native worker used
10,335,842,304bytes RSS and the independent paired CPU benchmark2,808,856,576.
The planned native32GiB, paired-replay16GiB and this replay8GiB allowances total
56GiB, below available memory. They are admission estimates, not enforced quotas;
each job refreshes resources before execution. No second native scene is launched.

Source SHA-256 identities:

- `lewm/direct_flow_maze03_prefix_development.py`:
  e60a95473f19aaa4dadd59cfd48c7f2d5a39e255df78e0aef40b4fb405b1c912
- `scripts/replay_go2_direct_flow_maze03_prefix_v1.py`:
  b4d774dd06fa3fb278179aec6c7c90929831ac2c6d8cd3bb63cf2586427df8b6
- `lewm/tests/test_direct_flow_maze03_prefix_development.py`:
  33a1f37424b7103e1217b590df877a2f5f7741c821651f595f8e8f0599c1c1c8
- `docs/go2_direct_flow_maze03_prefix_v1_2026-09-09.md`:
  a2f9c0bcf841cc6511e8ec2f69a9ff2b65d49d863d195943ad90b35a3dfe8b75

No maze3 recovery is yet established. A successful prefix would justify assessing
a fresh physical continuation; it would not establish a new arrival, round trip,
independent generalization, timing result or hardware qualification. The native
queue remains residual maze2, tracking maze1, then supervised layouts1–3.
