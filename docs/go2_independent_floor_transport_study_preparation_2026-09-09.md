# Three-layout native study prepared while maze0 executes

Implemented the fixed cohort launcher
scripts/run_go2_independent_floor_transport_mazes_v1.py and protocol
docs/go2_independent_floor_transport_mazes_v1_2026-09-09.md. It uses the same
frozen collection/controller/raw-audit implementation on layouts1,2,3 in order,
fresh process/model/controller/memory per case, one scene at a time. The wrapper
changes only the scope metadata of each returned audit. It preserves all
physical calculations, strict visibility and verified-round-trip outcomes.

Admission requires the actual completed result from the currently executing
go2_measured_floor_transport_maze_pilot_v1_attempt_001. It requires complete
raw audits and the exact prospective maze0 prefix, while allowing a negative
scientific outcome. All three layouts execute without controller/model changes
or outcome-based omission. Infrastructure/resource failure stops the partial
cohort without retry. Each case has an immutable progress snapshot and artifact
bindings. Aggregate completion requires all three raw-audited cases, not success.

Testing12167 CLOSED:30passed/2failed. The two failures exposed reuse of the
exclusive-create cohort_progress.json path. Corrected the new, unlaunched
runner to write cohort_progress_after_01.json,02,03 with separate hash bindings.
Testing9277 CLOSED:32passed2.63s. Checks include negative predecessor admission,
raw/prefix/model/controller/acquisition failures, exact physical-result
preservation, resource allowance for every remaining case, three-case negative
result accounting, partial-cohort persistence without retry/skip, nonexecuting
preflight, fresh spawn configuration and separate collection/replay models for
each layout. These synthetic orchestration checks create no native scenes.
CLI9048 CLOSED successfully with --help.

Source preparation29649 CLOSED:1658paths, including all1654live native source
bindings unchanged. No independent output root, native preflight or new scene.
New prepared source identities:

- lewm/independent_floor_transport_study_development.py:
  6b78ad6355053d272d1009c1c76379a24d5866f77133842db215046243843a14
- scripts/run_go2_independent_floor_transport_mazes_v1.py:
  fe8bb40eb9cc56dfca0de01b9b129765993dfa6c9de7ba9d673650c296c7eccd
- lewm/tests/test_independent_floor_transport_study_development.py:
  cffce55a7b70b66163d24af0bee5e33f37dbb6c4feab8eea1436c2eb8b558d27
- docs/go2_independent_floor_transport_mazes_v1_2026-09-09.md:
  4f7b7c034bf3452ff78f443c8a77c7daa6e9a2d1af7e51398e452f659482b715

Current maze0 native launch:
8dbd36ce4c300bef2b42b31cb6c04d5633624163684f1a5e88de5fcbb8002369.
Session19976, parent2446755, worker2447506. At inspection the worker was
live98.6%CPU,6m21s elapsed/6m16s CPU, RSS3,142,156KiB;242completed timing
rows through tick241. No terminal result or failure file. Latest monitor:
79,677,345,792bytes available RAM,103,683,518,464bytes artifact free,3.3%CPU
busy, GPUs0%busy. This is a live execution, not a completed twelfth result.

Next: follow19976through collection/fullrawaudit/prefix/finalbindings; run the
existing measured-floor-transport readout using its actual completed result
SHA. Then inspect hardware/competing scenes and run the new three-layout
launcher's --preflight-only with that actual --native-result-sha256. Execute
the admitted cohort after the current native scene is gone. The new runner
refreshes hardware after input checking and before each case. It reserves the
original40GiB plus11GiB for every remaining case and requires32GiBavailableRAM.
No matched-baseline, independent-layout success, timing or qualification claim
has been established by this preparation. Full goal remains active.
