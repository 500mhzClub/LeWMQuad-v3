# Direct-flow observer integration and actual prefix preparation

The direct-corner-flow pair diagnostic recovered original rigid fits on three
of four failed camera pairs. This successor integrates that association into
the complete original temporal observer, then tests the full learned controller
from the start of actual independent maze1. It does not claim native recovery.

Only missing measured pose after the original full two-camera pass permits the
fallback. It uses the immediately previous 100 ms reference, retains every
original qualified measurement as a conflict veto, and preserves rigid, gyro,
anchor/increment and bridge limits. There is no reference, pose or gyro reset.
The floor registration, mapping, mission and learned controller consumers are
inherited unchanged. Successful original observations retain exact receipts.

The 11 integration tests cover original success, direct-flow passage through
the original temporal anchor checks, bridge exhaustion, four original conflict
or invalid states, qualified witness conflicts and failure latching. The 19
prefix tests cover exact preceding decisions, changed forecasts/state/commands,
borrowed original failure evidence, current pose plus full-controller admission,
retained terminal negatives, and exclusion of the next recorded observation.
Combined run: **30 passed in 1.94 seconds**, exit 0.

The new prefix stops at observation 214 even when its command remains zero.
Its original decision tick is 213 because the original observer fails before
controller advancement. All preceding complete decisions must match exactly;
at the boundary both raw dual-camera and registered-floor pose contracts are
rechecked if the full controller recovers. Actual input bytes and model weights
are checked before/after, and candidate rows are retained on comparator errors.
See `docs/go2_direct_flow_maze01_prefix_v1_2026-09-09.md` for the fixed protocol.

Source identities at submission:

| Path | SHA-256 |
| --- | --- |
| `lewm/direct_flow_dual_camera_pose_development.py` | `13dc761c424a45a108d573bc6f4869fc4217bb43d7eec3066edd9c5a28411845` |
| `lewm/direct_flow_floor_transport_controller_development.py` | `5cabd93c4cd10ed32624535abcd4dfcc117443ecbc8e8c3202f877c3d49947e2` |
| `lewm/tests/test_direct_flow_dual_camera_pose_development.py` | `c791f6ec9bc49ce5f10a5ad6c308853f6a5d6dfbd9c1c78a481fe851f4004b3a` |
| `lewm/direct_flow_prefix_development.py` | `7660f2175042e704e3a224e7840d427688c12659e6086a5f32f720a51a4d708b` |
| `scripts/replay_go2_direct_flow_maze01_prefix_v1.py` | `f864e126c77f6271fc852e85f40aae528edd48ebb40a79c940dc2067e0126dc8` |
| `lewm/tests/test_direct_flow_prefix_development.py` | `c471e4403fa25ac33e0dc83729c349998c3dd1396603c350b91ca040fe33c9b6` |
| `docs/go2_direct_flow_maze01_prefix_v1_2026-09-09.md` | `d091f24adec3a8ddf035e11cd429c4cd0f726fd86924533d34b1b459b72da36a` |

Submitted session **40411**, PID **2479734**, using the recorded Genesis
interpreter, deterministic algorithms, one Torch/OpenCV/BLAS thread and the
standard development PYTHONPATH. Initial process observation: running,
30.31 CPU seconds, 1,330,671,616 bytes RSS, input authentication still pending.
Output `go2_direct_flow_maze01_prefix_v1_attempt_001` is exclusively owned by
this attempt. Preserve its sources and poll this handle; do not restart it.

Before submission: 16 physical/32 logical CPUs, all 32 in affinity, CPU 3.3%,
81,083,490,304 available RAM bytes, 88,894,181,376 free artifact bytes,
21,359,779,840 free workspace bytes. Both GPUs 0% busy; discrete GPU
34,208,743,424 total VRAM bytes, 1,398,722,560 used. This permits one CPU replay
alongside the separately owned fixed reactive cohort; it does not justify a
second native scene. Native planning-memory comparison still follows that
cohort, then the already prepared residual-feasibility native trial.

Launch admission subsequently passed. Launch SHA-256
`4d9900f40d39eb40d7aa1740b99838756b7dc48cb595fafacae5c8b01fc72d2f`,
1,668 source bindings and 6,201 completed cohort artifact bindings, plus
upstream input verification. Refreshed launch hardware: 80,470,720,512
available RAM bytes, 88,893,059,072 free artifact bytes, CPU 3.4%, both GPUs
0% busy. Replay reached frame 32 without a reported mismatch. This is live
progress, not a completed prefix result.

## V1 terminal validation failure and separate V2 correction

40411 subsequently exited 1 at the boundary's extra live-pose validation.
All 214 preceding decisions were exact. The controller returned an accepted
auxiliary anchor pose and left-turn command at frame214, but the runner passed
its serialized identity list to a checker that requires the live tuple.
Final input/model checks were not reached. V1 remains a terminal validation
failure, fully preserved in
`docs/go2_direct_flow_maze01_prefix_v1_failure_2026-09-09.md`.

V2 keeps the live controller return for live checks and separately verifies
that its serialization exactly equals the recorded decision. The observer,
controller, model, comparator and scientific rules are unchanged. Five focused
tests pass in0.17s, including reproduction of the actual tuple/list rejection,
both raw and registered checks, altered serialized predictions/commands/identity
rejection, and preservation of negative controller results.

| New path | SHA-256 |
| --- | --- |
| `lewm/direct_flow_live_replay_validation_development.py` | `022c2922ede97fbdddb5f46d729a29f94b0954b4242fbbd65312cece01243639` |
| `lewm/tests/test_direct_flow_live_replay_validation_development.py` | `bc3cce7e876cd01fed5a03b8e599550493510658bc7b8a9aae63bdf01f1571bd` |
| `scripts/replay_go2_direct_flow_maze01_prefix_v2.py` | `09835e213d9a0a99709424ee2275fea8144b8a1363d5388e60515f59573bcf0e` |
| `docs/go2_direct_flow_maze01_prefix_v1_failure_2026-09-09.md` | `aaee6f1d2c9638b23a9aae0e22d4b868c1fcb9d7f8ef182811c9e00593aecba4` |
| `docs/go2_direct_flow_maze01_prefix_v2_2026-09-09.md` | `1b1ce0267fa596d530c1b9cc5a5cf7fdca97d8c6e7316b5b3b7c3975ee02369f` |

V2 submitted as session74784,PID2481351. Last verified running58.61CPU seconds,
RSS1,353,830,400bytes; input authentication pending. Preserve this new attempt
and all frozen source identities. Refreshed80,109,371,392availableRAM bytes,
88,455,069,696artifactfree bytes,CPU3.4%,GPUs0%,all32CPUaffinity. Same single
CPU replay concurrency; reactive maze3 is raw-audited but its worker and parent
still finalize before the next native scene may launch.
