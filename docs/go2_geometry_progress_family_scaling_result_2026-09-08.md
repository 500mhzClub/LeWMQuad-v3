# Four-process native collection comparison: exact records and 3.34x speedup

The separately frozen serial/four-process comparison completed all eight native
episodes and raw audits. Four concurrent fresh processes took50.601s versus
169.124s serial, a3.3423x total speedup including startup, source/input checks,
recording and raw audit. All816 compared array/pixel signature fields agree
exactly across the four matched cases. No hard measurement failure occurred.
The frozen rule selects four workers for the separate96episode family dataset.
Benchmark episodes remain excluded from training and transfer evaluation.

Fixed cases: `family_episode_052`, `family_episode_039`, `family_episode_083`,
`family_episode_092`. These cover the two training parameter clusters, both
openings and left-arc/forward/hold/right-arc actions at appearance2026090940.
Each serial and parallel execution used the same actual specification and
existing checkpoint gait. Each process handled one episode then retired.
Every episode also completed the unchanged physical-first-surface, raster,
sensor, exact-command, setup/contact/stop and native-gain/friction checks.

This comparison qualifies a prospective scheduling choice for this local
simulation pipeline. It is not RGB/JEPA benefit, calibrated risk, real-time
execution, independent-maze navigation or hardware qualification. Physics paused
during processing. The overall navigation goal remains active.

Preflight inspected16 physical/32 logical CPUs, about82GB available RAM,
103GB free artifact storage, GPU activity/VRAM and competing jobs. The benchmark
reserved2GiB planned storage above a40GiB floor and required32GiB available RAM.
All execution/source/test/protocol identities were frozen before launch;833
source paths were bound.23 new focused tests passed before launch, including
constructor/command/audit equivalence, new role/cluster tensor admission,
full-denominator accounting, exclusive worker claims and actual spawned-process
retirement. The pre-existing five layout-family tests also passed earlier.

All roots below are under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.
Every benchmark process is terminal. No retry or resumption.

| Root | File | SHA-256 |
| --- | --- | --- |
| `go2_geometry_progress_family_scaling_v1_attempt_001` | `launch.json` | `c31e78806a24b4250c99b5a9ffcd14ee624d3cab43aa1b288d8efc5907773b66` |
| same | `result.json` | `ed64dfb610062caea529f96db01d3ce5a3be2c621a5ca8bcd04822f5c1523969` |
| `go2_geometry_progress_family_scaling_serial_v1_attempt_001` | `result.json` | `22d2f5ec8c27932162f7c33119e861dbfe90c37d90674440031430fda983061a` |
| `go2_geometry_progress_family_scaling_parallel_v1_attempt_001` | `result.json` | `23064c59e7c06b068a94ffad9882ea98f82abb22303261f26e4f02a61452f513` |

Implementation: `scripts/probe_go2_geometry_progress_family_scaling_v1.py`,
`scripts/geometry_progress_family_runtime_development.py`,
`scripts/geometry_progress_family_episode_development.py`,
`scripts/geometry_progress_family_session_development.py`,
`scripts/geometry_progress_family_audit_development.py`. The subsequent
`scripts/run_go2_geometry_progress_family_v1.py` was already source-bound by this
bench and accepts the exact terminal bench SHA256 explicitly. Later-added causal
derivation and learning-view/stream files are not part of this execution freeze.
