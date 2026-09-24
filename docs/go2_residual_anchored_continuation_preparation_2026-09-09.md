# Anchored continuation V1 preparation

Implemented a separate controller retaining original no-action and first-point
hold recovery, with a new later-path anchoring candidate only after both remain
at feasible hold. Full scientific definition, exclusions and prospective stop
contract: docs/go2_residual_anchored_continuation_prefix_v1_2026-09-09.md.
The completed negative predecessor and veto evidence are recorded in
docs/go2_residual_hold_negative_and_veto_result_2026-09-09.md.

23 component tests passed in 2.15 s (44835 exit 0). They use the actual original
planner and synthetic occupied cells to distinguish first-point-only recovery
from a later-path recovery; retain severe later collisions, contact cost,
surface/phase vetoes, raw residual targets, input ownership and failure latching.
Prefix tests initially found an incorrectly renamed predecessor status string
(25265: 1 failed, 40 passed). Corrected before freezing; 78387 then passed 41.
Added missing-segment, radius-reduction, hidden-veto and disconnected-path
rejections, plus a separate anchored intervention count. Final 86237 passed
45 tests in 4.15 s. Tests cover complete causal state and raw forecast retention,
truthful corrected horizons, old first-point precedence, terminal/limit/first
changed command boundaries, mutated input/model rejection, truncation and
exact completed predecessor admission. Total final focused coverage: 68 tests.

Hardware 10226 exited 0: RAM 80,594,472,960 bytes available, artifact free
73,061,388,288 bytes, workspace free 21,358,485,504 bytes. CPU 6.5%, 16 physical
and 32 logical CPUs, all affinity. Both GPUs idle; card 1 VRAM 1,398,722,560 of
34,208,743,424 bytes. The existing native parent/worker were the only substantial
competing Python workload. Replay16 + conservative native32 GiB fits available
RAM; 2 GiB output plus the 40 GiB reserve fits. No new native scene is requested.
The runner refreshes these resource checks after full input authentication.

Frozen sources:

| Path | SHA-256 |
| --- | --- |
| lewm/residual_anchored_continuation_development.py | 7360e27e6c24444ed3149350c371cf81ce2baeb3005663d7ecb2594cf7bae5b2 |
| lewm/residual_anchored_continuation_controller_development.py | f35ff18c78b4db81c0c3c766eed1015823bf972d484600907954941ef7fb946a |
| lewm/residual_anchored_continuation_prefix_development.py | c537724001973ac539a2bde98d7b72ffd8f3094c11e3616efa78e9d2fe228763 |
| scripts/replay_go2_residual_anchored_continuation_prefix_v1.py | 037ed494447c06db6d92822fef2505ff94cfe1b176dfbe262a563272036ff9e9 |
| lewm/tests/test_residual_anchored_continuation_development.py | 465139e6f8769f7f389dd0b2db9bb3a8b136a21ac9db8dcbb30ef091b820ef60 |
| lewm/tests/test_residual_anchored_continuation_prefix_development.py | b21251b25c8e2eeb56c9db7e3d4cda5616ea395a04be17811cddd92a2ddd9ea1 |
| docs/go2_residual_anchored_continuation_prefix_v1_2026-09-09.md | b4f324359338ac32e51412ce2557ad9877daf2409210d2bfd2b0dd69a1072c84 |

Submit the runner in the existing Genesis environment with one numerical-library
thread and --native-result-sha256
55a7d5071f39337b3c9ea329e5b48320f11c8a5a9ba6e34296926006768ce466.
Exclusive output: go2_residual_anchored_continuation_prefix_v1_attempt_001.
Authentication and resource admission precede output creation and model replay.
No positive result, new physical outcome or navigation success is assumed.
