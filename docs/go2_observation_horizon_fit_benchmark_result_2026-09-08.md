# Observation-horizon fitting benchmark result

All eight prescribed benchmark cases completed twenty optimizer updates.
Each of the four serial/parallel pairs has exactly matching update ledgers
and final model identities. Benchmark seeds 2026091610–13 are separate from
the three scientific seeds, and none of these weights is reused.

Serial wall time was 577.673 seconds; four-worker wall time was 156.784
seconds, giving 3.684515x speedup. Maximum recorded worker RSS was
2,091,855,872 bytes, below the fixed 8-GiB per-worker limit. The prescribed
decision therefore selects four workers for the complete eighteen-fit study.

Preflight observed 82,380,779,520 bytes available RAM and 69,884,481,536 bytes
artifact space, with no competing substantial Python workload and idle GPUs.
This is a measured CPU workflow, using one numerical thread per worker.

Artifact root: `go2_observation_horizon_fit_benchmark_v1_attempt_001`, under
the established navigation development artifact root. Launch SHA-256:
`0bf938905fdaff1fbf26e468112e636cf103ad4c25028e93e1b84a5a66f632bb`.
Result SHA-256:
`67a35ae4b94ceaa09627e806f9b1f894acb20aff0b1fb664bae1eda5db6909e6`.
The benchmark binds 1,125 source paths and 44 artifacts. All 160 benchmark
updates are accounted for; scientific optimizer updates remain separate.

Eleven focused fit-plan, schedule, scoring, snapshot, all-model admission and
benchmark-decision tests passed before launch. The scientific phase must use
exactly the benchmarked source bindings, temporal contract, input identities
and fixed settings. No predictive-accuracy, native navigation, calibration,
real-time or hardware-deployment claim follows from this throughput result.
