# Fitting benchmark: exact numerical match, four workers selected

All eight separate20-update full-JEPA benchmark runs completed. The four fixed
seeds2026091010–2026091013 were each started fresh once serially and once in
four spawned workers. Every paired update ledger and final model-state identity
matched exactly. No benchmark weights were saved as eligible checkpoints or
resumed for the scientific fits.

Serial phase wall time was231.956s; four-worker time was65.145s, a3.5606×
speedup. These measured phases include data admission, loading and receipt
verification, as well as fitting. Maximum parallel-worker peak RSS was
1,889,124,352 bytes, below the8GiB diagnostic allowance. The frozen decision
rule therefore selected four workers for the subsequent six fresh1200-update
fits. The benchmark does not measure GPU training or real-time control.

The hardware preflight found32 logical/16 physical CPUs,82.25GB available RAM,
96.55GB artifact storage,0.4% CPU activity, idle GPUs and no substantial
competing Python task. Each worker used one CPU/OpenCV/Torch/BLAS thread and
deterministic Torch algorithms. Resource snapshots were retained throughout.
No physics or native commands occurred.

Four focused tests passed before the benchmark: exact family-plan validation,
retention/latching after a receipt failure following an optimizer step,
complete raw-score/censoring accounting, and the fixed benchmark equality,
speed and memory decision. The complete real-data comparison independently
confirmed identical update records under the two execution schedules.

Root under the established navigation artifact base:
`go2_family_transition_fit_benchmark_v1_attempt_001`.

| Receipt | SHA-256 |
| --- | --- |
| launch.json | ce526b6271c2c59ebbfc27582aa4075abbe74e304264e30f687481e1d14f383b |
| result.json | 56774504c3296f9551951260e4392b43abdae08767c4906915ad2606f0952a4f |

The terminal binds42 artifacts totaling10,132,937 bytes and856 source files.
The scientific fit phase requires these exact source/scientific settings and
recomputes the benchmark decision before launch. The benchmark process exited0.
All prior source, measurement, task-design, observer and navigation failures
remain unchanged. The full navigation goal remains active.
