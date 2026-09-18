# Single-pass composition launched; external profiling failure preserved

The V2 external profiler completed its child workload but failed as an attempt.
The parent reported an exit-order race, although both owned children ultimately
exited zero. The profiler separately reported 58,659 samples and 3,129 errors,
violating the prospective zero-error rule. All four owners ended. No successful
profile result or completion verification exists. Its thirteen artifacts and
source bindings are preserved in
`go2_external_profile_v2_terminal_failure_audit_2026-09-11.json`, SHA-256
`d3a959fbc92a1792319513c3d96ac7fea749464113e875b0349cf389a42d4eb8`.
No profile subset was parsed or accepted by that audit. Do not retry this
external-profiling attempt series.

The earlier verified cProfile evidence identifies the original bounds-query
implementation in the current optimized controller. The already implemented
and previously checked single-pass query index had not been composed into this
controller. The new `SinglePassBodyProjectedController` replaces eight empty
packed-owned indices with that existing index type. Insertion, retained values,
sphere arithmetic, model, selector, body projection, recovery, costs and actions
remain unchanged. The original native queue keeps its original controllers.

Validation completed before launch:

- 30 component/composition checks passed in 5.84 s, including actual public
  packets, articulated footprints, independent ownership and accumulated state.
- 16 full synthetic replay and corruption checks passed in 4.66 s.
- 21 report-reconstruction and admission checks passed in 2.68 s.
- Source/resource preflight 52403 exited zero with 2,390 bindings.

The recorded-data comparison is now active under session 51355:

- Root: `go2_single_pass_body_projected_late_history_v1_attempt_001` in the
  RecoveryStorage navigation development artifact store.
- Launch SHA-256:
  `86e325f68d0f7d5f389e911009c9ccdf1b6c3d24291e9a2c974af838ed1307d8`.
- Owner PID 2895007, creation time 1789152442.7, boot
  `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`.
- Command: `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B
  scripts/run_go2_single_pass_body_projected_late_history_v1.py`.
- Initial actual-input admission completed. The second resource gate passed
  with 68,751,794,176 available RAM bytes. Paired frame zero was emitted.

This occupies the one full CPU replay slot. Use
`python -B -m scripts.status_go2_single_pass_and_native_v1` with the existing
runtime interpreter for bounded progress and authenticated owner status. Do not
infer completion from an elapsed time or restart on an observation timeout.

After the original owner ends, preserve any failure. If a result exists with no
failure, compute its actual SHA-256 and invoke this same runner once with
`--verify-result-sha256` and that hash. It reconstructs the complete report and
all timings against another actual original-input admission before writing
`go2_single_pass_body_projected_completion_verification_2026-09-11.json`.
No automatic completion watcher was launched for this replay.

The extended-budget native worker 2867880 remained live in its audit at launch;
its root result/failure were still absent. The preliminary return-trip sensor or
model failure remains unqualified pending that audit. The new query composition
has no measured whole-controller speedup or new navigation outcome yet.
