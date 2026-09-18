# Residual fallback outcome comparison ready

Implemented a separate readout for the completed residual maze-2 pilot and its
original learned maze-2 comparator. The frozen original waypoint readout requires
original nominal feasibility, so it cannot be used for the explicit corrected
feasibility exceptions. The new helper scores only the selected command's actual
100 ms native displacement, retains unsuccessful fallback attempts and censors
incomplete execution. It compares raw and causally corrected XY forecasts while
retaining original gates and requiring corrected surface, phase and all-eight-
segment nominal evidence. It does not infer unexecuted physical alternatives or
certify clearance against model error.

Verification: 13 execution/censoring/integrity tests passed in1.87s (20407 exit0).
Ten paired-admission and launch-binding tests passed in2.04s (59028 exit0).
The earlier nine-test admission run30296 also exited0; the additional regression
covers a preparation error discovered before any official readout execution.
Specifically, standalone source check77080 failed with missing input_sha256,
revealing that the proposed launch omitted required inherited verifier bindings.
The runner now retains those bindings and explicitly records zero native scenes,
no model loading and an8GiB readout allowance. Check75826 passed, followed after
the final metadata/test changes by30909 exit0: all1,679source/input/environment
bindings passed. No readout output directory or native execution was created.

Actual original-data smoke check20002 exited0. It authenticated the fixed learned
result a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720,
then checked the bound original maze2 physics trace, command tape, compressed
decision stream and raw audit before and after summarization. The helper reports
514observations, first terminal503
NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS, no fallback
attempts, no verified round trip and strict visibility PASS. Original native
path1.5037422533395475m; minimum outbound-goal distance3.946789928854942m;
duration51.3simulatedseconds. Receipt-inclusive median962.827517ms, maximum
1308.114697ms, all514over100ms. Receipt-write median18.7403455ms; controller/
observation median910.640801ms. This validates actual original input compatibility
and gives the preserved baseline; it is not the completed paired readout.

Before that CPU analysis:75,889,397,760bytes available RAM,83,034,927,104artifact
free bytes,16physical/32logical CPUs,all32affinity,CPU3.4%,GPUs0%. Residual
worker RSS7,160,795,136bytes. The8GiB CPU-only analysis fits current headroom;
its eventual full launcher refreshes resources.

Final source SHA-256 identities:

- `lewm/residual_first_interval_execution_readout_development.py`:
  8a982566ac4d6c012208c55be97c2b6874bddc75f8c591ff8bf68a2666a011c7
- `scripts/read_go2_residual_first_interval_maze_pilot_v1.py`:
  cf34eb6efceef6f0ffbc0a95f1664c31b56dcfb72ba5c71614c93035d6667e02
- `lewm/tests/test_residual_first_interval_execution_readout_development.py`:
  cef7fe3873979fbddfee35464ca9301d2dfb898b280d1c573019c8ba30cae96a
- `lewm/tests/test_residual_first_interval_readout_admission_development.py`:
  0dc90eff6bdbd8c48599e3cc429dae4d9a0b6c5dc87edf07b2990c567285df62
- `docs/go2_residual_first_interval_maze_readout_v1_2026-09-09.md`:
  723d4af18bd62919f50f1f18a67c0a01debe00ac3b3ba9406b7dae8793520cf3

Await the existing native pilot's complete final result; use its actual SHA
with --native-result-sha256. Do not run against a live or partial result. The
readout can overlap the queued tracking native job after resource assessment.
No new navigation, independent-layout, timing or hardware success is claimed.
