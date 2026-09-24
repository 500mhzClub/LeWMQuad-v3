# Actual chained native preflight passed; launch invocation running

The original actual preflight, PID 2990617 / creation time 1789192394.74,
completed with exit code 0 in tool session 2555 and printed:

```text
CHAINED_NATIVE_PREFLIGHT 2627
```

This passed the launcher's completed-input admission, matched configuration,
unchanged assigned-model and current-resource checks. It created no native
output root or scene. The original owner was confirmed ended. Its exact
command and the completed predecessor identities are recorded in
`docs/go2_measured_plane_chained_waiter_completed_native_admission_started_2026-09-12.md`.

The identical command was then invoked without `--preflight-only`:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled .generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/run_go2_measured_plane_chained_maze02_v1.py --chained-wait-result-sha256 0a41c3177c2696c86d4b8d21a56ed67baba936e8999184a105f7b804462b494b
```

Original launch-invocation owner: PID 2992412, creation time 1789193084.73,
tool session 69592. At registration it was live and running, with 23.3 CPU
seconds accumulated. It was still before native output creation. This is the
single intended launch invocation, not a retry or replacement. The command
repeats completed-input admission, checks native serialization and resources,
then freezes `launch.json` and starts its fresh spawned worker. A live parent
alone does not establish that the scene has started.

Continue polling this exact original owner/session. On the
`CHAINED_NATIVE_LAUNCHED` record, authenticate its actual launch and worker
identities and monitor the original worker through collection and raw audit.
Do not launch a duplicate on a quiet log or observation timeout. Preserve any
actual terminal failure. The expected exclusive output root is
`go2_measured_plane_chained_maze02_v1_attempt_001` beneath the established
navigation development artifact base.

The intended experiment remains the frozen reused-maze 02 test of the chained
tracking repair, with the same corrected no-RGB direct model and no single-pass
timing change. The completed replay's first intervention is frame 3113. The
native raw-prefix checker must verify the actual trajectory and public packets
through that boundary and the boundary command's physical execution. The full
round-trip, visibility and settling checks remain required. No fresh native
outcome, independent-maze success, real-time or deployment claim is made here.
