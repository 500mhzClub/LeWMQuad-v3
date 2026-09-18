# Measured-plane observer history launched

The separate estimator is implemented in
`lewm/measured_plane_dual_camera_pose_development.py`, exposed through
`lewm/measured_plane_visual_motion_development.py`, and composed with the
existing planner in `lewm/measured_plane_residual_controller_development.py`.
The original and queued controllers are unchanged. The complete synthetic
controller path uses actual image inference and retains the original global
floor correction limits, memory/residual ownership and failure latch.

Validation before launch: 31 component/observer/controller tests passed in
9.49 s; 10 additional history-runner integrity/resource tests passed in 2.08 s.
The observer tests include primary/auxiliary fallback, exact reference/floor
ownership, duplicate and invalid acquisition rejection, original temporal
conflicts, ten measured bridges followed by terminal exhaustion, rejection of
preliminary witnesses after failed refinement, missing-floor support without
invented plane admission, and terminal treatment of incoherent plane evidence.

The exclusive full raw-history diagnostic is now running under the protocol
`go2_measured_plane_observer_history_v1_2026-09-11.md`.

- Root: `go2_measured_plane_observer_history_v1_attempt_001` under the fixed
  navigation development artifact root.
- Launch SHA-256: `8f09edbb77d103e3fe37e6f021da16be810a1696b588dea2264e98489f30afe1`.
- Owner PID: `2902853`; creation time: `1789156022.79`.
- Boot: `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`.
- Command: `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/replay_go2_measured_plane_observer_history_v1.py`.
- Session: `35344`.
- Bound sources: 2,427; bound input artifacts: 15,362.
- Maximum observations: 3,838, including the original floor rejection.
- Maximum compressed output: 512 MiB.

The launch has passed its frame-100 progress observation with no candidate
failure. This is progress only, not completion. The owner was verified live by
boot, PID, creation time and exact command. Do not hash or fully read a growing
output stream. Read only complete bounded progress lines while running.

After this exact owner ends, retain any `failure.json` and all partial evidence.
If `result.json` is written, authenticate its actual bytes, original launch,
source/input bindings and complete output stream before treating its report as
completed evidence. A candidate failure is a negative scientific result even
when the diagnostic runner exits normally. No automatic completion watcher or
retry was started. This run occupies the full CPU replay slot until it ends.

The native extended-budget worker remains active separately. Its persisted
prefix comparison now reports exact original public packets, physics, commands
and normalized decisions before the original budget boundary. The first
decision/command difference is at frame 3003, where the extended execution
continues. The prefix file SHA-256 is
`7bcfb97d054a32702f0f378b79b5d4368e9466344a2e232678f63c548c35cb2c`.
Native worker/root terminal admission is still pending. The existing native
queue must not advance based solely on that prefix file or this note.
