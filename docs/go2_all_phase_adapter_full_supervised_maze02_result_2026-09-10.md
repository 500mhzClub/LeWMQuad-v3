# Completed supervised adapter maze-2 episode: budget exhausted

The full supervised-rollout adapter worker completed collection and full raw
audit. Its original worker ended; the parent accepted the terminal and started
the assigned direct-model case. This is the second completed case of the
six-case batch, not completion of the batch itself. The development ledger now
contains 39 completed audited episodes and zero verified round trips.

The negative result is behavioral: the controller selected hold on 2,931 of
3,000 navigation ticks (97.7%) and left turn on the other 69. It selected no
forward or arc action. There were no observed/native arrivals, cell crossings,
distinct traversed edges or return traversal. The mission exhausted its 3,000-
tick shared budget while still outbound, about 4.6823 m from the observed goal,
then completed ten terminal zero commands. No acquisition or physical stop
occurred and no native contact was recorded.

Collection contains 3,014 observations, 3,013 completed command intervals and
151,400 physics samples. Raw sensor reconstruction, model-command replay,
command audit and strict physical visibility pass; no hard-measurement failure
frames are reported. The four-observation/900-physics-sample startup matches
the original adapter preparation, and the first changed model command completed.
The assigned supervised model state is unchanged:
`755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.

Recorded observed-position error against simulation truth over 3004
valid observations had median 0.071552 mm and maximum
0.234552 mm. This nearly stationary simulated trajectory does
not establish localization accuracy during useful traversal or on hardware.
It supports retaining the existing action-score diagnosis: the early observed
ordinary choices were dominated by charging 800-ms learned contact cost against
100-ms execution benefit. The separately prepared contact-horizon candidate
still needs its fresh-model replay and new native outcome; this result does not
establish that the candidate will navigate successfully.

Across all 3,014 observations, acquisition plus controller median time was
2699.495077 ms; p95 was
3916.368287 ms. Including receipt persistence,
median iteration time was 2751.069424 ms.
Every observation exceeded 100 ms. These are complete-episode populations,
including warmup and terminal observations; the earlier provisional navigation-
only timing subset has a different denominator. Physics paused during compute;
no real-time or real-platform qualification follows.

Independent verification checked 1,908 source bindings and all 18,120 worker
artifact bindings (session 53655), the exact assigned model identity and stored
collection/startup/readout relationships. A second check authenticated terminal,
log and parent-completion identities, reconstructed the full native evaluation
from the recorded physics trace and recomputed the complete readout (session
43384). Both exited zero. These checks did not rerun neural inference or the
full original training/coefficient verifiers; the completed original worker
owns the full raw controller audit.

Artifact root: `go2_all_phase_adapter_maze02_matched_native_v1_attempt_001`.
Case: `all_phase_full_supervised_rollout_residual_maze_02`.

- Launch: `97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a`.
- Worker terminal: `60fdf8e04413f26cf689490b92d9797ef6a94b3b4161d6d2ed46c8b32bd7b8e2`.
- Raw audit: `48b056aabdd5d888129813c8656df9bf8393e0a36ef9dc869377a096791e3f5a`.
- Startup: `a2b6f5ce81c64b2aed64279aa3e35b3b63f5c94d95104804ae5d62a802a9b64d`.
- Readout: `0f185739a5d6b6fac5e47ab5a1e1ce8fb6da876edaa3e81269bb32c96309fdc8`.
- Independent verification: `344681315ef4640772760a38cd1ee114c476dd0fc527296e0997a32c8a7360ea`,
  `go2_all_phase_adapter_full_supervised_maze02_verification_2026-09-10.json`.

The original direct-model worker is PID 2709978, creation epoch 1789053472.31.
The original contact-score raw waiter launched child PID 2709917, creation
1789053455.13, with this exact completed supervised-worker SHA. Its initial
input verification is active. The existing frontier, hold-reorientation and
contact-score native ordering remains unchanged behind the complete six-case
batch. The independent eight-layout population is still unexecuted.

The original provisional result, early score diagnosis and all frozen source
and artifact evidence remain unchanged. This completed failure adds no evidence
of reliable navigation, planning/memory advantage or hardware readiness.
