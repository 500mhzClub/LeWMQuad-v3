# Exact completed-native input admission prepared for chained timing replay

`scripts/measured_plane_chained_full_history_inputs_development.py` binds the
original chained maze-02 launch SHA-256
`0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff`,
its exact parent argv/PID/creation time and boot, and the observed original
worker PID/creation time. Both original owners must end before any future
result is read or admitted. The checker does not wait, launch, queue, retry,
resume or modify a native attempt.

After termination it requires the actual completed result SHA-256, rejects a
failure record, verifies the complete raw artifact roster and all saved worker,
collection, readout and prefix receipts, preserves the native result's exact
scope and ancestry, and checks typed physical outcome accounting. Both a
completed negative outcome and a completed success are admissible. It calls
the frozen native worker-result checker to reconstruct the physical prefix,
then reauthenticates the complete artifact roster and new source union. It
does not rerun either controller or the complete native raw audit.

The returned input receipt supplies the actual native case, complete frame
population, original terminal and outcome, context and full-roster identities,
model identity and verification scope for the prepared paired replay adapter.
The future runner still needs its own frozen protocol, resource and CPU
scheduling checks, exclusive output creation, execution and final output
authentication. No such runner or queue was started by this preparation.

Thirty-two focused tests passed in 2.35 seconds. Synthetic fixtures cover
positive and negative completed outcomes; changed ancestry, scientific scope,
source union and typed counts; both live-owner guards before future-result
access; full-roster and receipt checks; physical-prefix checker invocation;
and final input reauthentication. Missing raw artifacts, altered saved
receipts, inconsistent worker/log bindings, a failure file and a placeholder
result hash are rejected.

The actual source preflight validated 2,636 source paths when seeded with the
new input, replay, timing and chained-composition tests. Against the actual
live native attempt, admission correctly stopped with `original chained parent
and worker must end before replay input admission`. This was an expected
read-only guard check, not a new replay attempt or a failure of the native run.

| File | SHA-256 |
| --- | --- |
| `scripts/measured_plane_chained_full_history_inputs_development.py` | `3beb6c16fe804f3dbd7c1a29cf3deb4e7b46d2d209d59ded9c2b4ef4d71b195b` |
| `lewm/tests/test_measured_plane_chained_full_history_inputs_development.py` | `5f3aa5ee83c7353a0ece6f6043d65d02f8bdbf20cd7086547974671701d0e450` |

All 2,627 sources bound to the active native launch were independently
rehashed and remain unchanged. Neither new file is in that live binding.
The original worker remains active in its raw audit, and no completed native
result has yet been admitted. These preparation checks establish no additional
navigation, timing, hardware or deployment result.
