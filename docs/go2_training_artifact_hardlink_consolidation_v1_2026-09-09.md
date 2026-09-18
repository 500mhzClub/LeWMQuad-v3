# Exact training-artifact storage consolidation V1

The completed read-only inventory result is
732b0a95a98d598843e1d56f85612093b3d78e3e1ed145f4e27d24f2ceb6d577,
under go2_training_artifact_duplicate_inventory_v1_attempt_001. It freshly
verified32,733 paths against their original artifact hashes:5,523 canonical
copies and27,210 duplicates, with7,219,273,728 bytes of duplicate allocation.
Every candidate belongs to an exact declared artifact binding in the three
completed training roots listed in that inventory. Nothing has been removed.

The proposed operation retains each canonical file and atomically replaces each
listed duplicate with a hard link to the canonical on the same volume. Every
original path and byte remains readable. No training data regeneration, changed
scientific content, directory removal, checkpoint removal or relaxed verifier
is involved. Files will share an inode and must remain immutable: later in-place
edits to one linked path would affect its other names. Inode/link-count/change
time metadata changes are intrinsic to the operation; original artifact byte
identities and modes/ownership are retained.

Execute only after explicit user approval bound to this inventory, the reviewed
runner/test/protocol hashes and exclusive output root
go2_training_artifact_hardlink_consolidation_v1_attempt_001. The required local
authorization record is docs/go2_training_artifact_hardlink_authorization_2026-09-09.json.
It does not exist merely because this proposal exists. --preflight-only requires
no approval, writes no output and changes no input. No execution is implied by
the user's earlier questions about storage options.

Require all known owned development artifact consumers to have finished before
conversion. Inspect owned command lines and accessible open files/mappings;
reject another development interpreter or a visible training-artifact user.
Reject failures to establish process ownership or command identity. Record
descriptor-inspection denial for otherwise identified unrelated processes
(for example the protected user systemd service); do not claim universal
process quiescence. Recheck every reviewed path's hash and original metadata.
This operational check is not an OS lock against other or subsequently started
processes. Do not launch any collection/replay/admission during conversion.
No frozen source or running experiment definition changes.

Immediately before each replacement, rehash the canonical and duplicate and
check their expected identities. Require same device/mode/ownership/size and
single-link original duplicate. Fsync an intent journal, create a uniquely named
temporary hard link in that duplicate's directory, verify its inode/hash and
unchanged target, then atomically replace the target. Fsync the directory and
completion journal. Never unlink an unrelated temporary file. A failure keeps
the exact partial journal and temporary links, stops, and permits no automatic
retry. All original paths remain available even at the rename boundary.

After conversion, verify every original artifact binding in all three completed
results, every original result identity and source binding, and the operation's
source hashes. Record observed free-space change separately from predicted
allocated savings, because unrelated filesystem usage can change concurrently.
Keep the40 GiB artifact reserve and128 MiB transaction-evidence allowance.
Refresh the fixed supervised cohort's original73 GiB resource gate afterward;
do not lower it or reorder its three cases to claim the blocker resolved.

This exact storage proposal includes no pip-cache cleanup, GSD cleanup, source
export, sealed access, replay, simulation, retraining or deployment authority.
