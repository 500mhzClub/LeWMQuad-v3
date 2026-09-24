# Approved storage consolidation completed; comparison gate passes

User explicitly approved the reviewed operation with 'do it'. Exact scope and
source identities are retained in
docs/go2_training_artifact_hardlink_authorization_2026-09-09.json.

Runner30695/PID2528150 exited0. Completed result:
7f0b0ff8cc7ace44a2542738107f8dd3a7b6ef454b942e240c51ad8d80913297,
under go2_training_artifact_hardlink_consolidation_v1_attempt_001. All27,210
approved duplicate paths were replaced with hard links in5,523 groups. Every
original path and byte remains available. The runner reverified every original
artifact binding in all three completed training results, their result identities
and source bindings. No pip/GSD or unlisted cleanup occurred.

Independent42200 exited0: operation source/output hashes match; all27,210 intent
records have an ordered matching completion record, with no unfinished intent.
Independent46398 exited0: all32,733 affected names share exactly their reviewed
canonical inode, group link counts match, and modes/ownership/sizes are preserved.
Allocated storage for these files fell from9,692,139,520 to2,472,865,792 bytes:
verified recovery7,219,273,728 bytes, or6.723472595214844 GiB. All linked artifacts
must remain immutable; any later intended content change requires separate
storage before editing.

The operation's observed filesystem free-space change was507,672,141,824 bytes,
far greater than the verified allocation saving. That volume-wide change is not
attributed to this operation. At gate check42200 the artifact volume reported
709,855,936,512 free bytes. The original supervised runner's resources_for(...,3)
passed with its unchanged78,383,153,152-byte requirement (73 GiB),32 GiB RAM
allowance and one native worker. Available RAM71,053,438,976 bytes;16 physical/
32 logical CPUs, all affinity, CPU3.3%, card0 GPU0%, card1 GPU8%. No other native
or replay worker was running.

Submitted original frozen run_go2_supervised_rollout_mazes_v1.py with learned
cohort SHA a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720.
Handle19047/PID2534319 is live in input authentication, before physical collection.
It will use the fixed original layouts1,2,3 order, fresh assigned supervised
models and unchanged raw audits. No comparison outcome is claimed yet.
Storage blocker resolved; preserve this same process and every outcome.
